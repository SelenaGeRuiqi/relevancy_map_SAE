import os
import argparse
import sys
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPAttention

sys.path.append('../models')
from sae import SAE

def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def overlay_relevancy_on_image(image: Image.Image, relevancy_map: np.ndarray, out_path: str, out_size: int = 224):
    heat = cv2.resize(relevancy_map.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image.resize((out_size, out_size)))
    ax.imshow(heat, cmap='hot', alpha=0.5)
    ax.axis('off')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

class ExplainableCLIPAttention(CLIPAttention):
    """
    基于Hila Chefer方法的可解释CLIP Attention模块
    """
    def __init__(self, original_attention):
        super().__init__(original_attention.config)
        
        # 复制原始权重
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj  
        self.q_proj = original_attention.q_proj
        self.out_proj = original_attention.out_proj
        self.dropout = original_attention.dropout
        self.config = original_attention.config
        
        # 添加explainability属性
        self.attn_probs = None
        self.attn_grad = None
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """从原始CLIPAttention复制_shape方法"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        """重写forward方法以捕获attention权重"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        # 计算query, key, value
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}")

        # 应用attention mask
        if causal_attention_mask is not None:
            causal_attention_mask = causal_attention_mask.to(attn_weights.dtype)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            attention_mask = attention_mask.to(attn_weights.dtype)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        # 🔥 关键：存储attention probabilities并设置梯度hook
        self.attn_probs = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if self.attn_probs.requires_grad:
            self.attn_probs.register_hook(self._save_attn_gradients)

        # dropout
        if self.training:
            attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=True)

        # 应用attention到value
        attn_output = torch.bmm(attn_weights, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}")

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (self.attn_probs,)

        return outputs

    def _save_attn_gradients(self, grad):
        """保存attention梯度的hook函数"""
        self.attn_grad = grad

def modify_clip_for_explainability(model):
    """
    修改CLIP模型以支持explainability，基于Hila Chefer的方法
    """
    # 修改vision model的attention层
    for i, layer in enumerate(model.vision_model.encoder.layers):
        original_attention = layer.self_attn
        # 创建新的explainable attention层
        explainable_attention = ExplainableCLIPAttention(original_attention)
        # 替换原始attention层
        layer.self_attn = explainable_attention
        print(f"[DEBUG] Modified layer {i} attention")
    
    return model

def compute_rollout(image_attn_blocks, discard_ratio=0.9):
    """
    基于Hila Chefer方法的attention rollout计算
    """
    if len(image_attn_blocks) == 0:
        raise ValueError("No attention blocks provided!")
    
    # 获取第一个attention block的形状信息
    first_block = image_attn_blocks[0]
    if not hasattr(first_block, 'attn_probs') or first_block.attn_probs is None:
        raise ValueError("No attention probabilities found in blocks!")
    
    seq_len = first_block.attn_probs.size(-1)
    device = first_block.attn_probs.device
    
    result = torch.eye(seq_len, device=device)
    
    with torch.no_grad():
        for i, blk in enumerate(image_attn_blocks):
            if hasattr(blk, 'attn_probs') and blk.attn_probs is not None:
                # 🔥 关键：gradient * attention
                cam = blk.attn_probs
                grad = blk.attn_grad if hasattr(blk, 'attn_grad') else None
                
                # 取第一个batch，跨head平均
                cam = cam[0].mean(dim=0)  # [seq_len, seq_len]
                grad = grad[0].mean(dim=0) if grad is not None else None
                
                if grad is not None:
                    # Gradient-weighted attention
                    cam = cam * grad
                    cam = torch.clamp(cam, min=0)  # 只保留positive
                
                # 丢弃最低的attention但保持CLS token
                flat = cam.view(-1)
                _, indices = flat.topk(int(flat.size(0) * discard_ratio), largest=False)
                flat[indices] = 0
                cam = flat.view(cam.size())
                
                # 添加residual connection
                I = torch.eye(cam.size(0)).to(device)
                cam = cam + I
                cam = cam / cam.sum(dim=-1, keepdim=True)  # 归一化
                
                result = torch.matmul(cam, result)
    
    # 提取CLS token到patches的attention
    mask = result[0, 1:]  # [num_patches]
    width = int(mask.size(0) ** 0.5)
    mask = mask.reshape(width, width)
    
    return mask.cpu().numpy()

class CLIPMSAERelevancy:
    def __init__(
        self,
        clip_model_name: str,
        sae_ckpt_path: str,
        device: str = "cuda",
        centered_mean_npy: Optional[str] = None,
        concept_scores_npy: Optional[str] = None,
        vocab_txt: Optional[str] = None,
        verbose: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.verbose = verbose

        # --- Load CLIP ---
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # 🔥 关键：修改CLIP模型以支持explainability
        self.clip_model = modify_clip_for_explainability(self.clip_model)
        
        # 强制使用eager attention
        if hasattr(self.clip_model.vision_model.config, "attn_implementation"):
            self.clip_model.vision_model.config.attn_implementation = "eager"
        
        self.clip_model.eval()

        # --- Load SAE ---
        self.sae = SAE(sae_ckpt_path)
        if hasattr(self.sae, "to"):
            self.sae = self.sae.to(self.device)

        # --- Optional centered mean ---
        self.clip_mean_768 = None
        if centered_mean_npy is not None and os.path.exists(centered_mean_npy):
            self.clip_mean_768 = np.load(centered_mean_npy)
            print(f"Loaded centered mean: {centered_mean_npy}")

        # --- Optional concept naming artifacts ---
        self.concept_scores = None
        self.vocab = None
        if concept_scores_npy is not None and os.path.exists(concept_scores_npy):
            self.concept_scores = np.load(concept_scores_npy)
            print(f"Loaded concept_scores: {concept_scores_npy} with shape {self.concept_scores.shape}")
        if vocab_txt is not None and os.path.exists(vocab_txt):
            with open(vocab_txt, "r") as f:
                self.vocab = [ln.strip() for ln in f.readlines()]
            print(f"Loaded vocab of size {len(self.vocab)} from {vocab_txt}")
        
    def concept_relevancy_map(
        self,
        pil_img: Image.Image,
        neuron_id: Optional[int] = None,
        vocab_concept_id: Optional[int] = None,
        topk_neurons_for_concept: int = 10,
        only_positive_weights: bool = True,
        discard_ratio: float = 0.9,
    ) -> Tuple[np.ndarray, dict]:
        """
        生成concept relevancy map，基于Hila Chefer的gradient attention rollout
        """
        assert (neuron_id is not None) ^ (vocab_concept_id is not None), \
            "Provide exactly one of neuron_id or vocab_concept_id."

        # 🔥 关键：设置requires_grad和train模式
        self.clip_model.train()  # 需要梯度
        if hasattr(self.sae, 'train'):
            self.sae.train()

        try:
            # 预处理图像
            inputs = self.clip_processor(images=pil_img, return_tensors="pt").to(self.device)
            inputs['pixel_values'].requires_grad_(True)

            # 🔥 前向传播获取CLIP特征
            vision_outputs = self.clip_model.vision_model(
                pixel_values=inputs['pixel_values'],
                output_attentions=True
            )
            
            # 获取pooled feature
            pooled_output = vision_outputs.pooler_output
            img_768 = self.clip_model.visual_projection(pooled_output)
            
            # 应用centered mean
            if self.clip_mean_768 is not None:
                mean = torch.as_tensor(self.clip_mean_768, device=img_768.device, dtype=img_768.dtype)
                img_768 = img_768 - mean

            # 🔥 SAE编码
            z = self.sae.encode(img_768)
            if isinstance(z, tuple):
                z = z[0]

            # 获取attention blocks
            image_attn_blocks = []
            for layer in self.clip_model.vision_model.encoder.layers:
                if hasattr(layer.self_attn, 'attn_probs') and layer.self_attn.attn_probs is not None:
                    image_attn_blocks.append(layer.self_attn)

            if self.verbose:
                print(f"[DEBUG] CLIP feature shape: {img_768.shape}")
                print(f"[DEBUG] SAE output shape: {z.shape}")
                print(f"[DEBUG] Number of attention blocks: {len(image_attn_blocks)}")

            # 构建目标标量
            info = {}
            if neuron_id is not None:
                assert 0 <= neuron_id < z.shape[1], f"neuron_id {neuron_id} out of range [0, {z.shape[1]})"
                y = z[0, neuron_id]
                info["target_mode"] = "single_neuron"
                info["neuron_id"] = int(neuron_id)
            else:
                assert self.concept_scores is not None, "concept_scores required for vocab_concept_id mode"
                assert 0 <= vocab_concept_id < self.concept_scores.shape[0], \
                    f"vocab_concept_id {vocab_concept_id} out of range [0, {self.concept_scores.shape[0]})"
                
                # 获取概念权重
                concept_weights = self.concept_scores[vocab_concept_id, :].copy()
                if only_positive_weights:
                    concept_weights = np.maximum(concept_weights, 0)
                
                concept_weights_tensor = torch.from_numpy(concept_weights).to(z.device).float()
                
                # 计算加权激活
                z_batch = z[0]
                weighted_activations = z_batch * concept_weights_tensor
                
                # Top-k选择
                if topk_neurons_for_concept > 0:
                    values, indices = torch.topk(
                        torch.abs(weighted_activations), 
                        k=min(topk_neurons_for_concept, weighted_activations.shape[0])
                    )
                    mask = torch.zeros_like(weighted_activations)
                    mask[indices] = 1.0
                    y = (weighted_activations * mask).sum()
                else:
                    y = weighted_activations.sum()
                
                info["target_mode"] = "vocab_concept"
                info["vocab_concept_id"] = int(vocab_concept_id)
                info["topk_neurons"] = int(topk_neurons_for_concept)
                if self.vocab and vocab_concept_id < len(self.vocab):
                    info["vocab_token"] = self.vocab[vocab_concept_id]

            if self.verbose:
                print(f"[DEBUG] Target scalar y = {y.item():.6f}")

            # 🔥 关键：梯度反向传播
            self.clip_model.zero_grad()
            if hasattr(self.sae, "zero_grad"):
                self.sae.zero_grad()
            
            y.backward(retain_graph=True)

            # 检查attention和梯度
            valid_blocks = 0
            for i, blk in enumerate(image_attn_blocks):
                if hasattr(blk, 'attn_probs') and blk.attn_probs is not None:
                    valid_blocks += 1
                    if self.verbose and i < 3:
                        has_grad = blk.attn_grad is not None
                        print(f"[DEBUG] Block {i}: attn_probs shape={blk.attn_probs.shape}, has_grad={has_grad}")

            if valid_blocks == 0:
                raise RuntimeError("No attention blocks have attention probabilities! Check model modification.")

            # 计算rollout
            relevance_map = compute_rollout(
                [blk for blk in image_attn_blocks if hasattr(blk, 'attn_probs') and blk.attn_probs is not None],
                discard_ratio=discard_ratio
            )
            
            # 归一化
            if relevance_map.max() > relevance_map.min():
                relevance_map = (relevance_map - relevance_map.min()) / (relevance_map.max() - relevance_map.min())
            
            info["rollout_size"] = relevance_map.shape
            info["relevance_range"] = (float(relevance_map.min()), float(relevance_map.max()))
            info["num_attention_blocks"] = valid_blocks
            
            return relevance_map, info
            
        finally:
            # 清理
            self.clip_model.eval()
            if hasattr(self.sae, 'eval'):
                self.sae.eval()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./outputs")

    # models
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    ap.add_argument("--sae_ckpt", type=str, required=True)

    # centered mean (optional)
    ap.add_argument("--centered_mean_npy", type=str, default=None)

    # concept naming artifacts
    ap.add_argument("--concept_scores_npy", type=str, default=None)
    ap.add_argument("--vocab_txt", type=str, default=None)

    # target choice
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--neuron_id", type=int, default=None)
    group.add_argument("--vocab_concept_id", type=int, default=None)

    ap.add_argument("--topk_neurons_for_concept", type=int, default=10)
    ap.add_argument("--only_positive_weights", action="store_true", default=True)
    ap.add_argument("--discard_ratio", type=float, default=0.9)
    ap.add_argument("--verbose", action="store_true")

    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    engine = CLIPMSAERelevancy(
        clip_model_name=args.clip_model,
        sae_ckpt_path=args.sae_ckpt,
        device="cuda",
        centered_mean_npy=args.centered_mean_npy,
        concept_scores_npy=args.concept_scores_npy,
        vocab_txt=args.vocab_txt,
        verbose=args.verbose,
    )

    img = load_image(args.image)

    if args.neuron_id is not None:
        rel, info = engine.concept_relevancy_map(
            img, 
            neuron_id=args.neuron_id,
            discard_ratio=args.discard_ratio
        )
        tag = f"neuron_{args.neuron_id}"
    else:
        rel, info = engine.concept_relevancy_map(
            img,
            vocab_concept_id=args.vocab_concept_id,
            topk_neurons_for_concept=args.topk_neurons_for_concept,
            only_positive_weights=args.only_positive_weights,
            discard_ratio=args.discard_ratio
        )
        tok = info.get("vocab_token", str(args.vocab_concept_id))
        tag = f"vocab_{args.vocab_concept_id}_{tok}_k{args.topk_neurons_for_concept}"

    out_path = os.path.join(args.out_dir, f"relmap_{tag}.png")
    overlay_relevancy_on_image(img, rel, out_path=out_path)

    print("=== RESULTS ===")
    for k, v in info.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()