import os
import argparse
import sys
from typing import Optional, Tuple, List

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from transformers import CLIPModel, CLIPProcessor

sys.path.append('../models')
from sae import SAE
# ======================================================

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
        if hasattr(self.clip_model.vision_model.config, "attn_implementation"):
            self.clip_model.vision_model.config.attn_implementation = "eager"
        self.clip_model.vision_model.config.return_dict = True
        self.clip_model.config.return_dict = True
        self.clip_model.eval()

        # --- Load SAE ---
        self.sae = SAE(sae_ckpt_path)
        if hasattr(self.sae, "to"):
            self.sae = self.sae.to(self.device)

        # --- Optional centered mean (for centered checkpoints) ---
        self.clip_mean_768 = None
        if centered_mean_npy is not None and os.path.exists(centered_mean_npy):
            self.clip_mean_768 = np.load(centered_mean_npy)  # shape [768]
            print(f"ℹ️ Loaded centered mean: {centered_mean_npy}")

        # --- Optional concept naming artifacts ---
        self.concept_scores = None
        self.vocab = None
        if concept_scores_npy is not None and os.path.exists(concept_scores_npy):
            self.concept_scores = np.load(concept_scores_npy)  # [num_neurons, vocab_size]
            print(f"ℹ️ Loaded concept_scores: {concept_scores_npy} with shape {self.concept_scores.shape}")
        if vocab_txt is not None and os.path.exists(vocab_txt):
            with open(vocab_txt, "r") as f:
                self.vocab = [ln.strip() for ln in f.readlines()]
            print(f"ℹ️ Loaded vocab of size {len(self.vocab)} from {vocab_txt}")
        self._register_attnprob_hooks()
        
        # ======================= START OF FINAL FIX =======================
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!             正在应用最终修复：替换激活函数           !!!")
        # 导入 sae.py 中的 TopK 类
        from sae import TopK
        # 将原来模型的激活函数替换为 TopKabsReLU
        # k=64, use_abs=True (按绝对值选), act_fn=nn.ReLU()
        self.sae.model.activation = TopK(k=64, act_fn=torch.nn.ReLU(), use_abs=True)
        print("!!!         SAE 激活函数已被强制替换为 TopKabsReLU        !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
# ======================== END OF FINAL FIX ========================

    def _register_attnprob_hooks(self):
        """
        在 CLIP 的每一层 Attention 的 dropout 上挂 forward_hook，
        从 dropout 的输入里拿到 softmax 后的 attn_probs（带 grad 的那个）。
        确保每个 attention block 只被 hook 一次，避免重复收集。
        """
        self._attn_probs = []  # 存每层的 [B,H,S,S]
        self._hooked_modules = set()  # 记录已经 hook 过的模块

        def grab_from_attn_forward(module, inputs, output):
            # CLIPAttention.forward 通常返回 (attn_output, attn_weights) 或 (attn_output,)
            attn_weights = None
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
            if attn_weights is not None:
                attn_weights.retain_grad()
                self._attn_probs.append(attn_weights)
                if self.verbose:
                    print(f"[DEBUG] Hooked attn_weights from CLIPAttention.forward: {attn_weights.shape}")

        def grab_from_dropout(module, inputs, output):
            # dropout 的输入就是 softmax 后的概率
            A = inputs[0]
            A.retain_grad()
            self._attn_probs.append(A)
            if self.verbose:
                print(f"[DEBUG] Hooked attn_probs from dropout: {A.shape}")

        import torch.nn as nn
        try:
            from transformers.models.clip.modeling_clip import CLIPAttention
        except Exception:
            from transformers.models.clip.modeling_clip import CLIPEncoderLayer
            CLIPAttention = None

        # 遍历所有模块，确保每个 attention block 只被 hook 一次
        for name, m in self.clip_model.vision_model.named_modules():
            if CLIPAttention is not None and isinstance(m, CLIPAttention):
                if name not in self._hooked_modules:
                    m.register_forward_hook(grab_from_attn_forward)
                    self._hooked_modules.add(name)
                    if self.verbose:
                        print(f"[DEBUG] Registered hook on CLIPAttention: {name}")
            
            # 只有在没有成功 hook 到 CLIPAttention 时，才用 dropout 作备胎
            elif (name not in self._hooked_modules and 
                  hasattr(m, 'dropout') and 
                  isinstance(getattr(m, 'dropout'), nn.Dropout) and
                  'attention' in name):  # 确保是 attention 相关的 dropout
                m.dropout.register_forward_hook(grab_from_dropout)
                self._hooked_modules.add(name)
                if self.verbose:
                    print(f"[DEBUG] Registered hook on dropout: {name}")

        if self.verbose:
            print(f"[DEBUG] Total hooked modules: {len(self._hooked_modules)}")


    def _clip_image_768(self, pil_img: Image.Image) -> torch.Tensor:
        """
        Get CLIP 768-d image embedding WITHOUT L2-norm.
        If centered mean is provided, subtract it (centered checkpoints).
        """
        inputs = self.clip_processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            vis_out = self.clip_model.vision_model(
                inputs["pixel_values"],
                output_attentions=True
            )
            pooled = vis_out.pooler_output if hasattr(vis_out, 'pooler_output') else vis_out[1]  # [B, hidden]
            img_768 = self.clip_model.visual_projection(pooled)          # [B, 768]

        if self.clip_mean_768 is not None:
            mean = torch.as_tensor(self.clip_mean_768, device=img_768.device, dtype=img_768.dtype)
            img_768 = img_768 - mean

        return img_768, vis_out  # return attentions via vis_out

    @staticmethod
    def _gae_rollout_from_attentions(attentions: List[torch.Tensor]) -> torch.Tensor:
        """
        attentions: list of [B, H, S, S] tensors, each with .grad available after backward().
        Returns R: [S, S] rollout relevance matrix (batch index 0).
        """
        # Compute A_bar = mean_h( ReLU(grad * A) ) per layer
        A_bars = []
        for A in attentions:
            A.retain_grad()  # ensure grad retained
        # grads exist after backward
        for A in attentions:
            A_bar = (A.grad * A).clamp_min(0).mean(dim=1)[0]  # -> [S, S], take batch 0
            A_bars.append(A_bar)

        S = A_bars[0].shape[-1]
        R = torch.eye(S, device=A_bars[0].device)
        for A_bar in A_bars:
            R = R + A_bar @ R
        return R  # [S, S]

    def concept_relevancy_map(
        self,
        pil_img: Image.Image,
        neuron_id: Optional[int] = None,
        vocab_concept_id: Optional[int] = None,
        topk_neurons_for_concept: int = 10,
        only_positive_weights: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute concept->pixel relevancy using GAE on CLIP attentions.
        Target scalar y is:
        - if neuron_id is given: y = z[neuron_id]
        - elif vocab_concept_id is given: y = sum_k w_k * z[k], where w = concept_scores[:, concept_id]
            (optionally top-k positive weights)
        Returns (14x14 map, info_dict)
        """
        assert (neuron_id is not None) ^ (vocab_concept_id is not None), \
            "Provide exactly one of neuron_id or vocab_concept_id."

        self._attn_probs.clear()

        # Forward to get attentions and 768 features
        inputs = self.clip_processor(images=pil_img, return_tensors="pt").to(self.device)
        vis_out = self.clip_model.vision_model(
            inputs["pixel_values"],
            output_attentions=True
        )

        pooled = vis_out.pooler_output if hasattr(vis_out, 'pooler_output') else vis_out[1]  # [B, hidden]
        img_768 = self.clip_model.visual_projection(pooled)  # [B, 768]
        if self.clip_mean_768 is not None:
            mean = torch.as_tensor(self.clip_mean_768, device=img_768.device, dtype=img_768.dtype)
            img_768 = img_768 - mean

        # SAE encode -> z
        z = self.sae.encode(img_768)
        if isinstance(z, tuple):
            z = z[0]                      # [B, num_neurons]
        assert z.dim() == 2 and z.shape[0] == 1, "Expect batch size 1"

        # Build target scalar y
        info = {}
            
        # ======================= START OF ULTIMATE SANITY CHECK =======================
        # !!! 注意：此代码块仅用于调试 !!!
        # 我们暂时绕过了原来计算 'y' 的复杂逻辑，
        # 强制 'y' 来自一个写死的神经元，以此来测试梯度回传和热力图生成部分是否正常工作。

        # 第1步: 使用当前值运行一次脚本。
        # 第2步: 将下面的数字改成另一个不同的值 (例如 500)，然后再次运行脚本。
        # 第3步: 比较两次生成的图片。如果代码的下游部分没问题，这两张图必须不同。
        target_neuron_idx = 100

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!                正在进入调试模式                    !!!")
        print(f"!!!  强制目标 y = 单个神经元的激活值: neuron {target_neuron_idx}   !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # 确保我们选择的神经元索引是有效的
        assert 0 <= target_neuron_idx < z.shape[1], f"调试用的神经元ID {target_neuron_idx} 超出范围。"

        # 强制 'y' 等于这单个神经元的激活值
        y = z[:, target_neuron_idx].sum()

        # 为调试运行填充 info 字典，以生成清晰的文件名
        info['target_mode'] = f'forced_debug_neuron_{target_neuron_idx}'
        info['vocab_concept_id'] = -1 # 文件名占位符
        if self.vocab is not None:
            info['vocab_token'] = f'NEURON_{target_neuron_idx}' # 文件名占位符

        # --- 原来的逻辑已被上面的调试代码块绕过。 ---
        # --- 如果您想恢复正常逻辑，请删除或注释掉上面的调试块，并取消下面原始代码的注释。---
        # ======================== END OF ULTIMATE SANITY CHECK ========================

        # if neuron_id is not None:
        #     assert 0 <= neuron_id < z.shape[1], f"neuron_id out of range [0,{z.shape[1]-1}]"
        #     y = z[:, neuron_id].sum()
        #     info["target_mode"] = "single_neuron"
        #     info["neuron_id"] = int(neuron_id)
        # else:
        #     assert self.concept_scores is not None, "concept_scores.npy required for vocab_concept_id mode"
        #     assert 0 <= vocab_concept_id < self.concept_scores.shape[0], \
        #         f"vocab_concept_id out of range [0,{self.concept_scores.shape[0]-1}]"
        #     w = self.concept_scores[vocab_concept_id, :].copy()
        #     if only_positive_weights:
        #         w[w < 0] = 0.0
        #     w_t = torch.as_tensor(w, device=z.device, dtype=z.dtype).unsqueeze(0)
        #     z_pos = torch.relu(z)
        #     contrib = z_pos * w_t
        #     num_active_neurons = torch.count_nonzero(contrib)
        #     k = min(topk_neurons_for_concept, num_active_neurons)
        #     if k == 0:
        #         print("WARNING: No active neurons for this concept and image. Cannot generate relevancy map.")
        #         side = int(round((self.clip_model.vision_model.config.num_patches) ** 0.5))
        #         return np.zeros((side, side)), {"error": "No active neurons", "topk_neurons_used": 0}
        #     info["topk_neurons_used"] = int(k)
        #     vals, idx = torch.topk(contrib, k=k, dim=1)
        #     print(f"DEBUG: For concept ID {vocab_concept_id}, ACTUALLY using top-{k}. Neuron indices are: {idx.cpu().numpy()}")
        #     mask = torch.zeros_like(contrib)
        #     mask.scatter_(1, idx, 1.0)
        #     y = (contrib * mask).sum()
        #     info["target_mode"] = "vocab_linear_combo_gated"
        #     info["vocab_concept_id"] = int(vocab_concept_id)
        #     if self.vocab is not None and 0 <= vocab_concept_id < len(self.vocab):
        #         info["vocab_token"] = self.vocab[vocab_concept_id]
        #     info["topk_neurons"] = int(k)

        # Backprop to attentions
        self.clip_model.zero_grad(set_to_none=True)
        if hasattr(self.sae, "zero_grad"):
            self.sae.zero_grad(set_to_none=True)
        y.backward()

        if self.verbose:
            print(f"[DEBUG] captured layers: {len(self._attn_probs)}")
        for i, A in enumerate(self._attn_probs[-12:]): # Use last 12 layers for ViT-L
            g = A.grad
            if self.verbose:
                print(f"[DEBUG] L{-len(self._attn_probs)+i+1}: shape={tuple(A.shape)}, "
                f"grad_none={g is None}, grad_norm={(g.float().norm().item() if g is not None else 0):.3e}")

        # GAE rollout
        if len(self._attn_probs) == 0:
            raise RuntimeError("No attn_probs were captured by hooks.")

        att_list = self._attn_probs
        if self.verbose:
            print(f"[DEBUG] Using all {len(att_list)} layers for GAE rollout")
        R = self._gae_rollout_from_attentions(att_list)

        r_cls = R[0]
        r_patches = r_cls[1:]

        S = R.shape[0]
        num_patches = S - 1
        side = int(round(num_patches ** 0.5))
        if side * side != num_patches:
            side = int(np.sqrt(num_patches))
        rel = r_patches[: side * side].reshape(side, side)
        rel = rel.detach().cpu().float().numpy()
        mn, mx = rel.min(), rel.max()
        rel = (rel - mn) / (mx - mn + 1e-8)

        info["rollout_size"] = (side, side)
        return rel, info


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./outputs")

    # models
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    ap.add_argument("--sae_ckpt", type=str, required=True)

    # centered mean (optional)
    ap.add_argument("--centered_mean_npy", type=str, default=None)

    # concept naming artifacts (optional, for vocab mode)
    ap.add_argument("--concept_scores_npy", type=str, default=None)
    ap.add_argument("--vocab_txt", type=str, default=None)

    # target choice
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--neuron_id", type=int, default=None)
    group.add_argument("--vocab_concept_id", type=int, default=None)

    ap.add_argument("--topk_neurons_for_concept", type=int, default=10)
    ap.add_argument("--only_positive_weights", action="store_true", default=True)
    ap.add_argument("--verbose", action="store_true", help="Enable verbose debug output")

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
            img, neuron_id=args.neuron_id
        )
        tag = f"neuron_{args.neuron_id}"
    else:
        rel, info = engine.concept_relevancy_map(
            img,
            vocab_concept_id=args.vocab_concept_id,
            topk_neurons_for_concept=args.topk_neurons_for_concept,
            only_positive_weights=args.only_positive_weights,
        )
        tok = info.get("vocab_token", str(args.vocab_concept_id))
        tag = f"vocab_{args.vocab_concept_id}_{tok}"

    out_path = os.path.join(args.out_dir, f"relmap_{tag}.png")
    overlay_relevancy_on_image(img, rel, out_path=out_path)

    print("=== INFO ===")
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
