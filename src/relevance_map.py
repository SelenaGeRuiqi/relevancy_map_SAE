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

        #--- Optional centered mean (for centered checkpoints) ---
        self.clip_mean_768 = None
        if centered_mean_npy is not None and os.path.exists(centered_mean_npy):
            self.clip_mean_768 = np.load(centered_mean_npy)  # shape [768]
            print(f"Loaded centered mean: {centered_mean_npy}")

        # --- Optional concept naming artifacts ---
        self.concept_scores = None
        self.vocab = None
        if concept_scores_npy is not None and os.path.exists(concept_scores_npy):
            self.concept_scores = np.load(concept_scores_npy)  # [num_neurons, vocab_size]
            print(f"Loaded concept_scores: {concept_scores_npy} with shape {self.concept_scores.shape}")
        if vocab_txt is not None and os.path.exists(vocab_txt):
            with open(vocab_txt, "r") as f:
                self.vocab = [ln.strip() for ln in f.readlines()]
            print(f"Loaded vocab of size {len(self.vocab)} from {vocab_txt}")
        self._register_attnprob_hooks()
    

    def _register_attnprob_hooks(self):
        """
        Register forward hooks on CLIP attention dropout layers to capture softmax attention probabilities.
        Ensures each attention block is hooked only once to avoid duplicate collection.
        """
        self._attn_probs = []  # Store [B,H,S,S] for each layer
        self._hooked_modules = set()  # Track hooked modules

        def grab_from_attn_forward(module, inputs, output):
            # CLIPAttention.forward typically returns (attn_output, attn_weights) or (attn_output,)
            attn_weights = None
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
            if attn_weights is not None:
                attn_weights.retain_grad()
                self._attn_probs.append(attn_weights)
                if self.verbose:
                    print(f"[DEBUG] Hooked attn_weights from CLIPAttention.forward: {attn_weights.shape}")

        def grab_from_dropout(module, inputs, output):
            # Dropout input is post-softmax probabilities
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

        # Traverse modules, ensure each attention block is hooked once
        for name, m in self.clip_model.vision_model.named_modules():
            if CLIPAttention is not None and isinstance(m, CLIPAttention):
                if name not in self._hooked_modules:
                    m.register_forward_hook(grab_from_attn_forward)
                    self._hooked_modules.add(name)
                    if self.verbose:
                        print(f"[DEBUG] Registered hook on CLIPAttention: {name}")
            
            # Only use dropout as fallback if CLIPAttention hook fails
            elif (name not in self._hooked_modules and 
                  hasattr(m, 'dropout') and 
                  isinstance(getattr(m, 'dropout'), nn.Dropout) and
                  'attention' in name):  # Ensure dropout is attention-related
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
        if neuron_id is not None:
            assert 0 <= neuron_id < z.shape[1], f"neuron_id out of range [0,{z.shape[1]-1}]"
            y = z[:, neuron_id].sum()
            info["target_mode"] = "single_neuron"
            info["neuron_id"] = int(neuron_id)
        else:
            assert self.concept_scores is not None, "concept_scores.npy required for vocab_concept_id mode"
            assert 0 <= vocab_concept_id < self.concept_scores.shape[0], \
                f"vocab_concept_id out of range [0,{self.concept_scores.shape[0]-1}]"
            # Use positive weights and current image activation gating
            w = self.concept_scores[vocab_concept_id, :].copy()  # [K]
            w[w < 0] = 0.0
            w_t = torch.as_tensor(w, device=z.device, dtype=z.dtype).unsqueeze(0)  # [1,K]
            z_pos = torch.relu(z)           # Only use truly activated slots in this image
            contrib = z_pos * w_t           # Contribution scores
            
            # 1. Count actually contributing (non-zero) neurons
            num_active_neurons = torch.count_nonzero(contrib)

            # 2. k should not exceed min of requested k and actual active count
            k = min(topk_neurons_for_concept, num_active_neurons)

            # 3. Edge case: if no neurons active, cannot generate heatmap
            if k == 0:
                print("WARNING: No active neurons for this concept and image. Cannot generate relevancy map.")
                # Return empty heatmap and error info
                side = int(round((self.clip_model.vision_model.config.num_patches) ** 0.5))
                return np.zeros((side, side)), {"error": "No active neurons", "topk_neurons_used": 0}

            # Record actual k used for debugging
            info["topk_neurons_used"] = int(k) 

            # 4. Use corrected k value for topk operation
            vals, idx = torch.topk(contrib, k=k, dim=1) 
            # --- END OF THE FIX ---

            print(f"DEBUG: For concept ID {vocab_concept_id}, ACTUALLY using top-{k}. Neuron indices are: {idx.cpu().numpy()}")
            mask = torch.zeros_like(contrib)
            mask.scatter_(1, idx, 1.0)
            y = (contrib * mask).sum()    

            info["target_mode"] = "vocab_linear_combo_gated"
            info["vocab_concept_id"] = int(vocab_concept_id)
            if self.vocab is not None and 0 <= vocab_concept_id < len(self.vocab):
                info["vocab_token"] = self.vocab[vocab_concept_id]
            info["topk_neurons"] = int(k)

        # Backprop to attentions
        self.clip_model.zero_grad(set_to_none=True)
        if hasattr(self.sae, "zero_grad"):
            self.sae.zero_grad(set_to_none=True)
        y.backward()  # 移除 retain_graph=True，因为每次调用只做一次反向传播
        
        if self.verbose:
            print(f"[DEBUG] captured layers: {len(self._attn_probs)}")
        for i, A in enumerate(self._attn_probs[-8:]):
            g = A.grad
            if self.verbose:
                print(f"[DEBUG] L{-len(self._attn_probs)+i+1}: shape={tuple(A.shape)}, "
                f"grad_none={g is None}, grad_norm={(g.float().norm().item() if g is not None else 0):.3e}")


        # Debug: 看看最后几层的 A 和 grad 的范数
        for i, A in enumerate(self._attn_probs[-8:]):
            g = A.grad
            if self.verbose:
                print(f"[DEBUG] layer{-8+i}: A.requires_grad={A.requires_grad}, "
                      f"A.mean={A.float().mean().item():.4e}, "
                      f"grad_none={g is None}, grad_norm={(g.float().norm().item() if g is not None else 0):.4e}")

        # GAE rollout
        if len(self._attn_probs) == 0:
            raise RuntimeError("No attn_probs were captured by hooks. Is the hook registration correct?")

        # --- 关键修正：确保我们使用所有捕获到的注意力层 ---
        # 原始的 "rollout" 方法旨在聚合从第一层到最后一层的全部信息
        att_list = self._attn_probs
        
        if self.verbose:
            print(f"[DEBUG] Using all {len(att_list)} layers for GAE rollout.")

        R = self._gae_rollout_from_attentions(att_list)

        r_cls = R[0]                                       # CLS -> tokens, [S]
        r_patches = r_cls[1:]                              # drop CLS itself

        # infer patch grid size from S
        S = R.shape[0]
        num_patches = S - 1
        side = int(round(num_patches ** 0.5))
        if side * side != num_patches:
            # 非 224/14 的情况；尽量 reshape 成近似方阵
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
