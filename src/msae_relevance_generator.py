import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
sys.path.append('../models')
from sae import SAE
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']

sys.path.append('/home/selena/Transformer-Explainability')
sys.path.append('/home/selena/Transformer-Explainability/baselines/ViT')

try:
    # 导入Transformer-Explainability的核心模块
    from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
    from baselines.ViT.ViT_LRP import deit_base_patch16_224 as deit_LRP  
    from baselines.ViT.ViT_explanation_generator import LRP
    print("✅ 成功导入Transformer-Explainability模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已经克隆了Transformer-Explainability仓库")

class MSAETransformerExplainabilityIntegration:
    """
    集成MSAE和Transformer-Explainability的综合解决方案
    """
    
    def __init__(self, 
                 clip_model: CLIPModel,
                 clip_processor: CLIPProcessor,
                 sae_model: SAE,
                 concept_scores: np.ndarray,
                 vocab: List[str],
                 device: str = "cuda"):
        
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.sae_model = sae_model
        self.concept_scores = concept_scores
        self.vocab = vocab
        self.device = device
        
        # 初始化ViT-LRP模型用于relevance map生成
        self.init_vit_explainability()
        
    def init_vit_explainability(self):
        """
        初始化ViT explainability模型
        """
        try:
            # 加载预训练的ViT模型用于explainability
            self.vit_model = vit_LRP(pretrained=True).to(self.device)
            self.vit_model.eval()
            
            # 创建LRP解释器
            self.lrp_generator = LRP(self.vit_model)
            
            print("✅ ViT explainability模型初始化成功")
        except Exception as e:
            print(f"❌ ViT explainability初始化失败: {e}")
            print("尝试使用DeiT模型...")
            try:
                self.vit_model = deit_LRP(pretrained=True).to(self.device)
                self.vit_model.eval()
                self.lrp_generator = LRP(self.vit_model)
                print("✅ DeiT explainability模型初始化成功")
            except Exception as e2:
                print(f"❌ DeiT explainability也失败了: {e2}")
                raise e2
    
    def prepare_image_for_vit(self, image: Image.Image) -> torch.Tensor:
        """
        为ViT模型准备图像 - 修复gradient hook问题
        """
        import torchvision.transforms as transforms
        
        # ViT标准预处理
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        
        tensor = transform(image).unsqueeze(0).to(self.device)
        # 🔥 关键修复：确保tensor需要梯度
        if not tensor.requires_grad:
            tensor.requires_grad_(True)
        
        return tensor
    
    def get_clip_features(self, image: Image.Image, text: Optional[str] = None) -> torch.Tensor:
        """
        获取CLIP特征
        """
        self.clip_model.eval()
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            # 1) 走视觉塔，拿 pooler_output（未投影）
            vision_out = self.clip_model.vision_model(inputs["pixel_values"])
            pooled = vision_out.pooler_output  # [B, hidden_size=1024] for ViT-L/14

            # 2) 过 projection 得 768（未归一化）
            img_768 = self.clip_model.visual_projection(pooled)  # [B, 768]

            # 3) 如果你的 checkpoint 是 centered 版，这里减均值（自己存个 mean.npy）
            if getattr(self, "clip_mean_768", None) is not None:
                mean = torch.as_tensor(self.clip_mean_768, device=img_768.device, dtype=img_768.dtype)
                img_768 = img_768 - mean  # 对齐“centered”

            return img_768  # 别再 /norm 了

    def generate_transformer_relevance_map(self, 
                                         image: Image.Image,
                                         method: str = "transformer_attribution",
                                         class_index: Optional[int] = None) -> np.ndarray:
        """
        使用Transformer-Explainability生成relevance map - 修复hook问题
        """
        # 预处理图像
        vit_input = self.prepare_image_for_vit(image)
        
        try:
            # 🔥 关键修复：在gradient模式下运行
            vit_input.requires_grad_(True)
            
            # 获取模型预测
            output = self.vit_model(vit_input)
            if class_index is None:
                class_index = output.argmax(dim=1).item()
            
            print(f"[DEBUG] 预测类别: {class_index}")
            
            # 生成relevance map - 不使用torch.no_grad()
            if method == "transformer_attribution":
                relevance_map = self.lrp_generator.generate_LRP(
                    vit_input, 
                    method="transformer_attribution",
                    index=class_index
                )
            elif method == "rollout":
                relevance_map = self.lrp_generator.generate_LRP(
                    vit_input,
                    method="rollout", 
                    index=class_index
                )
            elif method == "lrp":
                relevance_map = self.lrp_generator.generate_LRP(
                    vit_input,
                    method="lrp",
                    index=class_index  
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # 转换为numpy
            if isinstance(relevance_map, torch.Tensor):
                relevance_map = relevance_map.detach().cpu().numpy()
            
            print(f"[DEBUG] Raw relevance map shape: {relevance_map.shape}")
            
            if relevance_map.ndim == 4:
                # [batch, channels, height, width] -> [height, width]
                relevance_map = relevance_map[0, 0]  # 取第一个batch和channel
            elif relevance_map.ndim == 3:
                # [batch, height, width] -> [height, width]  
                relevance_map = relevance_map[0]
            elif relevance_map.ndim == 2:
                if relevance_map.shape[0] == 1 and relevance_map.shape[1] == 196:
                    # [1, 196] -> [14, 14]
                    relevance_map = relevance_map[0].reshape(14, 14)
                elif relevance_map.shape[0] == 196:
                    # [196, 1] -> [14, 14] 
                    relevance_map = relevance_map[:, 0].reshape(14, 14)
                elif relevance_map.shape == (196,):
                    # [196] -> [14, 14]
                    relevance_map = relevance_map.reshape(14, 14)
                else:
                    print(f"[WARNING] Unexpected 2D shape: {relevance_map.shape}")
                    # 尝试强制reshape到14x14
                    if relevance_map.size == 196:
                        relevance_map = relevance_map.flatten()[:196].reshape(14, 14)
                    else:
                        print(f"[ERROR] Cannot reshape {relevance_map.shape} to (14, 14)")
                        return np.zeros((14, 14)), class_index
            elif relevance_map.ndim == 1:
                if relevance_map.shape[0] == 196:
                    # [196] -> [14, 14]
                    relevance_map = relevance_map.reshape(14, 14)
                else:
                    print(f"[WARNING] Unexpected 1D shape: {relevance_map.shape}")
                    return np.zeros((14, 14)), class_index
            
            print(f"[DEBUG] Final relevance map shape: {relevance_map.shape}")
            return relevance_map, class_index
            
        except Exception as e:
            print(f"[ERROR] Transformer relevance generation failed: {e}")
            import traceback
            traceback.print_exc()
            # 返回空的relevance map
            return np.zeros((14, 14)), 0
    
    def generate_msae_concept_analysis(self, image: Image.Image) -> Dict:
        """
        使用MSAE生成概念级分析 - 修复维度问题和概念阈值
        """
        # 获取CLIP特征
        clip_features = self.get_clip_features(image)
        print(f"[DEBUG] CLIP features shape: {clip_features.shape}")
        
        # 通过SAE分析
        with torch.no_grad():
            sae_activations = self.sae_model.encode(clip_features)
            print(f"[DEBUG] SAE activations type: {type(sae_activations)}")
            
            # 处理不同的返回类型
            if isinstance(sae_activations, tuple):
                sae_activations = sae_activations[0]
                print(f"[DEBUG] After tuple unwrap: {sae_activations.shape}")
            
            # 确保是2D tensor (batch_size, features)
            if sae_activations.dim() == 1:
                sae_activations = sae_activations.unsqueeze(0)  # 添加batch维度
                
            print(f"[DEBUG] Final SAE activations shape: {sae_activations.shape}")
            
            sae_reconstruction = self.sae_model.decode(sae_activations)
        
        # 🔥 关键修复：降低阈值并增加调试信息
        concept_mappings = []
        
        if sae_activations.dim() == 2:
            # 对于batch形式 [batch_size, features]
            batch_size, num_features = sae_activations.shape
            
            for batch_idx in range(batch_size):
                # 找到该batch中激活的神经元
                batch_activations = sae_activations[batch_idx]
                active_indices = torch.nonzero(batch_activations > 0, as_tuple=False).squeeze(-1)
                
                print(f"[DEBUG] Batch {batch_idx}: {active_indices.numel()} active neurons")
                
                if active_indices.numel() == 0:
                    continue
                    
                # 限制处理的激活神经元数量，但先看看所有激活强度
                activation_values = batch_activations[active_indices].cpu().numpy()
                print(f"[DEBUG] Top 10 activation values: {sorted(activation_values, reverse=True)[:10]}")
                
                # 限制处理的激活神经元数量
                for i in range(min(100, active_indices.shape[0])):  # 增加到100个
                    neuron_idx_int = active_indices[i].item()
                    activation_strength = batch_activations[neuron_idx_int].item()
                    
                    # 检查neuron_idx是否在concept_scores范围内
                    if neuron_idx_int < self.concept_scores.shape[0]:
                        neuron_concept_scores = self.concept_scores[neuron_idx_int]
                        best_concept_idx = np.argmax(neuron_concept_scores)
                        best_similarity = neuron_concept_scores[best_concept_idx]
                        
                        # 🔥 降低阈值从0.4到0.2，并添加调试信息
                        if i < 5:  # 只对前5个神经元打印详细信息
                            print(f"[DEBUG] Neuron {neuron_idx_int}: activation={activation_strength:.4f}, "
                                  f"best_concept_idx={best_concept_idx}, similarity={best_similarity:.4f}")
                        
                        if best_similarity > 0.3:  # 降低阈值
                            concept_name = (self.vocab[best_concept_idx] 
                                          if best_concept_idx < len(self.vocab) 
                                          else f"concept_{best_concept_idx}")
                            
                            concept_mappings.append({
                                'neuron_id': neuron_idx_int,
                                'concept': concept_name,
                                'activation_strength': activation_strength,
                                'concept_similarity': best_similarity
                            })
        else:
            # 对于1D情况 [features]
            active_indices = torch.nonzero(sae_activations > 0, as_tuple=False).squeeze(-1)
            
            print(f"[DEBUG] 1D case: {active_indices.numel()} active neurons")
            
            if active_indices.numel() > 0:
                activation_values = sae_activations[active_indices].cpu().numpy()
                print(f"[DEBUG] Top 10 activation values: {sorted(activation_values, reverse=True)[:10]}")
                
                for i in range(min(100, active_indices.shape[0])):
                    neuron_idx_int = active_indices[i].item()
                    activation_strength = sae_activations[neuron_idx_int].item()
                    
                    if neuron_idx_int < self.concept_scores.shape[0]:
                        neuron_concept_scores = self.concept_scores[neuron_idx_int]
                        best_concept_idx = np.argmax(neuron_concept_scores)
                        best_similarity = neuron_concept_scores[best_concept_idx]
                        
                        if i < 5:  # 只对前5个神经元打印详细信息
                            print(f"[DEBUG] Neuron {neuron_idx_int}: activation={activation_strength:.4f}, "
                                  f"best_concept_idx={best_concept_idx}, similarity={best_similarity:.4f}")
                        
                        if best_similarity > 0.2:  # 降低阈值
                            concept_name = (self.vocab[best_concept_idx] 
                                          if best_concept_idx < len(self.vocab) 
                                          else f"concept_{best_concept_idx}")
                            
                            concept_mappings.append({
                                'neuron_id': neuron_idx_int,
                                'concept': concept_name,
                                'activation_strength': activation_strength,
                                'concept_similarity': best_similarity
                            })
        
        # 按激活强度排序
        concept_mappings.sort(key=lambda x: x['activation_strength'], reverse=True)
        
        print(f"[DEBUG] Found {len(concept_mappings)} valid concepts (threshold=0.2)")
        if concept_mappings:
            print(f"[DEBUG] Top 3 concepts: {[(c['concept'], c['activation_strength'], c['concept_similarity']) for c in concept_mappings[:3]]}")
        
        return {
            'concept_mappings': concept_mappings,
            'reconstruction_error': torch.nn.functional.mse_loss(clip_features, sae_reconstruction).item(),
            'sparsity': (sae_activations == 0).float().mean().item(),
            'num_active_concepts': len(concept_mappings)
        }
    
    def generate_comprehensive_analysis(self, 
                                      image: Image.Image,
                                      methods: List[str] = ["transformer_attribution"],
                                      return_visualizations: bool = True) -> Dict:
        """
        生成综合分析，结合Transformer explainability和MSAE
        """
        results = {}
        
        print("🔍 生成Transformer relevance maps...")
        # 1. 生成多种transformer relevance maps
        transformer_results = {}
        for method in methods:
            try:
                relevance_map, class_idx = self.generate_transformer_relevance_map(image, method)
                transformer_results[method] = {
                    'relevance_map': relevance_map,
                    'predicted_class': class_idx
                }
                print(f"✅ {method} 完成")
            except Exception as e:
                print(f"❌ {method} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        results['transformer_results'] = transformer_results
        
        print("🧠 生成MSAE概念分析...")
        # 2. 生成MSAE概念分析
        try:
            msae_results = self.generate_msae_concept_analysis(image)
            results['msae_results'] = msae_results
        except Exception as e:
            print(f"❌ MSAE concept analysis failed: {e}")
            import traceback
            traceback.print_exc()
            # 提供默认值
            results['msae_results'] = {
                'concept_mappings': [],
                'reconstruction_error': 0.0,
                'sparsity': 0.0,
                'num_active_concepts': 0
            }
        
        print("🔗 融合分析...")
        # 3. 融合分析
        fusion_results = self.fuse_transformer_and_concept_analysis(
            transformer_results, results['msae_results']
        )
        results['fusion_results'] = fusion_results
        
        # 4. 生成可视化
        if return_visualizations:
            print("📊 生成可视化...")
            try:
                visualizations = self.create_comprehensive_visualizations(
                    image, results
                )
                results['visualizations'] = visualizations
                # === 自动保存所有可视化到 outputs 目录 ===
                import os
                output_dir = "/home/selena/relevance_map_SAE/outputs"
                os.makedirs(output_dir, exist_ok=True)
                for name, fig in visualizations.items():
                    save_path = os.path.join(output_dir, f"{name}.png")
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"✅ Saved visualization: {save_path}")
            except Exception as e:
                print(f"❌ 可视化生成失败: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def fuse_transformer_and_concept_analysis(self, 
                                            transformer_results: Dict,
                                            msae_results: Dict) -> Dict:
        """
        融合Transformer explainability和概念分析
        """
        fusion = {
            'summary': {},
            'spatial_concept_correlation': {},
            'insights': []
        }
        
        # 统计信息
        fusion['summary'] = {
            'num_transformer_methods': len(transformer_results),
            'num_active_concepts': msae_results['num_active_concepts'],
            'reconstruction_quality': 1 - msae_results['reconstruction_error'],
            'sparsity': msae_results['sparsity']
        }
        
        # 如果有transformer结果，尝试关联空间和概念
        if transformer_results:
            method_with_result = None
            for method, result in transformer_results.items():
                if result['relevance_map'] is not None:
                    method_with_result = method
                    break
                    
            if method_with_result:
                relevance_map = transformer_results[method_with_result]['relevance_map']
                
                # 简单的空间-概念关联分析
                if relevance_map is not None and relevance_map.size > 0:
                    high_relevance_regions = np.where(relevance_map > np.percentile(relevance_map, 75))
                    fusion['spatial_concept_correlation'] = {
                        'high_relevance_pixels': len(high_relevance_regions[0]),
                        'max_relevance': float(np.max(relevance_map)),
                        'mean_relevance': float(np.mean(relevance_map))
                    }
        
        # 生成insights
        top_concepts = msae_results['concept_mappings'][:5] if msae_results['concept_mappings'] else []
        
        if top_concepts:
            concept_names = [c['concept'] for c in top_concepts]
            fusion['insights'] = [
                f"模型主要关注以下概念: {', '.join(concept_names)}",
                f"稀疏性: {fusion['summary']['sparsity']:.1%}",
                f"重建质量: {fusion['summary']['reconstruction_quality']:.1%}"
            ]
        else:
            fusion['insights'] = [
                "未找到激活的概念",
                f"稀疏性: {fusion['summary']['sparsity']:.1%}",
                f"重建质量: {fusion['summary']['reconstruction_quality']:.1%}"
            ]
        
        return fusion
    
    def create_comprehensive_visualizations(self, 
                                           image: Image.Image,
                                           results: Dict) -> Dict:
        """
        创建综合可视化 - 修复axes reshape和缩进问题
        """
        visualizations = {}
        
        # 1. Transformer relevance maps可视化
        transformer_results = results['transformer_results']
        valid_transformer_results = {k: v for k, v in transformer_results.items() 
                                   if v['relevance_map'] is not None and v['relevance_map'].size > 0}
        
        if valid_transformer_results:
            n_methods = len(valid_transformer_results)
            n_cols = max(2, n_methods)  # 至少2列
            fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
            # 保证axes是2D
            axes = np.atleast_2d(axes)
            print(f"[DEBUG] Figure axes shape: {axes.shape}, n_methods: {n_methods}, n_cols: {n_cols}")
            # 原图
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            # 各种方法的relevance maps
            for i, (method, result) in enumerate(valid_transformer_results.items()):
                relevance_map = result['relevance_map']
                if relevance_map.shape != (14, 14):
                    print(f"[WARNING] Relevance map shape is {relevance_map.shape}, expected (14, 14)")
                    continue
                relevance_resized = cv2.resize(relevance_map.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
                col_idx = i + 1 if i + 1 < n_cols else i  # 避免索引越界
                if col_idx < axes.shape[1]:
                    axes[0, col_idx].imshow(image, alpha=0.7)
                    im1 = axes[0, col_idx].imshow(relevance_resized, cmap='hot', alpha=0.5)
                    axes[0, col_idx].set_title(f'{method}\n(Class: {result["predicted_class"]})')
                    axes[0, col_idx].axis('off')
                if col_idx < axes.shape[1]:
                    im2 = axes[1, col_idx].imshow(relevance_resized, cmap='hot')
                    axes[1, col_idx].set_title(f'{method} - Heatmap')
                    axes[1, col_idx].axis('off')
                    try:
                        plt.colorbar(im2, ax=axes[1, col_idx], fraction=0.046, pad=0.04)
                    except Exception as e:
                        print(f"[WARNING] Could not add colorbar: {e}")
            # 隐藏多余的子图
            for i in range(len(valid_transformer_results) + 1, n_cols):
                axes[0, i].axis('off')
                axes[1, i].axis('off')
            plt.tight_layout()
            visualizations['transformer_maps'] = fig
        
        # 2. MSAE概念分析可视化
        msae_results = results['msae_results']
        if msae_results['concept_mappings']:
            concepts = [m['concept'] for m in msae_results['concept_mappings'][:15]]
            activations = [m['activation_strength'] for m in msae_results['concept_mappings'][:15]]
            similarities = [m['concept_similarity'] for m in msae_results['concept_mappings'][:15]]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # 激活强度图
            if len(concepts) > 0:
                bars1 = ax1.barh(concepts, activations, color='skyblue', alpha=0.7)
                ax1.set_xlabel('Activation Strength')
                ax1.set_title('Top Concepts by Activation Strength')
                ax1.grid(axis='x', alpha=0.3)
                
                # 添加数值标签
                max_act = max(activations) if activations else 1
                for bar, act in zip(bars1, activations):
                    ax1.text(bar.get_width() + max_act * 0.01, 
                            bar.get_y() + bar.get_height()/2, 
                            f'{act:.3f}', va='center', ha='left', fontsize=8)
                
                # 概念相似度图
                bars2 = ax2.barh(concepts, similarities, color='lightcoral', alpha=0.7)
                ax2.set_xlabel('Concept Similarity')
                ax2.set_title('Concept Similarity Scores')
                ax2.grid(axis='x', alpha=0.3)
                
                max_sim = max(similarities) if similarities else 1
                for bar, sim in zip(bars2, similarities):
                    ax2.text(bar.get_width() + max_sim * 0.01,
                            bar.get_y() + bar.get_height()/2,
                            f'{sim:.3f}', va='center', ha='left', fontsize=8)
            else:
                # 如果没有概念，显示空图表
                ax1.text(0.5, 0.5, 'No concepts found\n(Try lowering threshold)', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Top Concepts by Activation Strength')
                
                ax2.text(0.5, 0.5, 'No concepts found\n(Try lowering threshold)', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Concept Similarity Scores')
            
            plt.tight_layout()
            visualizations['concept_analysis'] = fig
        
        # 3. 融合分析可视化 - 使用英文避免字体问题
        fusion_results = results['fusion_results']
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 使用英文避免字体问题
        info_text = "\n".join([
            "=== Comprehensive Analysis Report ===",
            f"Transformer Methods: {fusion_results['summary']['num_transformer_methods']}",
            f"Active Concepts: {fusion_results['summary']['num_active_concepts']}",
            f"Reconstruction Quality: {fusion_results['summary']['reconstruction_quality']:.1%}",
            f"Sparsity: {fusion_results['summary']['sparsity']:.1%}",
            "",
            "=== Key Insights ===",
        ])
        
        # 添加insights，但转换为英文或过滤中文
        for insight in fusion_results['insights']:
            try:
                # 尝试编码为ASCII，过滤掉中文字符
                clean_insight = insight.encode('ascii', 'ignore').decode('ascii')
                if clean_insight.strip():  # 如果过滤后还有内容
                    info_text += f"\n- {clean_insight}"
            except:
                pass
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.axis('off')
        ax.set_title('Comprehensive Analysis Summary', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        visualizations['summary'] = fig
        
        return visualizations


if __name__ == "__main__":
    print("🚀 Testing Relevance Map...")

    # 1. Load all models
    print("📥 Loading models...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Load SAE
    sae_path = "../models/cache/models--WolodjaZ--MSAE/snapshots/e6a85249444be9b25c7751e9677464dedfdc7307/ViT-L_14/centered/6144_768_TopKReLU_64_RW_False_False_0.0_cc3m_ViT-L~14_train_image_2905936_768.pth"
    sae_model = SAE(sae_path)
    if hasattr(sae_model, 'to'):
        sae_model = sae_model.to(device)

    # Load concept scores and vocabulary
    concept_path = "../models/cache/models--WolodjaZ--MSAE/snapshots/e6a85249444be9b25c7751e9677464dedfdc7307/ViT-L_14/centered/Concept_Interpreter_6144_768_TopKReLU_64_RW_False_False_0.0_cc3m_ViT-L~14_train_image_2905936_768_disect_ViT-L~14_-1_text_20000_768.npy"
    concept_scores = np.load(concept_path)

    vocab_path = "../models/cache/models--WolodjaZ--MSAE/snapshots/e6a85249444be9b25c7751e9677464dedfdc7307/clip_disect_20k.txt"
    with open(vocab_path, 'r') as f:
        vocab = [line.strip() for line in f.readlines()]

    print("✅ All models loaded!")

    # Create analyzer with loaded models
    analyzer = MSAETransformerExplainabilityIntegration(
        clip_model=clip_model,
        clip_processor=clip_processor,
        sae_model=sae_model,
        concept_scores=concept_scores,
        vocab=vocab,
        device=device
    )

    # Generate all resultsK
    test_image_path = "../data/images/Cat/dog_and_cat.png"
    image = Image.open(test_image_path).convert('RGB')

    results = analyzer.generate_comprehensive_analysis(image)