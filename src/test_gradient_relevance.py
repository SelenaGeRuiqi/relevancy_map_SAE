import sys
import os
sys.path.append('../models')

from msae_relevance_generator import MSAETransformerExplainabilityIntegration
from transformers import CLIPProcessor, CLIPModel
from sae import SAE
import numpy as np
import torch

print("🚀 测试Relevance Map...")

# 1. 加载所有模型
print("📥 加载模型...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 加载SAE
sae_path = "../models/cache/models--WolodjaZ--MSAE/snapshots/e6a85249444be9b25c7751e9677464dedfdc7307/ViT-L_14/centered/6144_768_TopKReLU_64_RW_False_False_0.0_cc3m_ViT-L~14_train_image_2905936_768.pth"
sae_model = SAE(sae_path)
if hasattr(sae_model, 'to'):
    sae_model = sae_model.to(device)

# 加载概念分数和词典
concept_path = "../models/cache/models--WolodjaZ--MSAE/snapshots/e6a85249444be9b25c7751e9677464dedfdc7307/ViT-L_14/centered/Concept_Interpreter_6144_768_TopKReLU_64_RW_False_False_0.0_cc3m_ViT-L~14_train_image_2905936_768_disect_ViT-L~14_-1_text_20000_768.npy"
concept_scores = np.load(concept_path)

vocab_path = "../models/cache/models--WolodjaZ--MSAE/snapshots/e6a85249444be9b25c7751e9677464dedfdc7307/clip_disect_20k.txt"
with open(vocab_path, 'r') as f:
    vocab = [line.strip() for line in f.readlines()]

print("✅ 所有模型加载完成！")


# 3. 使用你已经加载的模型创建分析器
analyzer = MSAETransformerExplainabilityIntegration(
   clip_model=clip_model,           # 你已有的
   clip_processor=clip_processor,   # 你已有的  
   sae_model=sae_model,            # 你已有的
   concept_scores=concept_scores,   # 你已有的
   vocab=vocab,                     # 你已有的
   device=device
)

# 4. 一键生成所有结果
from PIL import Image

test_image_path = "../data/images/Cat/two_cat.png"
image = Image.open(test_image_path).convert('RGB')

results = analyzer.generate_comprehensive_analysis(image)

# # 2. 创建gradient relevance map生成器
# generator = GradientRelevanceMapGenerator(
#     clip_model=clip_model,
#     clip_processor=clip_processor, 
#     sae_model=sae_model,
#     concept_scores=concept_scores,
#     vocab=vocab
# )

# # 3. 测试不同概念
# test_concepts_mapping = {
#     "Cat": "cat",           # 粗概念
#     "Siamese": "siamese",   # 细概念  
#     "Persian": "persian",   # 细概念
#     "British_Shorthair": "cat"  # 用通用概念
# }

# print("\n🎯 开始测试不同概念的relevance map...")

# for folder_concept, vocab_concept in test_concepts_mapping.items():
#     folder_path = f"../data/images/{folder_concept}"
#     if os.path.exists(folder_path):
#         images = [f for f in os.listdir(folder_path) 
#                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
#         if images:
#             # 测试第一张图片
#             test_image = os.path.join(folder_path, images[0])
#             print(f"\n🖼️  测试: {test_image} -> 概念: {vocab_concept}")
            
#             # 生成relevance map
#             result = generator.generate_relevance_map(test_image, vocab_concept)
            
#             if result:
#                 print("✅ Gradient relevance map生成成功！")
                
#                 # 保存结果
#                 os.makedirs("../outputs", exist_ok=True)
#                 save_path = f"../outputs/gradient_relevance_{folder_concept}_{vocab_concept}.png"
#                 generator.visualize_relevance(result, save_path)
                
#             else:
#                 print("❌ Relevance map生成失败")
                
#             # 清理GPU内存
#             torch.cuda.empty_cache()

# print("\n🎉 所有测试完成！")
# print("📂 结果保存在 ../outputs/ 文件夹中")