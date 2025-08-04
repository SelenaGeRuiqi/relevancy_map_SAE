import sys
sys.path.append('.')
from sae import SAE
import numpy as np

print("🧪 测试SAE模型加载...")

# 模型文件路径
model_path = "./cache/models--WolodjaZ--MSAE/snapshots/e6a85249444be9b25c7751e9677464dedfdc7307/ViT-L_14/centered/6144_768_TopKReLU_64_RW_False_False_0.0_cc3m_ViT-L~14_train_image_2905936_768.pth"

# 加载SAE模型
sae_model = SAE(model_path)
print("✅ SAE模型加载成功！")

# 加载概念分数
concept_path = "./cache/models--WolodjaZ--MSAE/snapshots/e6a85249444be9b25c7751e9677464dedfdc7307/ViT-L_14/centered/Concept_Interpreter_6144_768_TopKReLU_64_RW_False_False_0.0_cc3m_ViT-L~14_train_image_2905936_768_disect_ViT-L~14_-1_text_20000_768.npy"
concept_scores = np.load(concept_path)
print(f"✅ 概念分数加载成功！形状: {concept_scores.shape}")

# 加载词典
vocab_path = "./cache/models--WolodjaZ--MSAE/snapshots/e6a85249444be9b25c7751e9677464dedfdc7307/clip_disect_20k.txt"
with open(vocab_path, 'r') as f:
    vocab = [line.strip() for line in f.readlines()]
print(f"✅ 词典加载成功！词汇数量: {len(vocab)}")

print("\n🎉 所有组件测试完成！")
