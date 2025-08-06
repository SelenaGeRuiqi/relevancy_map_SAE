from huggingface_hub import hf_hub_download
import os

print("开始下载MSAE模型...")

# 下载MSAE模型权重 (RW版本)
msae_weights = hf_hub_download(
    repo_id="WolodjaZ/MSAE",
    filename="ViT-L_14/not_centered/6144_768_TopKReLU_64_RW_False_False_0.0_cc3m_ViT-L~14_train_image_2905936_768.pth",
    cache_dir="./cache"
)
print(f"✅ not centered MSAE权重下载完成: {msae_weights}")

# 下载概念匹配分数
concept_scores = hf_hub_download(
    repo_id="WolodjaZ/MSAE", 
    filename="ViT-L_14/not_centered/Concept_Interpreter_6144_768_TopKReLU_64_RW_False_False_0.0_cc3m_ViT-L~14_train_image_2905936_768_disect_ViT-L~14_-1_text_20000_768.npy",
    cache_dir="./cache"
)
print(f"✅ not centered 概念分数下载完成: {concept_scores}")

# 下载词典文件
vocab_file = hf_hub_download(
    repo_id="WolodjaZ/MSAE",
    filename="clip_disect_20k.txt",
    cache_dir="./cache"
)
print(f"✅ 词典文件下载完成: {vocab_file}")

print("\n🎉 所有模型文件下载完成！")
