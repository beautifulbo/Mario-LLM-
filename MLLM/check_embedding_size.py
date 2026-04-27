import torch
import numpy as np

image_path="../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B-Instruct_visual.npy"

text_path="../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean.npy"

print("Loading image features...")
image_features = np.load(image_path)
print(f"Image features shape: {image_features.shape}")

print("Loading text features...")
text_features = np.load(text_path)
print(f"Text features shape: {text_features.shape}")
