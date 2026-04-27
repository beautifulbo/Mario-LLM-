from torch.cuda.amp import autocast # PyTorch 2.0+ 也可以直接用 torch.autocast
import torch
from torch.nn import functional as F
import torch.nn as nn
import dgl
import numpy as np

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from torch.optim import AdamW

from Mario import MarioFeatureExtractor

import argparse
import tqdm
import wandb


def normalize_state_dict_keys(state_dict):
    """Strip torch.compile wrapper prefixes from checkpoint keys when present."""
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def get_model_state_dict(model):
    """Save the underlying module weights even when the model is compiled."""
    return model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()



class InfoNCELoss(nn.Module):
    """对应论文公式 (6) 的 InfoNCE 损失"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, h_text, h_image):
        # h_text: (batch, dim), h_image: (batch, dim)
        # 计算相似度矩阵
        h_text=F.normalize(h_text, dim=1)  # (batch, dim)
        h_image=F.normalize(h_image, dim=1)  # (batch, dim)
        sim = torch.matmul(h_text, h_image.T) / self.temperature
        # 得到节点和节点之间的相似度矩阵
        # 对称 InfoNCE
        labels = torch.arange(sim.size(0), device=sim.device)
        
        loss_t2i = F.cross_entropy(sim, labels)
        loss_i2t = F.cross_entropy(sim.T, labels)
        
        return (loss_t2i + loss_i2t)


def train(model, dataloader, optimizer, device, num_epochs,save_path="../trained_models"):
    model.train()
    model.to(device)
    loss_fn = InfoNCELoss()
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss = 0.0
        for batch in dataloader:
            # 假设 batch 包含 'text' 和 'image' 两个字段
            texts = batch['text'].to(device)
            images = batch['image'].to(device)
            idx = batch['idx'].to(device)
            if texts.shape[0] != images.shape[0]:
                raise ValueError("文本和图像的批次大小不匹配")
            if len(texts.shape)==2:
                texts = texts.unsqueeze(0)  # 如果文本是二维的，添加一个批次维度
            if len(images.shape)==2:
                images = images.unsqueeze(0)  # 如果图像也是二维的，添加一个批次维度

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 将文本和图像输入模型，获取特征表示
                text_features, image_features = model(texts, images, idx) # (1,seq_len, dim)
                # 计算 InfoNCE 损失
                text_features = text_features.squeeze(0)  # (seq_len, dim)
                image_features = image_features.squeeze(0)  # (seq_len, dim)
                loss = loss_fn(text_features, image_features)
            
            optimizer.zero_grad() # 清除之前的梯度
            loss.backward() # 反向传播计算新的梯度
            optimizer.step() # 更新模型参数
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch": epoch, "loss": avg_loss})
        if (epoch+1) % 50 == 0:
            print("Saving model checkpoint...")
            import os
            torch.save(get_model_state_dict(model), os.path.join(save_path, f"mario_feature_extractor_epoch_{epoch+1}.pth"))
            print(f"Model checkpoint saved to {os.path.join(save_path, f'mario_feature_extractor_epoch_{epoch+1}.pth')}")

class MultiModalDataset(Dataset):
    def __init__(self, text_path,image_path):
        self.data = {
            "text": np.load(text_path,mmap_mode='r'),  # 加载文本数据 (node_nums,embed_dim)
            "image": np.load(image_path,mmap_mode='r')  # 加载图像数据 (node_nums,embed_dim)
        }
        self.length = len(self.data["text"])  # 假设文本和图像数据长度相同
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            "idx": torch.tensor(idx),
            "text": self.data["text"][idx].astype(np.float32),  # 转换为 float32 类型
            "image": self.data["image"][idx].astype(np.float32)  # 转换为 float32 类型
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_data_path", type=str, default="../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_train.npy", help="Path to the training text data")
    parser.add_argument("--image_data_path", type=str, default="../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_train.npy", help="Path to the image training data")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the pre-trained model")
    parser.add_argument("--save_path", type=str, default="../trained_models", help="Path to save the trained model")
    parser.add_argument("--graph_path", type=str, default="../Data/Reddit/RedditSGraph_train.pt" , help="Path to the graph data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dist_matrix_path", type=str, default="../Data/Reddit/RedditDist_train.pt", help="Path to the distance matrix")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--embed_dim", type=int, default=3584, help="Embedding dimension for the model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads for the model")
    parser.add_argument("--buckets_num", type=int, default=6, help="Number of buckets for the model")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers for the model")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    graph = dgl.load_graphs(args.graph_path)[0][0]  # 加载图数据
    if args.model_path:
        model = MarioFeatureExtractor(embed_dim=args.embed_dim, num_heads=args.num_heads, graph=graph, buckets_num=args.buckets_num, layers=args.layers, dist_matrix_path=args.dist_matrix_path)
        state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
        state_dict = normalize_state_dict_keys(state_dict)
        model.load_state_dict(state_dict, strict=True)
    else:
        model = MarioFeatureExtractor(embed_dim=args.embed_dim, num_heads=args.num_heads, graph=graph, buckets_num=args.buckets_num, layers=args.layers, dist_matrix_path=args.dist_matrix_path)

    if torch.cuda.is_available():
        print("Compiling model with torch.compile...")
        model = torch.compile(model)


    train_data = MultiModalDataset(text_path=args.text_data_path, image_path=args.image_data_path)
    dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True,prefetch_factor=4)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate,weight_decay=0.01)

    train(model, dataloader, optimizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs=args.num_epochs)

    print("Training completed. Saving model...")
    import os
    torch.save(get_model_state_dict(model), os.path.join(args.save_path, "mario_feature_extractor_final.pth"))

    print(f"Model saved to {os.path.join(args.save_path, 'mario_feature_extractor_final.pth')}")

if __name__ == "__main__":
    wandb.init(project="MAGB-Benchmark", name="MarioFeatureExtractor_Training")
    main()
    wandb.finish()
