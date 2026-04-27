import torch
import numpy as np
from Mario import MarioFeatureExtractor
import os
from torch.nn import functional as F
from torch import nn
import argparse
from torch.utils.data import DataLoader, Dataset
import tqdm
from train import InfoNCELoss,MultiModalDataset,normalize_state_dict_keys
import dgl


def feature_extraction(embed_dim, num_heads,graph,buckets_num=6,layers=6,dataloader=None,model_path=None,dist_matrix_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor=MarioFeatureExtractor(embed_dim=embed_dim, num_heads=num_heads, graph=graph, buckets_num=buckets_num, layers=layers,dist_matrix_path=dist_matrix_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = normalize_state_dict_keys(state_dict)
    extractor.load_state_dict(state_dict,strict=True)
    extractor.to(device)
    extractor.eval()
    text_feature_all=[]
    image_feature_all=[]
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
        with torch.no_grad():
            text_nodefeatures, image_nodefeatures = extractor(texts, images,idx)
        text_nodefeatures = text_nodefeatures.squeeze(0).cpu()
        image_nodefeatures = image_nodefeatures.squeeze(0).cpu()
        text_feature_all.append(text_nodefeatures)
        image_feature_all.append(image_nodefeatures)
    text_feature_all=torch.stack(text_feature_all, dim=0)
    image_feature_all=torch.stack(image_feature_all, dim=0)


    np.save("../Data/Movies/TextFeature/new_text_nodefeatures.npy", text_feature_all.cpu().numpy())
    np.save("../Data/Movies/ImageFeature/new_image_nodefeatures.npy", image_feature_all.cpu().numpy())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_data_path", type=str, default="../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean.npy", help="Path to the training text data")
    parser.add_argument("--image_data_path", type=str, default="../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual.npy", help="Path to the image training data")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the pre-trained model")
    parser.add_argument("--save_path", type=str, default="../trained_models", help="Path to save the trained model")
    parser.add_argument("--graph_path", type=str, default="../Data/Reddit/RedditSGraph.pt" , help="Path to the graph data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dist_matrix_path", type=str, default="../Data/Reddit/RedditDist.pt", help="Path to the distance matrix")
    parser.add_argument("--embed_dim", type=int, default=3584, help="Embedding dimension for the model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads for the model")
    parser.add_argument("--buckets_num", type=int, default=6, help="Number of buckets for the model")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers for the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model_path= os.path.join(args.save_path, "mario_feature_extractor_epoch_500.pth") if args.model_path is None else args.model_path
    graph = dgl.load_graphs(args.graph_path)[0][0]  # 加载图数据

    test_data = MultiModalDataset(text_path=args.text_data_path, image_path=args.image_data_path)
    dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    feature_extraction(embed_dim=args.embed_dim, num_heads=args.num_heads, graph=graph, buckets_num=args.buckets_num, layers=args.layers, dataloader=dataloader, model_path=model_path, dist_matrix_path=args.dist_matrix_path)

if __name__ == "__main__":
    main()
