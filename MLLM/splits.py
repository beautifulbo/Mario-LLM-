import dgl
import torch
import numpy as np

train_ratio=0.7
val_ratio=0.1
test_ratio=0.2

img=np.load("../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B-Instruct_visual.npy")
text=np.load("../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean.npy")
graphs=dgl.load_graphs("../Data/Reddit/RedditSGraph.pt")[0][0]

train_node_idx, val_node_idx, test_node_idx = dgl.data.utils.split_dataset(graphs.nodes(), [train_ratio, val_ratio, test_ratio], shuffle=False)

print(f"训练集节点数量: {len(train_node_idx)}")
print(f"验证集节点数量: {len(val_node_idx)}")
print(f"测试集节点数量: {len(test_node_idx)}")

train_subgraph=dgl.node_subgraph(graphs,train_node_idx)
val_subgraph=dgl.node_subgraph(graphs,val_node_idx)
test_subgraph=dgl.node_subgraph(graphs,test_node_idx)

dgl.save_graphs("../Data/Reddit/RedditSGraph_train.pt",[train_subgraph])
dgl.save_graphs("../Data/Reddit/RedditSGraph_val.pt",[val_subgraph])
dgl.save_graphs("../Data/Reddit/RedditSGraph_test.pt",[test_subgraph])

print("数据集划分完成，子图已保存。")


train_img=img[train_node_idx]
train_text=text[train_node_idx]
val_img=img[val_node_idx]
val_text=text[val_node_idx]
test_img=img[test_node_idx]
test_text=text[test_node_idx]

np.save("../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_train.npy",train_img)
np.save("../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_train.npy",train_text)
np.save("../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_val.npy",val_img)
np.save("../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_val.npy",val_text)
np.save("../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_test.npy",test_img)
np.save("../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_test.npy",test_text)

print("特征数据划分完成，已保存。")