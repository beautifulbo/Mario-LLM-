import os
import ast
import argparse
import torch
import time
import warnings
import pandas as pd
from PIL import Image
import numpy as np
import dgl
import random
import wandb
from dgl import load_graphs
import networkx as nx
from Library import load_model_and_processor, prepare_inputs_for_model
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 确保 Python 的哈希行为可复现
    random.seed(seed)  # Python 内置的随机种子
    np.random.seed(seed)  # NumPy 的随机种子
    torch.manual_seed(seed)  # PyTorch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)  # PyTorch 的 GPU 随机种子（仅影响当前 GPU）
    torch.cuda.manual_seed_all(seed)  # 影响所有可用 GPU



def split_dataset(nodes_num, train_ratio, val_ratio):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)

    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)

    train_ids = indices[:train_size]
    val_ids = indices[train_size:train_size + val_size]
    test_ids = indices[train_size + val_size:]

    return train_ids, val_ids, test_ids



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multimodal Node Classification with MLLM and RAG-enhanced inference')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-11B-Vision-Instruct',
                        help='HuggingFace模型名称或路径')
    parser.add_argument('--dataset_name', type=str, default='Movies',
                        help='数据集名称（对应Data目录下的子目录名）')
    parser.add_argument('--base_dir', type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help='项目根目录路径')
    parser.add_argument('--max_new_tokens', type=int, default=15,
                        help='生成的最大token数量')
    parser.add_argument('--image_ext', type=str, default='.jpg',
                        help='图像文件扩展名')
    parser.add_argument('--neighbor_mode', type=str, default='both', choices=['text', 'image', 'both'],
                        help='邻居信息的使用模式（文本、图像或两者）')
    # 添加参数 upload_image, 控制在wandb的 table 中是否上传图像
    parser.add_argument('--upload_image', type=bool, default=False,
                        help='是否将图像上传到WandB')
    parser.add_argument('--use_center_text', type=str, default='True', help='是否使用中心节点文本')
    parser.add_argument('--use_center_image', type=str, default='True', help='是否使用中心节点图像')
    parser.add_argument('--add_CoT', type=str, default='False',
                        help='是否添加CoT')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='测试样本数量')
    parser.add_argument('--num_neighbours', type=int, default=0,
                        help='期望的邻居数')
    parser.add_argument(
        "--train_ratio", type=float, default=0.6, help="training ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="training ratio"
    )
    return parser.parse_args()


class DatasetLoader:
    def __init__(self, args):
        """初始化数据集加载器"""
        self.args = args
        self.data_dir = os.path.join(args.base_dir, 'Data', args.dataset_name)
        self._verify_paths()

    def _verify_paths(self):
        """验证必要路径是否存在"""
        required_files = [
            os.path.join(self.data_dir, f"{self.args.dataset_name}.csv"),
            os.path.join(self.data_dir, f"{self.args.dataset_name}Graph.pt"),
            os.path.join(self.data_dir, f"{self.args.dataset_name}Images")
        ]
        missing = [path for path in required_files if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"Missing required files/directories: {missing}")

    def load_data(self):
        """加载数据集"""
        # 加载CSV数据
        csv_path = os.path.join(self.data_dir, f"{self.args.dataset_name}.csv")
        df = pd.read_csv(csv_path, converters={'neighbors': ast.literal_eval})


        # 加载图数据（DGL格式）
        graph_path = os.path.join(self.data_dir, f"{self.args.dataset_name}Graph.pt")
        graph = load_graphs(graph_path)[0][0]

        return df, graph

    def load_image(self, node_id: int) -> Image.Image:
        """加载节点图像"""
        img_path = os.path.join(
            self.data_dir,
            f"{self.args.dataset_name}Images",
            f"{node_id}{self.args.image_ext}"
        )
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")


def get_k_hop_neighbors(nx_graph, node_id, k):
    """
    获取指定中心节点的 k-hop 邻居（不包含中心节点）
    使用 NetworkX 的 single_source_shortest_path_length 方法。
    """
    neighbors = set()
    for target, distance in nx.single_source_shortest_path_length(nx_graph, node_id).items():
        if 0 < distance <= k:
            neighbors.add(target)
    return list(neighbors)


def build_classification_prompt_with_neighbors(center_text: str, neighbor_texts: list, neighbor_images: list, classes: list, add_cot: bool, use_center_text: bool, use_center_image: bool) -> str:
    """
    Build a RAG-enhanced classification prompt by integrating the center node's text with its neighbors' information.
    """
    if neighbor_images:
        prompt = "These are the images related to the center node and its neighbor nodes.\n"
    elif use_center_image:
        prompt = "This is the image of the center node.\n"
    else:
        prompt = ""
    # 2️⃣ **中心节点文本**
    if use_center_text:
        prompt += f"\nDescription of the center node: {center_text}\n"

    if neighbor_texts:
        prompt += "\nBelow are descriptions of the neighbor nodes:\n"
        for idx, n_text in enumerate(neighbor_texts):
            prompt += f"Neighbor {idx+1}: {n_text}\n"

    prompt += f"\nAvailable categories: {', '.join(classes)}.\n"

    if neighbor_texts and neighbor_images:
        prompt += "\nConsidering the multimodal information (both text and image) from the center node and its neighbors, determine the most appropriate category."
    elif neighbor_texts:
        prompt += "\nConsidering the center node's multimodal information and the text information from its neighbors, determine the most appropriate category."
    elif neighbor_images:
        prompt += "\nConsidering the center node's multimodal information and the image information from its neighbors, determine the most appropriate category."
    elif use_center_text and use_center_image:
        prompt += "\nConsidering the center node's multimodal information, determine the most appropriate category."
    elif use_center_text:
        prompt += "\nConsidering the center node's text information, determine the most appropriate category."
    elif use_center_image:
        prompt += "\nConsidering the center node's image information, determine the most appropriate category."
    else:
        pass

    if add_cot:
        prompt += "\n\nLet's think step by step."
    # 添加要求仅返回准确的类别名称
    prompt += "\nAnswer ONLY with the exact category name."

    return prompt.strip()



def k_hop_neighbor_stats(nx_graph, k):
    """
    计算图中所有节点的 k 阶邻居数目的统计信息。

    参数：
    - nx_graph: networkx.Graph，输入的无向图
    - k: int，表示计算 k 阶邻居

    返回：
    - stats: dict，包含最小值、最大值、平均值、中位数、标准差
    """
    if nx_graph is None:
        raise ValueError("输入的图数据不能为空")

    all_k_hop_counts = []

    for node in nx_graph.nodes():
        k_hop_neighbors = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=k).keys())
        k_hop_neighbors.discard(node)  # 移除自身
        all_k_hop_counts.append(len(k_hop_neighbors))

    stats = {
        "min": np.min(all_k_hop_counts),
        "max": np.max(all_k_hop_counts),
        "mean": np.mean(all_k_hop_counts),
        "median": np.median(all_k_hop_counts),
        "std": np.std(all_k_hop_counts)
    }

    return stats


def print_k_hop_stats(nx_graph, ks=[1, 2, 3]):
    """打印 1, 2, 3 阶邻居的统计信息"""
    for k in ks:
        stats = k_hop_neighbor_stats(nx_graph, k)
        print(f"\n{k} 阶邻居统计信息：")
        print(f"  最小邻居数: {stats['min']}")
        print(f"  最大邻居数: {stats['max']}")
        print(f"  平均邻居数: {stats['mean']:.2f}")
        print(f"  中位数: {stats['median']}")
        print(f"  标准差: {stats['std']:.2f}")


def find_isolated_nodes(dgl_graph):
    """判断 DGL 图是否存在孤立点，并返回孤立点的节点 ID 和数量"""
    in_degrees = dgl_graph.in_degrees()
    out_degrees = dgl_graph.out_degrees()

    # 识别孤立点：入度和出度都为 0
    isolated_mask = (in_degrees == 0) & (out_degrees == 0)
    isolated_nodes = torch.nonzero(isolated_mask, as_tuple=True)[0]  # 获取孤立点 ID

    num_isolated_nodes = isolated_nodes.numel()  # 孤立点的总数

    if num_isolated_nodes > 0:
        print(f"存在 {num_isolated_nodes} 个孤立点")
        print(f"孤立点节点 ID: {isolated_nodes.tolist()}")
    else:
        print("图中没有孤立点")

    return isolated_nodes.tolist(), num_isolated_nodes


def main(args):
    start_time = time.time()  # 记录起始时间
    # 初始化数据加载器
    dataset_loader = DatasetLoader(args)
    df, dgl_graph = dataset_loader.load_data() # 这里得到的两个数据分别是 CSV 文件中的 DataFrame 和 DGL 格式的图数据
    # dgl格式的图都有
    # 预定义规则
    TEXT_COLUMN_RULES = ["text", "caption"]
    LABEL_COLUMN_RULES = ["second_category", "subreddit"]

    # 根据规则选择 text_column
    text_column = next((col for col in TEXT_COLUMN_RULES if col in df.columns), None)
    text_label_column = next((col for col in LABEL_COLUMN_RULES if col in df.columns), None)

    # 检查是否找到合适的列
    if text_column is None:
        raise ValueError(f"数据集中未找到合适的文本列，请检查数据集列名: {df.columns}")

    if text_label_column is None:
        raise ValueError(f"数据集中未找到合适的标签列，请检查数据集列名: {df.columns}")

    print(f"使用的文本列: {text_column}")
    print(f"使用的标签列: {text_label_column}")

    # 从CSV中提取所有唯一类别，并排序（可根据需要调整顺序）
    classes = sorted(df[text_label_column].str.lower().unique())
    # 构建一个从节点ID到节点数据的字典，便于后续查找邻居信息
    # 假设 CSV 中 "id" 列作为唯一标识符，且 "text" 为节点描述
    node_data_dict = {row["id"]: row for _, row in df.iterrows()}
    # dataframe里的iterrows()方法会返回每行的索引和内容，row["id"] 就是每行数据中 "id" 列的值，row["text"] 就是 "text" 列的值。通过这个字典，我们可以快速根据节点 ID 获取对应的文本描述和其他信息。

    isolated_nodes, num_isolated = find_isolated_nodes(dgl_graph) # 提取孤立点信息
    print(f"孤立点数量: {num_isolated}, 孤立点 ID: {isolated_nodes}")
    # 如果使用 RAG 增强推理，转换 DGL 图为 NetworkX 图
    if args.num_neighbours > 0:
        # # 添加反向边，转换为无向图
        # srcs, dsts = dgl_graph.all_edges()
        # dgl_graph.add_edges(dsts, srcs)
        dgl_graph.ndata["_ID"] = torch.arange(dgl_graph.num_nodes())

        nx_graph = dgl.to_networkx(dgl_graph, node_attrs=['_ID'])  # 根据实际情况设置节点属性
    else:
        nx_graph = None
    # print_k_hop_stats(nx_graph)

    model_name = args.model_name  # 这里可以传入你的模型名称
    model, processor = load_model_and_processor(model_name)


    # 初始化计数器
    y_true = []
    y_pred = []
    total_samples = 0
    mismatch_count = 0  # 统计预测类别完全不匹配的情况

    # 进行数据集划分
    train_ids, val_ids, test_ids = split_dataset(
        nodes_num=len(df),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    selected_ids = test_ids  # 这里可以选择 train_ids, val_ids, 或 test_ids
    sample_df = df.iloc[selected_ids]  # 使用 selected_ids 来选择相应的数据集
    # 如果 num_samples 为 0，使用全部 selected_ids
    num_samples = args.num_samples if args.num_samples > 0 else len(sample_df)
    # 输出用于调试
    print(f"Selected {num_samples} samples out of {len(sample_df)} available samples.")
    # 从所选的子集数据中，再选择前 num_samples 个样本
    sample_df = sample_df.head(num_samples)  # 选择前 num_samples 个样本
    add_CoT = True if str(args.add_CoT).lower() == "true" else False   # 是否添加简单的思维链提示
    print(f"Adding Chain of Thought: {add_CoT}")

    upload_image = True if str(args.upload_image).lower() == "true" else False   # 是否上传图片到WandB
    use_center_text = True if str(args.use_center_text).lower() == "true" else False   # 是否使用中心节点文本
    use_center_image = True if str(args.use_center_image).lower() == "true" else False   # 是否使用中心节点图像

    if upload_image:
        table = wandb.Table(columns=["node_id", "Image", "Neighbor_Images", "input", "ground_truth", "prediction_output",
                                     "predicted_class"])
    else:
        table = wandb.Table(columns=["node_id", "input", "ground_truth", "prediction_output", "predicted_class"])

    set_seed(42)  # 设置随机种子以确保结果可重现

    if args.num_neighbours > 0:
        neighbor_dict = {}  # 用来存储每个节点的邻居 ID 列表
        max_hop = 3  # 最大跳数
        for node_id in tqdm(sample_df["id"], desc="Fetching neighbors"):
            sampled_neighbors = set()  # 用 set 存储，去重
            k = args.num_neighbours  # 需要的邻居数量
            current_hop = 1  # 从 1-hop 开始

            while len(sampled_neighbors) < k and current_hop <= max_hop:
                # 获取当前 hop 的邻居
                neighbors_at_current_hop = set(nx_graph.neighbors(node_id))  # 使用 set 防止重复
                neighbors_at_current_hop.discard(node_id)  # 🔥 关键：去除自身 ID
                sampled_neighbors.update(neighbors_at_current_hop)  # 添加新邻居
                current_hop += 1  # 继续寻找更远的邻居

            # 只保留前 k 个邻居
            neighbor_dict[node_id] = list(sampled_neighbors)[:k]

    for idx, row in tqdm(sample_df.iterrows(), total=sample_df.shape[0], desc="Processing samples"):
        try:
            node_id = row["id"]
            center_text = row[text_column]
            text_label = row[text_label_column].lower()  # 文本类别标签

            # 加载图像
            center_image = dataset_loader.load_image(node_id)

            # 初始化存储邻居数据的变量
            neighbor_texts = []
            neighbor_images = []

            # **构造输入的 messages**
            messages = [{"role": "user", "content": [{"type": "image", "image": center_image}]}]

            if args.num_neighbours > 0:
                # 获取节点的邻居 ID
                sampled_neighbor_ids = neighbor_dict.get(node_id, [])

                for nid in sampled_neighbor_ids:
                    if nid in node_data_dict:
                        if nid == node_id:
                            warnings.warn(
                                f"采样到的邻居 ID ({nid}) 与当前节点 ID ({node_id}) 相同，可能存在自环或重复采样情况！")
                        else:
                            node_info = node_data_dict[nid]

                            # 处理邻居文本
                            if args.neighbor_mode in ["text", "both"]:
                                text = str(node_info.get(text_column, ""))
                                neighbor_texts.append(text)

                            # 处理邻居图像（正确加载）
                            if args.neighbor_mode in ["image", "both"]:
                                try:
                                    image = dataset_loader.load_image(nid)  # 通过 dataset_loader 正确加载邻居图像
                                    neighbor_images.append(image)
                                except Exception as e:
                                    print(f"加载邻居 {nid} 的图像失败: {e}")
                if args.neighbor_mode in ["image", "both"]:
                    for img in neighbor_images:
                        messages[0]["content"].append({"type": "image", "image": img})
                    images = [center_image] + neighbor_images

                # 构造最终的提示文本
                prompt_text = build_classification_prompt_with_neighbors(center_text, neighbor_texts, neighbor_images, classes, add_CoT, True, True)
            else:
                # 使用基本提示，不进行邻居增强
                prompt_text = build_classification_prompt_with_neighbors(center_text, neighbor_texts, neighbor_images, classes, add_CoT, use_center_text, use_center_image)


            messages[0]["content"].append({"type": "text", "text": prompt_text})

            # **使用处理器生成输入文本, 对于LLaMA，LLaVA，Qwen OK**
            input_text = processor.apply_chat_template(messages, add_generation_prompt=False)

            # 我想写一个函数，接收上面的messages，input_text, imges等信息，然后输出经过对应processor返回的inputs，用于后续传入model


            # **处理图像和文本输入**
            if args.neighbor_mode in ["image", "both"] and args.num_neighbours > 0:
                inputs = prepare_inputs_for_model(messages, input_text, images, center_image, processor, model, args, model_name)
            else:
                inputs = prepare_inputs_for_model(messages, input_text, None, center_image, processor, model, args, model_name)

            # 生成预测结果
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            output_tokens = output[0][len(inputs["input_ids"][0]):]
            prediction = processor.decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip().lower()
            # prediction = processor.decode(output[0], skip_special_tokens=True).strip().lower()

            # 简单解析预测结果，匹配类别列表中的关键词
            # print("Prediction:", prediction)
            predicted_class = next((c for c in classes if c in prediction), None)
            # print("Predicted Class:", predicted_class)

            total_samples += 1
            # 计算完全不匹配的情况
            if predicted_class is None:  # 预测结果与类别列表完全不匹配
                mismatch_count += 1

            # 收集真实值和预测值
            y_true.append(text_label)
            y_pred.append(predicted_class if predicted_class else "unknown")  # 用 "unknown" 代替未匹配的类别

            # ✅ 记录到 wandb.Table
            if args.upload_image:
                image_wandb = wandb.Image(center_image, caption=f"Node {node_id}")  # 转换为 WandB 格式

                neighbor_images_wandb = []
                if args.neighbor_mode in ["image", "both"] and args.num_neighbours > 0:
                    for i, neighbor_img in enumerate(neighbor_images):
                        neighbor_images_wandb.append(wandb.Image(neighbor_img, caption=f"Neighbor {i+1}"))
                else:
                    neighbor_images_wandb = None  # 仅文本模式时，不加入邻居图像

                table.add_data(node_id, image_wandb, neighbor_images_wandb, input_text, text_label, prediction, predicted_class if predicted_class else "unknown")
            else:
                table.add_data(node_id, input_text, text_label, prediction, predicted_class if predicted_class else "unknown")


        except Exception as e:
            print(f"Error processing node {node_id}: {str(e)}")

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算 Macro-F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 计算不匹配概率
    mismatch_probability = mismatch_count / total_samples

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Mismatch Probability: {mismatch_probability:.4f}")

    # ✅ 将 Table 记录到 wandb
    wandb.log({"predictions_table": table})

    wandb.log({
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mismatch_probability": mismatch_probability
    })

    # 结束 wandb 运行
    wandb.finish()
    # 记录结束时间并计算耗时
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time spent: {total_time:.2f} seconds")


if __name__ == "__main__":
    args = parse_args()
    wandb.init(config=args, reinit=True)
    main(args)
