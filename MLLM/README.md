# MLLM 使用说明

## 1. 适用范围

本文档覆盖 `./MLLM` 目录下的顶层 Python 代码与其对应的用途：

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `./Zero-shot.py` | 可执行脚本 | 使用多模态大模型进行零样本节点分类 |
| `./train.py` | 可执行脚本 | 训练 Mario 特征提取器 |
| `./FeatureExtractor.py` | 可执行脚本 | 使用训练好的 Mario 模型提取新的文本/图像节点特征 |
| `./graph_shorest_dist_matrix_generate.py` | 可执行脚本 | 生成图的最短路径距离矩阵 |
| `./splits.py` | 可执行脚本 | 将图和特征切分为训练/验证/测试子集 |
| `./check_embedding_size.py` | 可执行脚本 | 检查文本和图像嵌入的形状 |
| `./Library.py` | 模块 | 加载支持的 MLLM 模型与 processor，准备输入 |
| `./Mario.py` | 模块 | Mario 特征提取器与实验性提示模板代码 |
| `./TaskPrompt.py` | 模块 | 存放节点分类与链路预测任务描述模板 |
| `./__init__.py` | 模块 | Python 包标记文件 |

额外说明：

- `./wandb/` 是运行日志目录，不是源代码。
- `./Llama-3.1/README.md` 是第三方模型说明，不属于本目录的核心训练/推理脚本。

## 2. 路径约定

本文档中的命令统一假设你先进入 `./MLLM` 目录：

```bash
cd ./MLLM
```

从这一刻开始，所有路径都使用 Linux 风格相对路径，例如：

- `./Zero-shot.py`
- `../Data/Movies/MoviesGraph.pt`
- `../trained_models/mario_feature_extractor_epoch_500.pth`

## 3. 运行环境

建议使用仓库根目录下的环境配置，并额外确认以下依赖已安装：

- `torch`
- `dgl`
- `transformers`
- `qwen-vl-utils`
- `numpy`
- `pandas`
- `Pillow`
- `networkx`
- `scikit-learn`
- `wandb`

如果你不想把实验上传到 WandB，可以在运行前设置：

```bash
export WANDB_MODE=offline
```

## 4. 推荐执行顺序

如果你的目标是训练 Mario 特征提取器并生成新的节点嵌入，推荐顺序如下：

1. 准备原始图、文本特征、图像特征和图像目录。
2. 使用 `./splits.py` 切分训练/验证/测试数据。
3. 使用 `./graph_shorest_dist_matrix_generate.py` 生成最短路径距离矩阵。
4. 使用 `./train.py` 训练 Mario 特征提取器。
5. 使用 `./FeatureExtractor.py` 提取新的文本/图像节点嵌入。
6. 将生成的嵌入交给下游任务脚本使用，例如 `../run_movies_mario_nc.py` 或 `../run_movies_mario_lp.py`。

如果你的目标是直接做零样本节点分类，只需要准备数据后运行 `./Zero-shot.py`。

## 5. 逐文件使用说明

### 5.1 `./Zero-shot.py`

**用途**

使用多模态大模型对图节点进行零样本节点分类。脚本会：

- 读取 `../Data/<dataset>/<dataset>.csv`
- 读取 `../Data/<dataset>/<dataset>Graph.pt`
- 读取 `../Data/<dataset>/<dataset>Images/`
- 根据 `--neighbor_mode`、`--num_neighbours` 等参数构造提示
- 调用支持的 MLLM 生成分类结果
- 统计准确率、Macro-F1 和 mismatch probability

**输入要求**

对于 `--dataset_name Movies`，默认需要以下相对路径存在：

- `../Data/Movies/Movies.csv`
- `../Data/Movies/MoviesGraph.pt`
- `../Data/Movies/MoviesImages/`

默认图像扩展名是 `.jpg`，可通过 `--image_ext` 修改。

**常用参数**

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_name` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Hugging Face 模型名 |
| `--dataset_name` | `Movies` | 数据集目录名 |
| `--base_dir` | 自动推断到仓库根目录 | 项目根目录 |
| `--max_new_tokens` | `15` | 生成长度上限 |
| `--neighbor_mode` | `both` | 邻居使用文本、图像或二者 |
| `--num_neighbours` | `0` | 使用的邻居数量 |
| `--num_samples` | `5` | 评估样本数，设为 `0` 表示使用完整测试集 |
| `--use_center_text` | `True` | 是否使用中心节点文本 |
| `--use_center_image` | `True` | 是否使用中心节点图像 |
| `--add_CoT` | `False` | 是否追加简单 CoT 提示 |
| `--train_ratio` | `0.6` | 训练集比例 |
| `--val_ratio` | `0.2` | 验证集比例 |

**示例命令**

中心节点多模态零样本分类：

```bash
python ./Zero-shot.py \
  --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --dataset_name Movies \
  --num_samples 300 \
  --max_new_tokens 30
```

使用 1 个邻居的文本增强：

```bash
python ./Zero-shot.py \
  --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --dataset_name Movies \
  --neighbor_mode text \
  --num_neighbours 1 \
  --num_samples 300 \
  --max_new_tokens 30
```

使用 1 个邻居的图像增强：

```bash
python ./Zero-shot.py \
  --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --dataset_name Movies \
  --neighbor_mode image \
  --num_neighbours 1 \
  --num_samples 300 \
  --max_new_tokens 30
```

使用 1 个邻居的多模态增强：

```bash
python ./Zero-shot.py \
  --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --dataset_name Movies \
  --neighbor_mode both \
  --num_neighbours 1 \
  --num_samples 300 \
  --max_new_tokens 30
```

**输出**

- 终端打印准确率、Macro-F1、mismatch probability
- WandB 表格与日志

**备注**

- 该脚本会自动从 `text/caption` 中选择文本列，从 `second_category/subreddit` 中选择标签列。
- 如果 `--num_neighbours 0`，脚本不会构建邻居增强提示。

### 5.2 `./splits.py`

**用途**

把图和节点特征切分为训练/验证/测试子集，并分别保存子图和特征文件。

**当前实现特点**

该脚本当前写死为 `RedditS` 数据集路径，直接运行时会使用：

- `../Data/Reddit/RedditSGraph.pt`
- `../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B-Instruct_visual.npy`
- `../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean.npy`

**直接运行**

```bash
python ./splits.py
```

**输出文件**

- `../Data/Reddit/RedditSGraph_train.pt`
- `../Data/Reddit/RedditSGraph_val.pt`
- `../Data/Reddit/RedditSGraph_test.pt`
- `../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_train.npy`
- `../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_train.npy`
- `../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_val.npy`
- `../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_val.npy`
- `../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_test.npy`
- `../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_test.npy`

**备注**

- 如果你要切换到 `Movies`、`Toys` 或其他数据集，需要先修改脚本里的固定相对路径。
- 该脚本没有命令行参数，属于一次性数据准备工具。

### 5.3 `./graph_shorest_dist_matrix_generate.py`

**用途**

根据图结构生成最短路径距离矩阵，供 `MarioFeatureExtractor` 使用。

**当前实现特点**

该脚本当前写死为 `Movies` 路径，目标文件是：

- 图文件：`../Data/Movies/MoviesGraph.pt`
- 距离矩阵：`../Data/Movies/MoviesDist.pt`

**直接运行**

```bash
python ./graph_shorest_dist_matrix_generate.py
```

**输出文件**

- `../Data/Movies/MoviesDist.pt`

**重要说明**

- 当前脚本是数据集专用的一次性工具。
- 当前版本内部条件判断是固定写法，首次使用前请自行确认生成逻辑与目标路径一致。
- 如果要用于其他数据集，请把脚本中的 `Movies` 路径改成对应数据集的相对路径。

### 5.4 `./train.py`

**用途**

训练 Mario 特征提取器。训练完成后，会定期保存 checkpoint，并最终保存 `mario_feature_extractor_final.pth`。

**默认输入**

该脚本默认使用 `RedditS` 训练子集：

- `../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_train.npy`
- `../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_train.npy`
- `../Data/Reddit/RedditSGraph_train.pt`
- `../Data/Reddit/RedditDist_train.pt`

**常用参数**

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--text_data_path` | `../Data/Reddit/TextFeature/..._train.npy` | 文本训练特征 |
| `--image_data_path` | `../Data/Reddit/ImageFeature/..._train.npy` | 图像训练特征 |
| `--graph_path` | `../Data/Reddit/RedditSGraph_train.pt` | 训练子图 |
| `--dist_matrix_path` | `../Data/Reddit/RedditDist_train.pt` | 最短路径距离矩阵 |
| `--save_path` | `../trained_models` | 模型保存目录 |
| `--model_path` | `None` | 继续训练时加载已有 checkpoint |
| `--num_epochs` | `500` | 训练轮数 |
| `--batch_size` | `32` | batch size |
| `--learning_rate` | `1e-4` | 学习率 |
| `--embed_dim` | `3584` | 输入嵌入维度 |
| `--num_heads` | `8` | 注意力头数 |
| `--buckets_num` | `6` | 距离桶数量 |
| `--layers` | `6` | 重复注意力层数 |

**从头训练示例**

```bash
python ./train.py \
  --text_data_path ../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_train.npy \
  --image_data_path ../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_train.npy \
  --graph_path ../Data/Reddit/RedditSGraph_train.pt \
  --dist_matrix_path ../Data/Reddit/RedditDist_train.pt \
  --save_path ../trained_models \
  --num_epochs 500 \
  --batch_size 32
```

**继续训练示例**

```bash
python ./train.py \
  --text_data_path ../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_train.npy \
  --image_data_path ../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_train.npy \
  --graph_path ../Data/Reddit/RedditSGraph_train.pt \
  --dist_matrix_path ../Data/Reddit/RedditDist_train.pt \
  --model_path ../trained_models/mario_feature_extractor_epoch_450.pth \
  --save_path ../trained_models
```

**输出文件**

- 中间 checkpoint：`../trained_models/mario_feature_extractor_epoch_<N>.pth`
- 最终模型：`../trained_models/mario_feature_extractor_final.pth`

**备注**

- 训练脚本会尝试在 CUDA 上使用 `torch.compile`。
- checkpoint 保存逻辑已经兼容 `torch.compile` 的 `_orig_mod.` 前缀。

### 5.5 `./FeatureExtractor.py`

**用途**

加载训练好的 Mario 特征提取器，对现有文本/图像特征做二次提取，生成新的节点嵌入。

**输入**

脚本通过命令行接收：

- 原始文本特征 `--text_data_path`
- 原始图像特征 `--image_data_path`
- Mario 模型 `--model_path`
- 图文件 `--graph_path`
- 距离矩阵 `--dist_matrix_path`

**示例命令**

以 `Movies` 数据集为例：

```bash
python ./FeatureExtractor.py \
  --text_data_path ../Data/Movies/TextFeature/Movies_Qwen2_VL_7B_Instruct_512_mean.npy \
  --image_data_path ../Data/Movies/ImageFeature/Movies_Qwen2-VL-7B-Instruct_visual.npy \
  --model_path ../trained_models/mario_feature_extractor_epoch_450.pth \
  --graph_path ../Data/Movies/MoviesGraph.pt \
  --dist_matrix_path ../Data/Movies/MoviesDist.pt
```

**当前输出文件**

当前版本的输出路径固定写在脚本内部，为：

- `../Data/Movies/TextFeature/new_text_nodefeatures.npy`
- `../Data/Movies/ImageFeature/new_image_nodefeatures.npy`

**备注**

- 当前输出路径固定为 `Movies`，如果你处理的是其他数据集，需要手动修改脚本中的保存路径。
- 当前 `DataLoader` 使用了 `drop_last=True`。如果样本数不能被 `--batch_size` 整除，最后一个不完整 batch 会被丢弃。
- 当前输出是按 batch 堆叠后的 `npy` 数组。下游脚本 `../run_movies_mario_nc.py` 和 `../run_movies_mario_lp.py` 已经提供了自动 reshape 兼容逻辑。

### 5.6 `./check_embedding_size.py`

**用途**

快速检查文本特征和图像特征的 shape，适合排查模态维度是否一致。

**当前实现特点**

当前脚本固定检查 `RedditS`：

- `../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B-Instruct_visual.npy`
- `../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean.npy`

**直接运行**

```bash
python ./check_embedding_size.py
```

**输出**

- 终端打印图像特征 shape
- 终端打印文本特征 shape

**备注**

- 如果你要检查其他数据集或其他模型的嵌入，需要先修改脚本顶部的固定相对路径。

### 5.7 `./Library.py`

**用途**

为 `./Zero-shot.py` 提供模型加载与输入构造的辅助函数。

**主要函数**

- `load_model_and_processor(model_name)`
- `prepare_inputs_for_model(messages, input_text, images, center_image, processor, model, args, name)`

**当前支持的模型名**

- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2-VL-7B-Instruct`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- `google/paligemma2-3b-pt-448`
- `google/paligemma2-3b-pt-896`

**直接运行**

该文件底部带有一个最小示例，可以直接测试单个模型是否能被成功加载：

```bash
python ./Library.py
```

**备注**

- 该文件主要作为模块被 `./Zero-shot.py` 调用。
- 对于 Qwen 系列，processor 会显式设置 `max_pixels=1280 * 28 * 28`。

### 5.8 `./Mario.py`

**用途**

定义 Mario 特征提取器与实验性提示模板相关类。

**主要类**

| 类 | 说明 |
| --- | --- |
| `MultiHeadAttention` | 基于图最短路径距离桶的位置偏置多头注意力 |
| `RepeatedMultiHeadAttention` | 多层重复堆叠的注意力模块 |
| `FeatureExtractor` | 单模态特征提取器 |
| `MarioFeatureExtractor` | 同时处理文本与图像的 Mario 提取器 |
| `SpeacialTokenProjector` | 特殊 token 投影层 |
| `PromptTemplate` | 根据图结构和节点特征构造提示模板 |
| `MarioPromptTemplate` | 实验性任务模板类 |

**与训练/提取脚本的关系**

- `./train.py` 和 `./FeatureExtractor.py` 直接使用 `MarioFeatureExtractor`
- `PromptTemplate` / `MarioPromptTemplate` 当前并没有被 `./Zero-shot.py` 接入

**重要说明**

- `MarioFeatureExtractor` 依赖最短路径距离矩阵文件，例如 `../Data/Movies/MoviesDist.pt`
- `PromptTemplate` 当前通过 `torch.load()` 读取节点特征文件，这和 `./FeatureExtractor.py` 当前保存的 `.npy` 输出格式并不一致
- `MarioPromptTemplate` 中的 `generate_template()` 和 `return_prompt_template()` 仍是未完成占位实现，不建议直接在当前版本中作为正式入口使用

### 5.9 `./TaskPrompt.py`

**用途**

提供长文本任务描述模板：

- `NC_TASK`：节点分类任务描述
- `LP_TASK`：链路预测任务描述

**直接运行**

该文件本身不是入口脚本，不需要单独执行。通常由 `./Mario.py` 中的模板类引用。

### 5.10 `./__init__.py`

**用途**

将 `./MLLM` 目录标记为 Python 包，没有独立运行逻辑。

## 6. 常见工作流示例

### 6.1 Movies 零样本节点分类

```bash
cd ./MLLM
export WANDB_MODE=offline

python ./Zero-shot.py \
  --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --dataset_name Movies \
  --num_samples 300 \
  --max_new_tokens 30
```

### 6.2 RedditS Mario 特征训练

```bash
cd ./MLLM
export WANDB_MODE=offline

python ./splits.py

python ./train.py \
  --text_data_path ../Data/Reddit/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean_train.npy \
  --image_data_path ../Data/Reddit/ImageFeature/RedditS_Qwen2-VL-7B_Instruct_visual_train.npy \
  --graph_path ../Data/Reddit/RedditSGraph_train.pt \
  --dist_matrix_path ../Data/Reddit/RedditDist_train.pt \
  --save_path ../trained_models
```

### 6.3 Movies Mario 特征提取

```bash
cd ./MLLM
export WANDB_MODE=offline

python ./FeatureExtractor.py \
  --text_data_path ../Data/Movies/TextFeature/Movies_Qwen2_VL_7B_Instruct_512_mean.npy \
  --image_data_path ../Data/Movies/ImageFeature/Movies_Qwen2-VL-7B-Instruct_visual.npy \
  --model_path ../trained_models/mario_feature_extractor_epoch_450.pth \
  --graph_path ../Data/Movies/MoviesGraph.pt \
  --dist_matrix_path ../Data/Movies/MoviesDist.pt
```

## 7. 已知限制

- `./splits.py` 和 `./check_embedding_size.py` 当前是固定数据集脚本，不带命令行参数。
- `./graph_shorest_dist_matrix_generate.py` 当前也是固定数据集脚本，仅适合一次性预处理使用。
- `./FeatureExtractor.py` 当前输出路径固定为 `../Data/Movies/...`。
- `./FeatureExtractor.py` 当前使用 `drop_last=True`，最后一个不完整 batch 会被丢弃。
- `./Mario.py` 里的 `PromptTemplate` 系列代码仍处于实验阶段，其中一部分接口尚未完成。
- `./Zero-shot.py`、`./train.py`、`./FeatureExtractor.py` 都会写入 `./wandb/` 运行日志。

## 8. 与下游任务的衔接

`./FeatureExtractor.py` 生成的新嵌入可以直接用于仓库根目录下的下游任务脚本：

- `../run_movies_mario_nc.py`
- `../run_movies_mario_lp.py`

例如：

```bash
cd ./MLLM

python ../run_movies_mario_nc.py \
  --text_feature_path ../Data/Movies/TextFeature/new_text_nodefeatures.npy \
  --image_feature_path ../Data/Movies/ImageFeature/new_image_nodefeatures.npy \
  --fusion concat \
  --model gcn
```

```bash
cd ./MLLM

python ../run_movies_mario_lp.py \
  --text_feature_path ../Data/Movies/TextFeature/new_text_nodefeatures.npy \
  --image_feature_path ../Data/Movies/ImageFeature/new_image_nodefeatures.npy \
  --fusion concat \
  --model gcn
```
