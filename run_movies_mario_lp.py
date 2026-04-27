import argparse
import os

import dgl
import torch
import wandb
from torch_sparse import SparseTensor

from GNN.GraphData import Evaluator, Logger, set_seed, split_edge
from GNN.LinkPrediction.GCN import GCN, LinkPredictor
from GNN.LinkPrediction.SAGE import SAGE
from GNN.Utils.LinkTask import linkprediction
from Utils.mario_feature_utils import FUSION_CHOICES, prepare_feature_matrix, resolve_repo_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run link prediction on Movies with FeatureExtractor-produced embeddings."
    )
    parser.add_argument("--graph_path", type=str, default="Data/Movies/MoviesGraph.pt")
    parser.add_argument("--feature_path", type=str, default=None,
                        help="Use an existing fused feature file directly.")
    parser.add_argument("--text_feature_path", type=str, default="Data/Movies/TextFeature/new_text_nodefeatures.npy")
    parser.add_argument("--image_feature_path", type=str, default="Data/Movies/ImageFeature/new_image_nodefeatures.npy")
    parser.add_argument("--fusion", type=str, choices=FUSION_CHOICES, default="concat")
    parser.add_argument("--prepared_feature_path", type=str, default=None,
                        help="Where to save the prepared 2D feature matrix.")
    parser.add_argument("--skip_save_prepared", action="store_true")
    parser.add_argument("--link_path", type=str, default="Data/LinkPrediction/Movies")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "graphsage", "mlp"],
                        help="Backbone network used for link prediction.")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--neg_len", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.08)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="MAGB-Mario-LP")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline",
                        choices=["online", "offline", "disabled"])
    return parser


class IdentityLPModel(torch.nn.Module):
    def reset_parameters(self):
        return None

    def forward(self, x, adj_t):
        return x


def build_model_and_predictor(args, feat_dim, device):
    if args.model == "gcn":
        model = GCN(feat_dim, args.n_hidden, args.n_hidden, args.n_layers, args.dropout).to(device)
        predictor = LinkPredictor(args.n_hidden, args.n_hidden, 1, 3, args.dropout).to(device)
        return model, predictor
    if args.model == "graphsage":
        model = SAGE(feat_dim, args.n_hidden, args.n_hidden, args.n_layers, args.dropout).to(device)
        predictor = LinkPredictor(args.n_hidden, args.n_hidden, 1, 3, args.dropout).to(device)
        return model, predictor
    if args.model == "mlp":
        model = IdentityLPModel().to(device)
        predictor = LinkPredictor(feat_dim, args.n_hidden, 1, 3, args.dropout).to(device)
        return model, predictor
    raise ValueError(f"Unsupported model: {args.model}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args),
        reinit=True,
    )

    graph_path = resolve_repo_path(args.graph_path)
    link_path = resolve_repo_path(args.link_path)
    link_path.mkdir(parents=True, exist_ok=True)

    graph = dgl.load_graphs(str(graph_path))[0][0]
    feature, prepared_feature_path = prepare_feature_matrix(
        graph_path=graph_path,
        fusion=args.fusion,
        feature_path=args.feature_path,
        text_feature_path=args.text_feature_path,
        image_feature_path=args.image_feature_path,
        prepared_feature_path=args.prepared_feature_path,
        save_prepared=not args.skip_save_prepared,
    )
    print(f"Using feature matrix: {prepared_feature_path}")
    print(f"Feature shape: {tuple(feature.shape)}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")

    edge_split = split_edge(
        graph,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        neg_len=args.neg_len,
        path=str(link_path),
    )

    torch.manual_seed(args.seed)
    idx = torch.randperm(edge_split["train"]["source_node"].numel())[:len(edge_split["valid"]["source_node"])]
    edge_split["eval_train"] = {
        "source_node": edge_split["train"]["source_node"][idx],
        "target_node": edge_split["train"]["target_node"][idx],
        "target_node_neg": edge_split["valid"]["target_node_neg"],
    }

    train_edge_index = torch.stack(
        (edge_split["train"]["source_node"], edge_split["train"]["target_node"]),
        dim=1,
    ).t()

    feat = torch.from_numpy(feature).to(device)
    adj_t = SparseTensor.from_edge_index(train_edge_index).t().to_symmetric().to(device)

    model, predictor = build_model_and_predictor(args, feat.shape[1], device)
    print(f"Using backbone: {args.model}")

    evaluator = Evaluator()
    loggers = {
        "Hits@1": Logger(args.n_runs, args),
        "Hits@3": Logger(args.n_runs, args),
        "Hits@10": Logger(args.n_runs, args),
        "MRR": Logger(args.n_runs, args),
    }

    for run in range(args.n_runs):
        set_seed(args.seed + run)
        model.reset_parameters()
        predictor.reset_parameters()

        loggers = linkprediction(args, adj_t, edge_split, model, predictor, feat, evaluator, loggers, run, args.neg_len)

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(key=key)

    wandb.finish()


if __name__ == "__main__":
    main()
