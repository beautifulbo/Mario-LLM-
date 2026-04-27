import argparse
import copy
import os

import numpy as np
import torch as th
import torch.nn.functional as F
import wandb

from GNN.GraphData import load_data, set_seed
from GNN.Library.GCN import GCN
from GNN.Library.GAT import GAT
from GNN.Library.GraphSAGE import GraphSAGE
from GNN.Utils.NodeClassification import classification
from Utils.mario_feature_utils import FUSION_CHOICES, prepare_feature_matrix, resolve_repo_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run node classification on Movies with FeatureExtractor-produced embeddings."
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
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "graphsage", "gat"],
                        help="Backbone network used for node classification.")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--aggregator", type=str, default="mean",
                        choices=["mean", "gcn", "pool", "lstm"],
                        help="Neighborhood aggregator used when --model graphsage.")
    parser.add_argument("--n-heads", type=int, default=3,
                        help="Number of attention heads used when --model gat.")
    parser.add_argument("--attn-drop", type=float, default=0.0,
                        help="Attention dropout used when --model gat.")
    parser.add_argument("--edge-drop", type=float, default=0.0,
                        help="Edge dropout used when --model gat.")
    parser.add_argument("--no-attn-dst", dest="no_attn_dst", action="store_true",
                        help="Disable destination-node attention term when --model gat.")
    parser.add_argument("--use-attn-dst", dest="no_attn_dst", action="store_false",
                        help="Enable destination-node attention term when --model gat.")
    parser.set_defaults(no_attn_dst=True)
    parser.add_argument("--use-symmetric-norm", action="store_true",
                        help="Enable symmetric normalization when --model gat.")
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="accuracy",
                        choices=["accuracy", "precision", "recall", "f1"])
    parser.add_argument("--average", type=str, default="macro",
                        choices=["weighted", "micro", "macro"])
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--fewshots", type=int, default=None)
    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("--undirected", dest="undirected", action="store_true")
    parser.add_argument("--no-undirected", dest="undirected", action="store_false")
    parser.set_defaults(undirected=True)
    parser.add_argument("--selfloop", dest="selfloop", action="store_true")
    parser.add_argument("--no-selfloop", dest="selfloop", action="store_false")
    parser.set_defaults(selfloop=True)
    parser.add_argument("--wandb_project", type=str, default="MAGB-Mario-NC")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--data_name", type=str, default="Movies")
    return parser


def build_model(args, in_dim, n_classes, device):
    if args.model == "gcn":
        return GCN(in_dim, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout).to(device)
    if args.model == "graphsage":
        return GraphSAGE(
            in_dim,
            args.n_hidden,
            n_classes,
            args.n_layers,
            F.relu,
            args.dropout,
            aggregator_type=args.aggregator,
        ).to(device)
    if args.model == "gat":
        return GAT(
            in_dim,
            n_classes,
            args.n_hidden,
            args.n_layers,
            args.n_heads,
            F.relu,
            args.dropout,
            args.attn_drop,
            args.edge_drop,
            not args.no_attn_dst,
            args.use_symmetric_norm,
        ).to(device)
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

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() and args.gpu != -1 else "cpu")

    graph, labels, train_idx, val_idx, test_idx = load_data(
        str(graph_path),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        name=args.data_name,
        fewshots=args.fewshots,
    )

    if args.undirected:
        print("The Graph change to the undirected graph")
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    observe_graph = copy.deepcopy(graph)

    if args.inductive:
        isolated_nodes = th.cat((val_idx, test_idx))
        sort_isolated_nodes, _ = th.sort(isolated_nodes)
        observe_graph.remove_nodes(sort_isolated_nodes)
        observe_graph.add_nodes(len(sort_isolated_nodes))
        print(observe_graph)
        print("***************")
        print(graph)

    if args.selfloop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
        observe_graph = observe_graph.remove_self_loop().add_self_loop()

    feat = th.from_numpy(feature).to(device)
    n_classes = int((labels.max() + 1).item())
    print(f"Number of classes {n_classes}, Number of features {feat.shape[1]}")

    graph.create_formats_()
    observe_graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)
    observe_graph = observe_graph.to(device)

    print(f"Train_idx: {len(train_idx)}")
    print(f"Valid_idx: {len(val_idx)}")
    print(f"Test_idx: {len(test_idx)}")

    val_results = []
    test_results = []
    model = build_model(args, feat.shape[1], n_classes, device)
    print(f"Using backbone: {args.model}")
    train_numbers = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    print(f"Number of the all GNN model params: {train_numbers}")

    for run in range(args.n_runs):
        set_seed(args.seed + run)
        model.reset_parameters()
        val_result, test_result = classification(
            args, graph, observe_graph, model, feat, labels, train_idx, val_idx, test_idx, run + 1
        )
        wandb.log({f"Val_{args.metric}": val_result, f"Test_{args.metric}": test_result})
        val_results.append(val_result)
        test_results.append(test_result)

    print(f"Runned {args.n_runs} times")
    print(f"Average val {args.metric}: {np.mean(val_results)} ± {np.std(val_results)}")
    print(f"Average test {args.metric}: {np.mean(test_results)} ± {np.std(test_results)}")
    wandb.log(
        {
            f"Mean_Val_{args.metric}": np.mean(val_results),
            f"Mean_Test_{args.metric}": np.mean(test_results),
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
