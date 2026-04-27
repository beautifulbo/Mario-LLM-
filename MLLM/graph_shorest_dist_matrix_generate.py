import dgl
import torch

import os

if not os.path.exists("../Data/Movies/MoviesGraph.pt"):
    path= "../Data/Movies/MoviesGraph.pt"

    graph = dgl.load_graphs(path)[0][0]

    dist_matrix = dgl.shortest_dist(graph, root=None, return_paths=False)

    torch.save(dist_matrix, "../Data/Movies/MoviesDist.pt")

# check

dist_matrix_loaded = torch.load("../Data/Movies/MoviesDist.pt")
print(dist_matrix_loaded.shape)