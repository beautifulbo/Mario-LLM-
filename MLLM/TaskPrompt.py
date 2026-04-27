NC_TASK='''
Node classification is a fundamental task in graph machine learning. Given a graph \( G = (V, E) \), where \( V \) is the set of nodes and \( E \) is the set of edges, each node \( v \in V \) may be associated with a feature vector \( x_v \). In node classification, a subset of nodes \( V_{\text{train}} \subset V \) is labeled with ground-truth categories (e.g., user interests, document topics, or protein functions). The goal is to predict the missing labels for the remaining nodes \( V_{\text{unlabeled}} \) by leveraging:

- The **node features** (if available) – intrinsic attributes of each node.
- The **graph structure** – connections between nodes encode relationships, dependencies, or interactions that often imply label similarity or influence.

This task is semi-supervised by nature, as only a small fraction of nodes typically have labels. Models learn to propagate label information through edges, using inductive biases like homophily (connected nodes tend to share the same label) or structural equivalence. Common approaches include graph neural networks (e.g., GCN, GraphSAGE, GAT), label propagation algorithms, and graph-based regularization.

Applications range from social network analysis (predicting user attributes), citation network classification (paper topics), fraud detection (identifying malicious accounts), to knowledge graph reasoning and biological network analysis.
'''

LP_TASK='''
Link prediction aims to infer missing or future connections between nodes in a graph. Given a graph \( G = (V, E) \), where \( V \) is the set of nodes and \( E \) is the set of observed edges, the task is to predict which unobserved node pairs \((u, v) \notin E\) are likely to form an edge (or will form one in a temporal setting).

The problem is typically framed as a **binary classification** or **ranking** task:
- Positive samples: existing edges (or edges that appear in a future time window).
- Negative samples: randomly sampled non-edges (or edges that never appear).

Key information sources:
- **Topological structure**: Common neighbors, Jaccard coefficient, Adamic–Adar, preferential attachment, and Katz index capture local and global graph patterns.
- **Node features**: If available, similarity or interaction between feature vectors can be used.
- **Latent representations**: Graph neural networks (e.g., GAE, VGAE, SEAL) learn node embeddings that preserve both structural and feature proximity, then predict links via a decoder (e.g., inner product or MLP on node pairs).
'''