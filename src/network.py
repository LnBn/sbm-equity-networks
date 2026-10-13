"""Build graph-tool graphs from a correlation matrix.

Two constructions are used:
  * dense: the complete graph, every pair (i, j) carries its correlation as a
    real-valued edge covariate. Edge *presence* carries no information here,
    so all the signal is in the weights.
  * knn: each stock links to its k most-correlated peers (union, so the graph
    is undirected). Edge presence now carries the signal, which makes degree
    correction and the plain (unweighted) SBM meaningful.
"""
import graph_tool.all as gt
import numpy as np


def dense_graph(corr):
    """Complete undirected graph with edge property 'w' = correlation."""
    C = np.asarray(corr)
    n = C.shape[0]
    i, j = np.triu_indices(n, k=1)
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    w = g.new_edge_property("double")
    g.add_edge_list(np.column_stack([i, j, C[i, j]]), eprops=[w])
    g.ep["w"] = w
    return g


def knn_edges(corr, k):
    """Undirected edge list (i < j) linking each node to its k most-correlated peers."""
    C = np.array(corr, dtype=float)
    np.fill_diagonal(C, -np.inf)
    nbrs = np.argsort(-C, axis=1)[:, :k]
    edges = {tuple(sorted((i, j))) for i in range(len(C)) for j in nbrs[i]}
    return np.array(sorted(edges))


def knn_graph(corr, k=10):
    """Union k-nearest-neighbour graph with edge property 'w' = correlation."""
    C = np.asarray(corr)
    edges = knn_edges(C, k)
    g = gt.Graph(directed=False)
    g.add_vertex(C.shape[0])
    w = g.new_edge_property("double")
    g.add_edge_list(np.column_stack([edges, C[edges[:, 0], edges[:, 1]]]), eprops=[w])
    g.ep["w"] = w
    return g
