"""Baseline clusterings, partition similarity and block-mixing diagnostics."""
import graph_tool.all as gt
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .network import knn_edges


def labels_from(values):
    """Map any list of labels (e.g. sector names) to integers 0..K-1."""
    _, labels = np.unique(np.asarray(values), return_inverse=True)
    return labels


def nmi(a, b):
    return normalized_mutual_info_score(a, b)


def ari(a, b):
    return adjusted_rand_score(a, b)


# --- Baseline clusterings ---------------------------------------------------

def louvain_dense(corr, seed=0):
    """Louvain on the complete graph with weights max(corr, 0).

    Modularity needs non-negative weights, so negative correlations are dropped.
    """
    C = np.clip(np.asarray(corr), 0, None)
    np.fill_diagonal(C, 0)
    G = nx.from_numpy_array(C)
    return _communities_to_labels(nx.community.louvain_communities(G, seed=seed), len(C))


def louvain_knn(corr, k=10, seed=0):
    """Louvain on the unweighted union k-nearest-neighbour graph."""
    G = nx.Graph()
    G.add_nodes_from(range(len(corr)))
    G.add_edges_from(map(tuple, knn_edges(corr, k)))
    return _communities_to_labels(nx.community.louvain_communities(G, seed=seed), len(corr))


def _communities_to_labels(communities, n):
    labels = np.empty(n, dtype=int)
    for c, members in enumerate(communities):
        labels[list(members)] = c
    return labels


def hierarchical(corr, n_clusters):
    """Average-linkage clustering on the correlation distance sqrt(2(1 - rho))."""
    C = np.asarray(corr)
    D = np.sqrt(np.clip(2 * (1 - C), 0, None))
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=n_clusters, criterion="maxclust") - 1


# --- Description length of a *given* partition ------------------------------

def dl_of_partition(g, labels, weighted=True, deg_corr=False):
    """Description length (nats) of the graph under an SBM with fixed partition.

    Lets us ask: how much worse does GICS (or Louvain) compress the data than
    the SBM's own partition, under the same model?
    """
    b = g.new_vertex_property("int64_t", vals=labels)
    if weighted:
        state = gt.WeightedBlockState(g, b=b, deg_corr=deg_corr,
                                      rec=[g.ep["w"]], rec_types=["real-normal"])
    else:
        state = gt.BlockState(g, b=b, deg_corr=deg_corr)
    return state.entropy()


# --- Mixing pattern ---------------------------------------------------------

def block_mean_matrix(corr, labels):
    """Mean off-diagonal correlation within and between each pair of blocks."""
    C = np.asarray(corr, dtype=float).copy()
    np.fill_diagonal(C, np.nan)
    K = labels.max() + 1
    M = np.full((K, K), np.nan)
    for r in range(K):
        for s in range(K):
            block = C[np.ix_(labels == r, labels == s)]
            if np.isfinite(block).any():
                M[r, s] = np.nanmean(block)
    return M


def assortativity_summary(M, sizes):
    """Share of blocks (of size >= 2) whose internal mean correlation exceeds
    their mean correlation with every other block, plus the overall
    within-minus-between gap."""
    K = len(M)
    ok = sizes >= 2
    is_assortative = [M[r, r] > np.nanmax(np.delete(M[r], r)) for r in range(K) if ok[r]]
    off = M[~np.eye(K, dtype=bool)]
    return {
        "frac_assortative_blocks": float(np.mean(is_assortative)),
        "mean_within": float(np.nanmean(np.diag(M)[ok])),
        "mean_between": float(np.nanmean(off)),
    }
