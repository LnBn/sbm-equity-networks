"""Pairwise interaction types between blocks (Liu, Alexander & Altmann 2023, PNAS Nexus 2, pgad364).

For a partition into blocks, omega[r, s] is the mean edge weight between blocks r and s
(here: mean market-residual correlation; within a block, the mean over distinct pairs).
For an undirected graph every pair of blocks (r, s) is one of three types (their Eq. 3):

    assortative      omega_rs <  min(omega_rr, omega_ss)
    core-periphery   min(omega_rr, omega_ss) < omega_rs < max(omega_rr, omega_ss)
    disassortative   omega_rs >  max(omega_rr, omega_ss)

(Source-basin needs directed edges, so it cannot occur in a correlation network.)
Blocks with a single stock have no within-block density and are left out.
"""
import graph_tool.all as gt
import numpy as np

from src.compare import block_mean_matrix

TYPES = ["assortative", "core-periphery", "disassortative"]
A, CP, D = 0, 1, 2


def type_matrix(omega, sizes, min_size=2):
    """B x B matrix of type codes (0 A, 1 CP, 2 D); NaN on the diagonal and for small blocks."""
    K = len(omega)
    T = np.full((K, K), np.nan)
    ok = np.asarray(sizes) >= min_size
    for r in range(K):
        for s in range(r + 1, K):
            if not (ok[r] and ok[s]):
                continue
            lo, hi = sorted((omega[r, r], omega[s, s]))
            w = omega[r, s]
            if np.isnan([lo, hi, w]).any():
                continue
            T[r, s] = T[s, r] = A if w < lo else (D if w > hi else CP)
    return T


def type_fractions(T):
    """Share of block pairs of each type (upper triangle only)."""
    v = T[np.triu_indices(len(T), k=1)]
    v = v[~np.isnan(v)]
    return {name: float(np.mean(v == code)) if len(v) else np.nan
            for code, name in enumerate(TYPES)} | {"n_pairs": int(len(v))}


def classify(corr, labels, min_size=2):
    """omega, type matrix and fractions for one partition (labels 0..B-1)."""
    labels = np.unique(labels, return_inverse=True)[1]
    omega = block_mean_matrix(corr, labels)
    sizes = np.bincount(labels)
    T = type_matrix(omega, sizes, min_size)
    return omega, T, type_fractions(T)


def aligned_type_probabilities(corr, partitions, reference, min_size=2):
    """Per-cell type probabilities in the label space of a reference partition.

    Each sampled partition is relabelled to best match `reference`
    (graph-tool's align_partition_labels), classified, and its type matrix is
    accumulated on the blocks it shares with the reference. Returns
    counts[K, K, 3] and n[K, K] (number of samples in which the pair existed).
    """
    reference = np.unique(reference, return_inverse=True)[1]
    K = reference.max() + 1
    counts = np.zeros((K, K, 3))
    n = np.zeros((K, K))
    for b in partitions:
        b = np.asarray(gt.align_partition_labels(np.asarray(b), reference))
        keep = np.unique(b[b < K])                 # labels that map onto reference blocks
        omega = np.full((K, K), np.nan)
        sub = np.isin(b, keep)
        bk = b.copy()
        bk[~sub] = K                               # unmatched blocks pooled out of the way
        full = block_mean_matrix(corr, bk)
        omega[np.ix_(keep, keep)] = full[np.ix_(keep, keep)]
        sizes = np.zeros(K)
        sizes[keep] = np.bincount(bk, minlength=K + 1)[keep]
        T = type_matrix(omega, sizes, min_size)
        for code in range(3):
            counts[..., code] += (T == code)
        n += ~np.isnan(T)
    return counts, n


TYPE_COLOURS = ["#2a78d6", "#eda100", "#c4384b"]          # assortative, core-periphery, disassortative


def plot_type_matrix(omega, T, order, labels, prob=None, title_suffix="", path=None):
    """Two panels in the style of Liu et al. (2023): the block density matrix omega and
    the pairwise interaction-type matrix (opacity = posterior probability of the type,
    if `prob` [K, K, 3] is given). `order` lists the blocks to show, e.g. by omega_rr."""
    import matplotlib.pyplot as plt
    o = np.asarray(order)
    W, TT = omega[np.ix_(o, o)], T[np.ix_(o, o)]
    if prob is not None:
        idx = np.nan_to_num(TT).astype(int)[..., None]
        P = np.take_along_axis(prob[np.ix_(o, o)], idx, 2)[..., 0]
    else:
        P = np.ones_like(TT)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    lim = np.nanmax(np.abs(W))
    im = axes[0].imshow(W, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=axes[0], fraction=0.04, label=r"mean residual correlation $\omega_{rs}$")
    rgba = np.ones(TT.shape + (4,))
    for code, c in enumerate(TYPE_COLOURS):
        m = TT == code
        rgba[m, :3] = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        rgba[m, 3] = 0.25 + 0.75 * np.nan_to_num(P[m])
    axes[1].imshow(rgba, interpolation="nearest")
    for ax in axes:
        ax.set_xticks(range(len(o)), [str(r) for r in o], fontsize=6, rotation=90)
        ax.set_yticks(range(len(o)), [labels[r] for r in o], fontsize=6)
        ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_COLOURS]
    axes[1].legend(handles, TYPES, loc="upper right", fontsize=7, frameon=True)
    axes[0].set_title(r"Block densities $\omega$ (blocks ordered by $\omega_{rr}$, descending)" + title_suffix,
                      fontsize=9)
    axes[1].set_title("Pairwise interaction types (Liu et al. 2023)"
                      + ("; opacity = posterior probability" if prob is not None else "") + title_suffix,
                      fontsize=9)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
