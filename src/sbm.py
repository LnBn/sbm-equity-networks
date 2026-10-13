"""Bayesian stochastic block model fits with graph-tool.

Each fit runs the agglomerative heuristic (`minimize_*_blockmodel_dl`) from
several random starts, keeps the lowest description length (DL), then polishes
it with zero-temperature merge-split MCMC sweeps. This finds a good point
estimate (the MAP partition); it does not sample the posterior.
"""
import graph_tool.all as gt
import numpy as np


def _state_class_and_args(g, weighted, deg_corr, assortative=False):
    """Pick the graph-tool state class for the requested model variant.

    assortative=True gives the planted-partition model (PPBlockState): groups
    may only be assortative communities, as modularity assumes. It is only
    available for unweighted graphs.
    """
    if assortative:
        if weighted:
            raise ValueError("planted-partition model is unweighted only")
        return gt.PPBlockState, {"deg_corr": deg_corr}
    if not weighted:
        return gt.BlockState, {"deg_corr": deg_corr}
    # Edge weights (correlations) modelled as normal within each block pair.
    args = {"deg_corr": deg_corr, "rec": [g.ep["w"]], "rec_types": ["real-normal"]}
    return gt.WeightedBlockState, args


def fit_sbm(g, weighted=True, deg_corr=False, nested=False, assortative=False,
            n_init=5, n_sweeps=50, seed=42):
    """Fit one SBM variant; return (state, description length in nats).

    Note: gt.seed_rng(0) means 'seed from system entropy' in graph-tool, so
    the seed must be non-zero for reproducible fits.
    """
    if seed == 0:
        raise ValueError("seed=0 makes graph-tool non-deterministic; use a non-zero seed")
    gt.seed_rng(seed)
    np.random.seed(seed)
    cls, args = _state_class_and_args(g, weighted, deg_corr, assortative)
    best, best_dl = None, np.inf
    for _ in range(n_init):
        if nested:
            state = gt.minimize_nested_blockmodel_dl(
                g, base_state=cls, base_state_args=args)
        else:
            state = gt.minimize_blockmodel_dl(g, state=cls, state_args=args)
        # Greedy refinement: beta=inf only accepts moves that lower the DL.
        for _ in range(n_sweeps):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
        dl = state.entropy()
        if dl < best_dl:
            best, best_dl = state, dl
    return best, best_dl


def partition(state):
    """Level-0 block labels as a numpy array, relabelled to 0..B-1."""
    if isinstance(state, gt.NestedBlockState):
        b = state.get_levels()[0].get_blocks().a
    else:
        b = state.get_blocks().a
    _, labels = np.unique(b, return_inverse=True)
    return labels


def hierarchy_sizes(state):
    """Number of non-empty groups at each level of a nested fit (bottom to top)."""
    sizes = []
    for level in range(len(state.get_levels())):
        n_groups = len(np.unique(state.project_level(level).get_blocks().a))
        sizes.append(n_groups)
        if n_groups == 1:
            break
    return sizes
