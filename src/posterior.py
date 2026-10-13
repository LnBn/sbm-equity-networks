"""Posterior sampling for the nested weighted SBM, and summaries of the samples.

The pilot reports one partition per window: the lowest description length (DL)
found by the agglomerative heuristic plus zero-temperature (beta = inf)
merge-split sweeps. Here we instead sample partitions from the posterior
P(b | A) ∝ exp(-DL(b)) by running the same merge-split moves at beta = 1, so
every move that raises the DL by d is still accepted with probability ~e^{-d}.

Chains start from different places (the MAP fit, and independent fits with other
seeds), run a burn-in, then record one partition hierarchy per call to
multiflip_mcmc_sweep(beta=1, niter=10). Agreement between chains is the
convergence check.
"""
import graph_tool.all as gt
import numpy as np

from src.sbm import fit_sbm


def run_chain(g, seed, start="map", n_burn=500, n_samples=1000, niter=10,
              fit_kwargs=None):
    """One MCMC chain at beta = 1.

    start : "map" starts from the MAP fit made with seed 42 (same as the pilot);
            "fresh" starts from an independent fit made with `seed`.
    Returns a dict with the DL trace (burn-in and sampling) and the sampled
    hierarchies (one list of per-level block arrays per sample).
    """
    fit_kwargs = dict(weighted=True, deg_corr=False, nested=True, n_init=5, n_sweeps=50,
                      **(fit_kwargs or {}))
    state, dl0 = fit_sbm(g, seed=42 if start == "map" else seed, **fit_kwargs)
    gt.seed_rng(seed)
    np.random.seed(seed)
    burn = np.empty(n_burn)
    for t in range(n_burn):
        state.multiflip_mcmc_sweep(beta=1, niter=niter)
        burn[t] = state.entropy()
    dl, samples = np.empty(n_samples), []
    for t in range(n_samples):
        state.multiflip_mcmc_sweep(beta=1, niter=niter)
        dl[t] = state.entropy()
        samples.append([np.array(b, dtype=np.int32) for b in state.get_bs()])
    return {"seed": seed, "start": start, "dl_start": dl0, "dl_burn": burn,
            "dl": dl, "samples": samples}


def level0(samples):
    """Level-0 partitions (stock -> block) from a list of sampled hierarchies."""
    return [s[0] for s in samples]


def n_blocks(b):
    return len(np.unique(b))


def mean_pairwise(partitions_a, partitions_b, metric, n_pairs=2000, seed=0):
    """Mean of metric(x, y) over random pairs x from a, y from b (x != y if a is b)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_pairs):
        i, j = rng.integers(len(partitions_a)), rng.integers(len(partitions_b))
        if partitions_a is partitions_b and i == j:
            continue
        vals.append(metric(partitions_a[i], partitions_b[j]))
    return float(np.mean(vals))


def nmi(x, y):
    """Normalised mutual information (arithmetic-mean normalisation)."""
    return float(gt.mutual_information(x, y, norm=True))


def overlap(x, y):
    """Fraction of nodes in the same group after optimal label matching."""
    return float(gt.partition_overlap(x, y, norm=True))


def partition_modes(g, partitions, seed=1):
    """Cluster sampled partitions into modes (Peixoto 2021, ModeClusterState).

    Returns a list of dicts, largest mode first: weight (share of samples),
    the mode's consensus partition, per-node marginal block probabilities, and
    the mean overlap of the mode's members with its centre.
    """
    gt.seed_rng(seed)
    cs = gt.ModeClusterState(partitions)
    gt.mcmc_equilibrate(cs, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    out = []
    for m in cs.get_modes():
        pv = m.get_marginal(g)                 # per node: counts over the mode's aligned labels
        counts = [np.asarray(pv[v], dtype=float) for v in g.vertices()]
        P = np.zeros((g.num_vertices(), max(len(c) for c in counts)))
        for i, c in enumerate(counts):
            P[i, :len(c)] = c
        P /= P.sum(axis=1, keepdims=True)
        # consensus partition = each node's most probable label (relabelled 0..B-1)
        cons = np.unique(P.argmax(axis=1), return_inverse=True)[1]
        out.append({"weight": m.get_M() / len(partitions), "partition": cons,
                    "marginals": P, "B": int(cons.max() + 1)})
    return sorted(out, key=lambda d: -d["weight"])
