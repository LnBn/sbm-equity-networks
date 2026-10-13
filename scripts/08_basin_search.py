"""Global search for low-DL partitions (basins) of the nested weighted SBM, in parallel.

Why: at beta = 1 neither multiflip nor multilevel merge-split moves cross between the
local optima found for this network (gaps of 40-80 nats, i.e. posterior odds of e^40+),
so chains stay in whichever basin they start in. The posterior is therefore dominated
by the lowest basin found, and the useful questions become: how low can we get, and do
the good basins agree on the structure we report?

Each start: an independent fit (agglomerative heuristic, one initialisation), then a
gentle anneal with multiflip sweeps at beta = 0.5 -> 0.7 -> 0.85 -> 1 (so it can climb
out of shallow basins), then greedy refinement at beta = inf with multiflip and a few
multilevel sweeps. Output: results/posterior/<window>/basins.pkl.gz

Usage:  python scripts/08_basin_search.py calm_2023_24 [n_starts] [n_workers]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gzip
import importlib.util
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import graph_tool.all as gt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sbm import fit_sbm  # noqa: E402

_spec = importlib.util.spec_from_file_location("s06", ROOT / "scripts" / "06_posterior_sample.py")
s06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s06)

ANNEAL = [(0.5, 20), (0.7, 20), (0.85, 20), (1.0, 20)]   # (beta, calls of 10 multiflip sweeps)
GREEDY_MULTIFLIP, GREEDY_MULTILEVEL = 50, 3


def one_start(job):
    window, seed = job
    g, _ = s06.window_graph(window)
    t = time.time()
    state, dl_fit = fit_sbm(g, weighted=True, deg_corr=False, nested=True,
                            n_init=1, n_sweeps=0, seed=seed)
    gt.seed_rng(seed)
    trace = [dl_fit]
    for beta, n in ANNEAL:
        for _ in range(n):
            state.multiflip_mcmc_sweep(beta=beta, niter=10)
        trace.append(state.entropy())
    for _ in range(GREEDY_MULTIFLIP):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
    for _ in range(GREEDY_MULTILEVEL):
        state.multilevel_mcmc_sweep(beta=np.inf, niter=1)
    dl = state.entropy()
    trace.append(dl)
    return {"seed": seed, "dl_fit": dl_fit, "dl": dl, "trace": trace,
            "bs": [np.array(b, dtype=np.int32) for b in state.get_bs()],
            "seconds": time.time() - t}


def main(window, n_starts=30, n_workers=15):
    out = ROOT / "results" / "posterior" / window / "basins.pkl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [1000 + i for i in range(n_starts)]
    results = []
    with ProcessPoolExecutor(n_workers) as pool:
        for r in pool.map(one_start, [(window, s) for s in seeds]):
            results.append(r)
            print(f"seed {r['seed']}: fit {r['dl_fit']:.1f} -> final {r['dl']:.1f} "
                  f"(B={len(np.unique(r['bs'][0]))}, {r['seconds']/60:.1f} min)", flush=True)
    with gzip.open(out, "wb") as f:
        pickle.dump({"window": window, "anneal": ANNEAL, "starts": results}, f)
    dls = np.sort([r["dl"] for r in results])
    print(f"best DLs: {np.round(dls[:8], 1)}; spread best-to-median {np.median(dls) - dls[0]:.1f} nats")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], *(int(a) for a in args[1:]))
