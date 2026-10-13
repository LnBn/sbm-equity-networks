"""Sample the posterior of the nested weighted SBM for one pilot window.

Four chains at beta = 1: two start from the MAP fit (seed 42, as in the pilot)
with different MCMC seeds, two start from independent fits with other seeds.
Each chain: N_BURN burn-in calls, then N_SAMPLES recorded calls of
multiflip_mcmc_sweep(beta=1, niter=10). Output: results/posterior/<window>/chains.pkl.gz

Usage:  python scripts/06_posterior_sample.py calm_2023_24   (~20 min on 4 cores)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gzip
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import DATA  # noqa: E402
from src.network import dense_graph  # noqa: E402
from src.posterior import run_chain  # noqa: E402
from src.preprocess import spy_residuals  # noqa: E402

WINDOWS = {"calm_2023_24": ("2023-01-01", "2024-12-31"),
           "stress_2020": ("2020-01-01", "2020-12-31"),
           # one-year calm controls (same length as the 2020 stress window)
           "calm_2023": ("2023-01-01", "2023-12-31"),
           "calm_2024": ("2024-01-01", "2024-12-31")}
CHAINS = [(101, "map"), (102, "map"), (103, "fresh"), (104, "fresh")]
N_BURN, N_SAMPLES = 500, 1000


def window_graph(window):
    returns = pd.read_parquet(DATA / "log_returns.parquet").loc[slice(*WINDOWS[window])]
    corr = spy_residuals(returns.drop(columns="SPY"), returns["SPY"]).corr().values
    return dense_graph(corr), list(returns.drop(columns="SPY").columns)


def chain_job(job):
    window, seed, start = job
    g, _ = window_graph(window)
    t = time.time()
    out = run_chain(g, seed=seed, start=start, n_burn=N_BURN, n_samples=N_SAMPLES)
    out["seconds"] = time.time() - t
    return out


def main(window):
    out_dir = ROOT / "results" / "posterior" / window
    out_dir.mkdir(parents=True, exist_ok=True)
    _, tickers = window_graph(window)
    with ProcessPoolExecutor(len(CHAINS)) as pool:
        chains = list(pool.map(chain_job, [(window, s, st) for s, st in CHAINS]))
    for c in chains:
        print(f"chain seed={c['seed']} start={c['start']}: start DL {c['dl_start']:.1f}, "
              f"sampled DL {c['dl'].mean():.1f} ± {c['dl'].std():.1f}, {c['seconds']/60:.1f} min")
    with gzip.open(out_dir / "chains.pkl.gz", "wb") as f:
        pickle.dump({"window": window, "tickers": tickers, "n_burn": N_BURN,
                     "n_samples": N_SAMPLES, "chains": chains}, f)


if __name__ == "__main__":
    main(sys.argv[1])
