"""Re-fit the rolling windows with annealed starts (quarterly subsample).

03_rolling_fits.py fitted one partition per monthly window with a quick 3-start greedy
fit, which the basin search showed lands in poor local optima. Here every third window
(quarterly; same 252-day length, same end dates as 03) gets N_STARTS independent annealed
starts (src/posterior.anneal_start), run in parallel over all (window, start) pairs.
All starts are kept so the within-window spread can serve as a noise floor.

Usage:  python scripts/12_rolling_basins.py [n_starts] [n_workers]   (~1.5 h with 8 starts, 15 workers)
Output: results/rolling_basins.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import DATA  # noqa: E402
from src.network import dense_graph  # noqa: E402
from src.posterior import anneal_start  # noqa: E402
from src.preprocess import spy_residuals  # noqa: E402

WINDOW, STEP, EVERY = 252, 21, 3        # 03's monthly grid; keep every 3rd window


def job(args):
    end_idx, seed = args
    returns = pd.read_parquet(DATA / "log_returns.parquet")
    win = returns.iloc[end_idx - WINDOW:end_idx]
    corr = spy_residuals(win.drop(columns="SPY"), win["SPY"]).corr().values
    t = time.time()
    state, dl = anneal_start(dense_graph(corr), seed)
    b0 = np.unique(state.get_bs()[0], return_inverse=True)[1]
    level1 = np.unique(np.asarray(state.get_bs()[1])[state.get_bs()[0]], return_inverse=True)[1]
    return {"end_idx": end_idx, "date": str(returns.index[end_idx].date()), "seed": seed,
            "dl": float(dl), "sbm": b0.tolist(), "sbm_level1": level1.tolist(),
            "seconds": time.time() - t}


def main(n_starts=8, n_workers=15):
    n_days = len(pd.read_parquet(DATA / "log_returns.parquet"))
    ends = list(range(WINDOW, n_days, STEP))[::EVERY]
    jobs = [(e, 2000 + k) for e in ends for k in range(n_starts)]
    print(f"{len(ends)} windows x {n_starts} starts = {len(jobs)} jobs", flush=True)
    out, done = [], 0
    with ProcessPoolExecutor(n_workers) as pool:
        for r in pool.map(job, jobs):
            out.append(r)
            done += 1
            if done % n_starts == 0:
                print(f"{done}/{len(jobs)} done (last window {r['date']})", flush=True)
                (ROOT / "results" / "rolling_basins.json").write_text(json.dumps(out))   # checkpoint
    (ROOT / "results" / "rolling_basins.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
