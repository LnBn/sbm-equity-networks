"""Fit the SBM (and baseline clusterings) on rolling windows.

Every STEP trading days, take the previous WINDOW days, remove the market
(SPY regression), and fit the nested weighted SBM chosen in the pilot.
Only data before the rebalance date is used, so the partitions can be used
out-of-sample in the backtest.

Usage:  python scripts/03_rolling_fits.py     (about 18 minutes on 16 cores)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import compare as cmp  # noqa: E402
from src.data import DATA  # noqa: E402
from src.network import dense_graph  # noqa: E402
from src.preprocess import spy_residuals  # noqa: E402
from src.sbm import fit_sbm, hierarchy_sizes, partition  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
WINDOW, STEP = 252, 21     # one year of data, refit monthly
K_NN = 10


def fit_window(end_idx):
    """Fit all partitions on returns[end_idx - WINDOW : end_idx]."""
    returns = pd.read_parquet(DATA / "log_returns.parquet")
    win = returns.iloc[end_idx - WINDOW:end_idx]
    stocks = win.drop(columns="SPY")
    corr = spy_residuals(stocks, win["SPY"]).corr().values
    state, dl = fit_sbm(dense_graph(corr), weighted=True, deg_corr=False,
                        nested=True, n_init=3, n_sweeps=30, seed=42)
    sbm = partition(state)
    level1 = np.unique(state.project_level(1).get_blocks().a, return_inverse=True)[1]
    B = int(sbm.max() + 1)
    return {
        "date": str(returns.index[end_idx].date()),   # first out-of-sample day
        "dl_nats": dl,
        "levels": hierarchy_sizes(state),
        "sbm": sbm.tolist(),
        "sbm_level1": level1.tolist(),
        "louvain_knn": cmp.louvain_knn(corr, K_NN).tolist(),
        "hier_B": cmp.hierarchical(corr, B).tolist(),
        "hier_11": cmp.hierarchical(corr, 11).tolist(),
    }


def main():
    n_days = len(pd.read_parquet(DATA / "log_returns.parquet"))
    ends = list(range(WINDOW, n_days, STEP))
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as pool:
        fits = list(pool.map(fit_window, ends))
    (ROOT / "results" / "rolling_partitions.json").write_text(json.dumps(fits))
    print(f"fitted {len(fits)} windows; blocks per window: "
          f"median {np.median([max(f['sbm']) + 1 for f in fits]):.0f}")


if __name__ == "__main__":
    main()
