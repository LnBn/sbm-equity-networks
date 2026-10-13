"""Separate the effect of better partitions from the quarterly backtest design.

The quarterly backtest on the annealed partitions (12/13 + 05 --tag _quarterly) changes two
things at once relative to the original monthly backtest: the rebalance/holding design and
the partitions. This script
  1. writes the ORIGINAL quick-fit partitions (03) at the same quarterly dates
     -> results/rolling_partitions_quarterly_quickfit.json, and
  2. once both quarterly backtests have been run, bootstraps the GMV volatility ratio
     annealed / quick-fit partitions for the SBM-based estimators.

Usage:
  python scripts/15_partition_effect.py
  python scripts/05_backtest.py --fits results/rolling_partitions_quarterly_quickfit.json --hold 63 --tag _quarterly_quickfit
  python scripts/15_partition_effect.py      # second run prints/saves the comparison
Output: results/partition_effect.csv
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s05", ROOT / "scripts" / "05_backtest.py")


def block_bootstrap_ratio(a, b, block=21, n_boot=2000, seed=0):
    """95% circular block-bootstrap CI of std(a)/std(b) (same scheme as 05_backtest.py)."""
    rng = np.random.default_rng(seed)
    T = len(a)
    k = int(np.ceil(T / block))
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([(s + np.arange(block)) % T for s in rng.integers(0, T, k)])[:T]
        out.append(a[idx].std() / b[idx].std())
    return np.percentile(out, [2.5, 97.5])


def main():
    new = json.load(open(ROOT / "results" / "rolling_basins_partitions.json"))
    dates = {f["date"] for f in new}
    old = [f for f in json.load(open(ROOT / "results" / "rolling_partitions.json")) if f["date"] in dates]
    (ROOT / "results" / "rolling_partitions_quarterly_quickfit.json").write_text(json.dumps(old))
    print(f"{len(old)} quick-fit partitions written for the {len(new)} quarterly dates")
    a_path = ROOT / "results" / "gmv_daily_returns_quarterly.parquet"
    b_path = ROOT / "results" / "gmv_daily_returns_quarterly_quickfit.parquet"
    if not (a_path.exists() and b_path.exists()):
        print("run both quarterly backtests (see docstring), then rerun this script")
        return
    a, b = pd.read_parquet(a_path), pd.read_parquet(b_path)
    rows = []
    for k in ["LW to SBM blocks", "Block: SBM", "Block: SBM level 1"]:
        lo, hi = block_bootstrap_ratio(a[k].values, b[k].values)
        rows.append({"estimator": k, "vol_annealed": a[k].std() * np.sqrt(252) * 100,
                     "vol_quickfit": b[k].std() * np.sqrt(252) * 100,
                     "ratio": a[k].std() / b[k].std(), "ci_low": lo, "ci_high": hi})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "partition_effect.csv", index=False, float_format="%.4f")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
