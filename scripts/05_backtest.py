"""Out-of-sample global minimum-variance (GMV) backtest.

At each monthly rebalance date: estimate the covariance from the previous
252 days of raw daily log returns, form GMV weights (shorting allowed), and
hold them for the next 21 trading days. The partitions used by the block
estimators were fitted on the same trailing window only (03_rolling_fits.py).

The headline metric is the annualised realised volatility of the daily
out-of-sample portfolio returns: the GMV portfolio's only goal is low
variance, so a better covariance estimate should give lower realised vol.

Usage:  python scripts/05_backtest.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import compare as cmp  # noqa: E402
from src.data import DATA  # noqa: E402
from src import covariance as cv  # noqa: E402
from src.plotting import PALETTE, STRESS_SHADE, plt  # noqa: E402

WINDOW, HOLD = 252, 21
BOOT_BLOCK, N_BOOT = 21, 2000
# Head-to-head comparisons reported in the README: (candidate, reference, period)
PAIRS = [
    ("LW to SBM blocks", "LW const-corr", "full"),
    ("LW to SBM blocks", "LW to GICS-sector blocks", "full"),
    ("LW to SBM blocks", "LW identity", "full"),
    ("Block: SBM", "Block: GICS sector", "full"),
    ("Block: SBM", "Block: GICS sub-ind.", "full"),
    ("Block: SBM", "Block: hierarchical", "full"),
    ("Block: SBM", "Block: Louvain kNN", "full"),
    ("LW to SBM blocks", "LW const-corr", "2011-2019"),
    ("LW to GICS-sector blocks", "LW const-corr", "2011-2019"),
    ("LW to SBM blocks", "LW to GICS-sector blocks", "2011-2019"),
]


def estimators(X, part):
    """All covariance estimators for one estimation window X (T x N)."""
    return {
        "Sample": cv.sample_cov(X),
        "LW identity": cv.ledoit_wolf_identity(X),
        "LW const-corr": cv.lw_shrink_to_block(X)[0],
        "Block: GICS sector": cv.block_cov(X, part["sector"]),
        "Block: GICS sub-ind.": cv.block_cov(X, part["sub_ind"]),
        "Block: SBM": cv.block_cov(X, part["sbm"]),
        "Block: SBM level 1": cv.block_cov(X, part["sbm_level1"]),
        "Block: hierarchical": cv.block_cov(X, part["hier_B"]),
        "Block: Louvain kNN": cv.block_cov(X, part["louvain_knn"]),
        "LW to GICS-sector blocks": cv.lw_shrink_to_block(X, part["sector"])[0],
        "LW to SBM blocks": cv.lw_shrink_to_block(X, part["sbm"])[0],
    }


def block_bootstrap_ratio(a, b, block=BOOT_BLOCK, n_boot=N_BOOT, seed=0):
    """Circular block bootstrap CI for vol(a) / vol(b) on paired daily returns."""
    rng = np.random.default_rng(seed)
    T = len(a)
    n_blocks = int(np.ceil(T / block))
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:T] % T
        ratios[i] = a[idx].std() / b[idx].std()
    return np.percentile(ratios, [2.5, 97.5])


def main():
    returns = pd.read_parquet(DATA / "log_returns.parquet")
    stocks = returns.drop(columns="SPY")
    simple = np.expm1(stocks)                       # portfolio returns need simple returns
    meta = pd.read_csv(DATA / "universe.csv", index_col=0).loc[stocks.columns]
    fixed = {"sector": cmp.labels_from(meta["sector"]),
             "sub_ind": cmp.labels_from(meta["sub_industry"])}
    fits = json.load(open(ROOT / "results" / "rolling_partitions.json"))

    daily, weights_prev, turnover, gross = {}, {}, {}, {}
    for f in fits:
        end = returns.index.get_loc(pd.Timestamp(f["date"]))
        X = stocks.iloc[end - WINDOW:end].values
        oos = simple.iloc[end:end + HOLD]
        part = dict(fixed, **{k: np.array(f[k]) for k in
                              ["sbm", "sbm_level1", "hier_B", "louvain_knn"]})
        for name, cov in estimators(X, part).items():
            w = cv.gmv_weights(cov)
            # Constant weights over the holding month (daily rebalanced back to w).
            daily.setdefault(name, []).append(pd.Series(oos.values @ w, index=oos.index))
            if name in weights_prev:
                turnover.setdefault(name, []).append(np.abs(w - weights_prev[name]).sum())
            weights_prev[name] = w
            gross.setdefault(name, []).append(np.abs(w).sum())

    port = pd.DataFrame({k: pd.concat(v) for k, v in daily.items()})
    port.to_parquet(ROOT / "results" / "gmv_daily_returns.parquet")
    ann = np.sqrt(252)
    periods = {"full": (None, None), "2011-2019": ("2011", "2019"),
               "2020": ("2020", "2020"), "2021-2026": ("2021", "2026")}
    table = pd.DataFrame({p: port.loc[a:b].std() * ann * 100 for p, (a, b) in periods.items()})
    table.columns = [f"vol% {p}" for p in table.columns]
    table["mean turnover"] = pd.Series({k: np.mean(v) for k, v in turnover.items()}).round(2)
    table["mean gross lev."] = pd.Series({k: np.mean(v) for k, v in gross.items()}).round(2)
    ref = "LW const-corr"
    cis = {k: block_bootstrap_ratio(port[k].values, port[ref].values) for k in port}
    table[f"vol ratio vs {ref} [95% CI]"] = [
        f"{port[k].std() / port[ref].std():.3f} [{cis[k][0]:.3f}, {cis[k][1]:.3f}]"
        for k in table.index]
    table = table.round(2).sort_values("vol% full")
    table.to_csv(ROOT / "results" / "backtest_summary.csv")
    print(f"{len(fits)} rebalances, {port.index[0].date()} to {port.index[-1].date()}")
    print(table.to_string())

    rows = []
    for a, b, period in PAIRS:
        start, end = periods[period]
        pa, pb = port[a].loc[start:end].values, port[b].loc[start:end].values
        lo, hi = block_bootstrap_ratio(pa, pb)
        rows.append({"candidate": a, "reference": b, "period": period,
                     "vol_ratio": round(pa.std() / pb.std(), 3),
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(ROOT / "results" / "backtest_pairwise.csv", index=False)
    print(pairwise.to_string(index=False))
    plot(port)


def plot(port):
    """Rolling 1-year realised vol of selected GMV portfolios."""
    show = ["LW const-corr", "LW identity", "Block: GICS sector", "Block: SBM",
            "LW to SBM blocks"]
    roll = port[show].rolling(252).std().dropna() * np.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"),
               color=STRESS_SHADE, lw=0, zorder=0)
    for name, colour in zip(show, PALETTE):
        ax.plot(roll.index, roll[name], color=colour, label=name)
    ax.set_ylabel("trailing 1-year realised vol (%)")
    ax.set_title("Out-of-sample GMV volatility by covariance estimator (shaded: Feb-Apr 2020)",
                 loc="left")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "backtest_rolling_vol.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
