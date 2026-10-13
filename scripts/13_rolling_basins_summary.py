"""Summarise the annealed quarterly rolling refit (output of 12_rolling_basins.py).

Per window: the best start's structure (B, NMI with GICS, Liu core-periphery share,
per-block assortative share), the within-window agreement between starts (the
optimiser "noise floor"), and the DL gain over the original quick fit (03).
Between consecutive windows: agreement of the best partitions, compared with that
noise floor and with hierarchical / Louvain baselines. Also writes the partitions in
03's format so the backtest can be re-run on them:
    python scripts/05_backtest.py --fits results/rolling_basins_partitions.json --hold 63 --tag _quarterly

Usage:  python scripts/13_rolling_basins_summary.py
Outputs: results/rolling_basins_summary.csv|json, results/rolling_basins_partitions.json,
         figures/rolling_basins.{png,pdf}
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import compare as cmp  # noqa: E402
from src import interactions as ix  # noqa: E402
from src.data import DATA  # noqa: E402
from src.plotting import GREY, PALETTE, STRESS_SHADE, plt, savefig  # noqa: E402
from src.preprocess import spy_residuals  # noqa: E402

WINDOW, K_NN = 252, 10
CRASH = (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"))


def main():
    runs = pd.DataFrame(json.load(open(ROOT / "results" / "rolling_basins.json")))
    old = {f["date"]: f for f in json.load(open(ROOT / "results" / "rolling_partitions.json"))}
    returns = pd.read_parquet(DATA / "log_returns.parquet")
    stocks = returns.drop(columns="SPY")
    meta = pd.read_csv(DATA / "universe.csv", index_col=0).loc[stocks.columns]
    sector, subind = cmp.labels_from(meta["sector"]), cmp.labels_from(meta["sub_industry"])

    rows, parts_out, prev = [], [], None
    for date, g in runs.sort_values("date").groupby("date", sort=True):
        end = returns.index.get_loc(pd.Timestamp(date))
        win = returns.iloc[end - WINDOW:end]
        corr = spy_residuals(win.drop(columns="SPY"), win["SPY"]).corr().values
        starts = [np.array(s) for s in g.sort_values("dl")["sbm"]]
        best, best_l1 = starts[0], np.array(g.sort_values("dl")["sbm_level1"].iloc[0])
        B = int(best.max() + 1)
        hier = cmp.hierarchical(corr, B)
        louv = cmp.louvain_knn(corr, K_NN)
        blk = cmp.assortativity_summary(cmp.block_mean_matrix(corr, best), np.bincount(best))
        fr = ix.classify(corr, best)[2]
        within = [cmp.nmi(a, b) for a, b in combinations(starts, 2)]
        row = {"date": pd.Timestamp(date), "window_start": win.index[0], "B": B,
               "dl_best": g.dl.min(), "dl_spread": g.dl.max() - g.dl.min(),
               "dl_gain_vs_quick_fit": old[date]["dl_nats"] - g.dl.min() if date in old else np.nan,
               "nmi_best_vs_quick_fit": cmp.nmi(best, old[date]["sbm"]) if date in old else np.nan,
               "nmi_sector": cmp.nmi(best, sector), "nmi_subind": cmp.nmi(best, subind),
               "pairs_CP": fr["core-periphery"], "pairs_A": fr["assortative"],
               "blocks_assortative": blk["frac_assortative_blocks"],
               "nmi_within_mean": np.mean(within), "nmi_within_min": np.min(within)}
        if prev is not None:
            row["nmi_between_best"] = cmp.nmi(best, prev["best"])
            row["nmi_between_all"] = np.mean([cmp.nmi(a, b) for a in starts for b in prev["starts"]])
            row["nmi_between_hier"] = cmp.nmi(hier, prev["hier"])
            row["nmi_between_louvain"] = cmp.nmi(louv, prev["louv"])
        rows.append(row)
        prev = {"best": best, "starts": starts, "hier": hier, "louv": louv}
        parts_out.append({"date": date, "dl_nats": float(g.dl.min()), "sbm": best.tolist(),
                          "sbm_level1": best_l1.tolist(), "louvain_knn": louv.tolist(),
                          "hier_B": hier.tolist(), "hier_11": cmp.hierarchical(corr, 11).tolist()})

    df = pd.DataFrame(rows).set_index("date")
    df.to_csv(ROOT / "results" / "rolling_basins_summary.csv", float_format="%.4f")
    (ROOT / "results" / "rolling_basins_partitions.json").write_text(json.dumps(parts_out))
    crash = (df.window_start <= CRASH[1]) & (df.index >= CRASH[0])
    summ = {"n_windows": len(df), "starts_per_window": int(runs.groupby("date").size().iloc[0]),
            "median_dl_gain_vs_quick_fit": float(df.dl_gain_vs_quick_fit.median()),
            "median_dl_spread_between_starts": float(df.dl_spread.median()),
            "nmi_within_window_mean": float(df.nmi_within_mean.mean()),
            "nmi_between_windows_best": float(df.nmi_between_best.mean()),
            "nmi_between_windows_all_starts": float(df.nmi_between_all.mean()),
            "nmi_between_windows_hier": float(df.nmi_between_hier.mean()),
            "nmi_between_windows_louvain": float(df.nmi_between_louvain.mean()),
            "crash_vs_other": {c: [float(df.loc[crash, c].mean()), float(df.loc[~crash, c].mean())]
                               for c in ["nmi_sector", "nmi_subind", "pairs_CP", "B", "nmi_between_best"]}}
    json.dump(summ, open(ROOT / "results" / "rolling_basins_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax in axes:
        ax.axvspan(CRASH[0], CRASH[1] + pd.Timedelta(days=365), color=STRESS_SHADE, lw=0, zorder=0)
    ax = axes[0]
    ax.fill_between(df.index, df.nmi_within_min, df.nmi_within_mean, color=GREY, alpha=0.3,
                    label="same window, different starts (min-mean): noise floor")
    ax.plot(df.index, df.nmi_between_best, color=PALETTE[0], label="SBM: consecutive quarters (best starts)")
    ax.plot(df.index, df.nmi_between_hier, color=PALETTE[1], label="hierarchical: consecutive quarters")
    ax.set_ylabel("NMI")
    ax.set_title("Partition stability vs optimiser noise (shaded: windows containing Feb-Apr 2020)", loc="left")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    ax.plot(df.index, df.nmi_sector, color=PALETTE[0], label="vs GICS sector")
    ax.plot(df.index, df.nmi_subind, color=PALETTE[2], label="vs GICS sub-industry")
    ax.set_ylabel("NMI of best partition")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[2]
    ax.plot(df.index, 100 * df.pairs_CP, color=PALETTE[3])
    ax.set_ylabel("% block pairs core-periphery\n(Liu et al. 2023)")
    fig.tight_layout()
    savefig(fig, ROOT / "figures" / "rolling_basins")


if __name__ == "__main__":
    main()
