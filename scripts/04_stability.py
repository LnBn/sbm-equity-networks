"""Stability of the rolling SBM partitions and their behaviour around 2020.

Reads results/rolling_partitions.json (from 03_rolling_fits.py).

Usage:  python scripts/04_stability.py
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
from src.plotting import GREY, PALETTE, STRESS_SHADE, plt, savefig  # noqa: E402
from src.preprocess import spy_residuals  # noqa: E402

WINDOW = 252
YEAR_STEPS = 12          # 12 monthly steps = 252 days, i.e. non-overlapping windows
CRASH = (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"))


def main():
    fits = json.load(open(ROOT / "results" / "rolling_partitions.json"))
    returns = pd.read_parquet(DATA / "log_returns.parquet")
    meta = pd.read_csv(DATA / "universe.csv", index_col=0)
    stocks = returns.drop(columns="SPY")
    sector = cmp.labels_from(meta.loc[stocks.columns, "sector"])
    sub_ind = cmp.labels_from(meta.loc[stocks.columns, "sub_industry"])

    rows = []
    for k, f in enumerate(fits):
        date = pd.Timestamp(f["date"])
        end = returns.index.get_loc(date)
        win = returns.iloc[end - WINDOW:end]
        resid_corr = spy_residuals(win.drop(columns="SPY"), win["SPY"]).corr().values
        raw_corr = win.drop(columns="SPY").corr().values
        sbm = np.array(f["sbm"])
        M = cmp.block_mean_matrix(resid_corr, sbm)
        row = {
            "date": date,
            "window_start": win.index[0],
            "B": sbm.max() + 1,
            "B_level1": max(f["sbm_level1"]) + 1,
            "nmi_sbm_sector": cmp.nmi(sbm, sector),
            "nmi_sbm_subind": cmp.nmi(sbm, sub_ind),
            "nmi_hier_sector": cmp.nmi(f["hier_B"], sector),
            "nmi_louvain_sector": cmp.nmi(f["louvain_knn"], sector),
            "frac_assortative": cmp.assortativity_summary(M, np.bincount(sbm))[
                "frac_assortative_blocks"],
            "mean_raw_corr": raw_corr[np.triu_indices_from(raw_corr, 1)].mean(),
        }
        for method in ["sbm", "hier_B", "louvain_knn"]:
            prev = fits[k - 1][method] if k >= 1 else None
            year = fits[k - YEAR_STEPS][method] if k >= YEAR_STEPS else None
            row[f"nmi_month_{method}"] = cmp.nmi(f[method], prev) if prev else np.nan
            row[f"nmi_year_{method}"] = cmp.nmi(f[method], year) if year else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date")
    df.to_csv(ROOT / "results" / "rolling_stability.csv")

    # Windows that contain the Feb-Apr 2020 crash
    has_crash = (df["window_start"] <= CRASH[1]) & (df.index >= CRASH[0])
    # Windows that newly take in crash days (first ~3 months after the crash starts)
    entering = (df.index >= CRASH[0]) & (df.index <= CRASH[1] + pd.Timedelta(days=45))
    summary = {
        "n_windows": len(df),
        "median_B": float(df["B"].median()),
        "B_range": [int(df["B"].min()), int(df["B"].max())],
        "nmi_sector": {m: df[f"nmi_{m}_sector"].agg(["mean", "std"]).round(3).to_dict()
                       for m in ["sbm", "hier", "louvain"]},
        "nmi_sbm_subind_mean": round(df["nmi_sbm_subind"].mean(), 3),
        "nmi_month_mean": {m: round(df[f"nmi_month_{m}"].mean(), 3)
                           for m in ["sbm", "hier_B", "louvain_knn"]},
        "nmi_year_mean": {m: round(df[f"nmi_year_{m}"].mean(), 3)
                          for m in ["sbm", "hier_B", "louvain_knn"]},
        "crash_windows": {
            "nmi_sbm_sector": round(df.loc[has_crash, "nmi_sbm_sector"].mean(), 3),
            "nmi_sbm_sector_other": round(df.loc[~has_crash, "nmi_sbm_sector"].mean(), 3),
            "B": round(df.loc[has_crash, "B"].mean(), 1),
            "B_other": round(df.loc[~has_crash, "B"].mean(), 1),
            "frac_assortative": round(df.loc[has_crash, "frac_assortative"].mean(), 3),
            "frac_assortative_other": round(df.loc[~has_crash, "frac_assortative"].mean(), 3),
            "mean_raw_corr": round(df.loc[has_crash, "mean_raw_corr"].mean(), 3),
            "mean_raw_corr_other": round(df.loc[~has_crash, "mean_raw_corr"].mean(), 3),
            "nmi_month_sbm_entering_crash": round(df.loc[entering, "nmi_month_sbm"].mean(), 3),
            "nmi_month_sbm_all": round(df["nmi_month_sbm"].mean(), 3),
        },
        "corr_nmi_sector_vs_mean_corr": round(
            df["nmi_sbm_sector"].corr(df["mean_raw_corr"]), 3),
    }
    (ROOT / "results" / "stability_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    plot(df, has_crash)


def plot(df, has_crash):
    fig, axes = plt.subplots(4, 1, figsize=(9, 9.5), sharex=True)
    shade_start = df.index[has_crash.values][0]
    shade_end = df.index[has_crash.values][-1]
    for ax in axes:
        ax.axvspan(shade_start, shade_end, color=STRESS_SHADE, lw=0, zorder=0)

    ax = axes[0]
    ax.plot(df.index, df["nmi_sbm_sector"], color=PALETTE[0], label="SBM (nested, weighted)")
    ax.plot(df.index, df["nmi_hier_sector"], color=PALETTE[1], label="Hierarchical (same K)")
    ax.plot(df.index, df["nmi_louvain_sector"], color=PALETTE[2], label="Louvain (kNN graph)")
    ax.set_ylabel("NMI with GICS sector")
    ax.set_title("a. Agreement with GICS sectors (trailing 1-year window, refit monthly)",
                 loc="left")
    ax.legend(ncol=3, loc="lower left")

    ax = axes[1]
    ax.plot(df.index, df["nmi_year_sbm"], color=PALETTE[0], label="SBM")
    ax.plot(df.index, df["nmi_year_hier_B"], color=PALETTE[1], label="Hierarchical")
    ax.plot(df.index, df["nmi_year_louvain_knn"], color=PALETTE[2], label="Louvain")
    ax.set_ylabel("NMI with partition\none year earlier")
    ax.set_title("b. Stability across non-overlapping windows", loc="left")
    ax.legend(ncol=3, loc="lower left")

    ax = axes[2]
    ax.plot(df.index, df["B"], color=PALETTE[0])
    ax.set_ylabel("SBM blocks (level 0)")
    ax.set_title("c. Number of inferred blocks", loc="left")

    ax = axes[3]
    ax.plot(df.index, df["mean_raw_corr"], color=GREY)
    ax.set_ylabel("mean pairwise\nraw correlation")
    ax.set_title("d. Market stress proxy (shaded: windows containing Feb-Apr 2020)", loc="left")
    fig.tight_layout()
    savefig(fig, ROOT / "figures" / "rolling_stability")
    plt.close(fig)


if __name__ == "__main__":
    main()
