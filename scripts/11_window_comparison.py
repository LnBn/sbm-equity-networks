"""Figure: structure of every basin, window by window (output of 10_basin_compare.py).

Each dot is one annealed start (basin); the best basin (lowest DL) is outlined.
If the stress window separates from the calm windows on every panel, and the 1-year
calm controls sit with the 2-year calm window, the 2020 differences are neither an
optimiser artefact nor a window-length effect.

Usage:  python scripts/11_window_comparison.py
Output: figures/posterior_window_comparison.png, results/posterior/window_comparison.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plotting import plt, savefig  # noqa: E402

# Chronological; the 2-year window goes last because its length differs (shown hollow).
WINDOWS = [("year_2019", "2019"), ("stress_2020", "2020\n(COVID)"), ("year_2021", "2021"),
           ("year_2022", "2022\n(rates)"), ("calm_2023", "2023"), ("calm_2024", "2024"),
           ("calm_2023_24", "2023\u201324\n(2 years)")]
PANELS = [("nmi_sector", "Agreement with GICS sectors (NMI)"),
          ("nmi_subind", "Agreement with GICS sub-industries (NMI)"),
          ("pairs_CP", "Share of pairs of groups that are core\u2013periphery"),
          ("B", "Number of groups")]
COLOURS = {"calm": "#2a78d6", "stress": "#c4384b", "rate": "#eda100"}


def colour(w):
    return COLOURS["stress" if w == "stress_2020" else "rate" if w == "year_2022" else "calm"]


def main():
    frames = []
    for w, _ in WINDOWS:
        f = ROOT / "results" / "posterior" / w / "basins_summary.csv"
        if f.exists():
            frames.append(pd.read_csv(f).assign(window=w))
    d = pd.concat(frames)
    present = [(w, lab) for w, lab in WINDOWS if w in set(d.window)]
    summary = d.groupby("window")[[c for c, _ in PANELS]].agg(["min", "median", "max"]).round(3)
    summary.to_csv(ROOT / "results" / "posterior" / "window_comparison.csv")
    print(summary.loc[[w for w, _ in present]].to_string())

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, PANELS):
        for i, (w, lab) in enumerate(present):
            x = d[d.window == w]
            c = colour(w)
            two_year = w == "calm_2023_24"
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(x)), x[col], s=14, alpha=0.6,
                       facecolor="none" if two_year else c, edgecolor=c, lw=0.8 if two_year else 0)
            best = x.loc[x.dl.idxmin()]
            ax.scatter(i, best[col], s=60, facecolor="none", edgecolor="k", lw=1.2, zorder=3)
        ax.set_xticks(range(len(present)), [lab for _, lab in present], fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[-1].scatter([], [], s=60, facecolor="none", edgecolor="k", label="best fit")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    savefig(fig, ROOT / "figures" / "posterior_window_comparison", dpi=150)


if __name__ == "__main__":
    main()
