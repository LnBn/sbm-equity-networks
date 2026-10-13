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

WINDOWS = [("calm_2023_24", "Calm 2023-24\n(2 yr)"), ("calm_2023", "Calm 2023\n(1 yr)"),
           ("calm_2024", "Calm 2024\n(1 yr)"), ("stress_2020", "Stress 2020\n(1 yr)")]
PANELS = [("nmi_sector", "NMI with GICS sector"), ("nmi_subind", "NMI with GICS sub-industry"),
          ("pairs_CP", "Share of block pairs core-periphery\n(Liu et al. 2023)"),
          ("B", "Number of blocks B")]
COLOURS = {"calm": "#2a78d6", "stress": "#c4384b"}


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
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.2))
    for ax, (col, title) in zip(axes, PANELS):
        for i, (w, lab) in enumerate(present):
            x = d[d.window == w]
            c = COLOURS["stress" if w.startswith("stress") else "calm"]
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(x)), x[col], s=14, color=c, alpha=0.6, lw=0)
            best = x.loc[x.dl.idxmin()]
            ax.scatter(i, best[col], s=60, facecolor="none", edgecolor="k", lw=1.2, zorder=3)
        ax.set_xticks(range(len(present)), [lab for _, lab in present], fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].scatter([], [], s=60, facecolor="none", edgecolor="k", label="best basin (lowest DL)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Structure across 30 annealed SBM basins per window (each dot = one basin)", fontsize=10)
    fig.tight_layout()
    savefig(fig, ROOT / "figures" / "posterior_window_comparison", dpi=150)


if __name__ == "__main__":
    main()
