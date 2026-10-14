"""Stock-level residual-correlation heatmaps ordered by each calendar year's best basin.

For each window, the 250 x 250 matrix of market-residual correlations is reordered so that
stocks in the same best-basin block are adjacent. Blocks are grouped by their level-1 group
(the next level of the nested SBM); within a group, blocks run from the highest internal
correlation omega_rr to the lowest; within a block, stocks are sorted by GICS sector. A strip on
the left and along the bottom gives each stock's sector (rows and columns are in the same order),
so a sector-pure block shows as one colour run, and a block that mixes sectors shows as several. Sector names are printed next to runs of at least
MIN_RUN stocks, so identity does not rest on colour alone (11 sectors are too many for a
colour-blind-safe palette).

Panel titles give the best basin's number of blocks, NMI with GICS sectors
(best_diagnostics.json) and the share of a stock's block-mates from other sectors
(sector_mixing_summary.csv, basin_rank 0).

Usage:  python scripts/17_heatmaps.py
Outputs: figures/posterior_heatmaps_years (2019-2024, 2 x 3), figures/posterior_heatmaps_shocks
         (2019, 2020, 2022), PNG + PDF
"""
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import compare as cmp  # noqa: E402
from src.data import DATA  # noqa: E402
from src.plotting import SECTOR_COLOURS, plt, savefig  # noqa: E402
from src.preprocess import spy_residuals  # noqa: E402

YEARS = [("year_2019", "2019"), ("stress_2020", "2020"), ("year_2021", "2021"),
         ("year_2022", "2022"), ("calm_2023", "2023"), ("calm_2024", "2024")]
WINDOWS = {"year_2019": ("2019-01-01", "2019-12-31"), "stress_2020": ("2020-01-01", "2020-12-31"),
           "year_2021": ("2021-01-01", "2021-12-31"), "year_2022": ("2022-01-01", "2022-12-31"),
           "calm_2023": ("2023-01-01", "2023-12-31"), "calm_2024": ("2024-01-01", "2024-12-31")}
SECTORS = list(SECTOR_COLOURS)
SECTOR_ABBR = {"Communication Services": "Comm", "Consumer Discretionary": "Disc",
               "Consumer Staples": "Stap", "Energy": "Enrg", "Financials": "Fin",
               "Health Care": "HC", "Industrials": "Ind", "Information Technology": "IT",
               "Materials": "Mat", "Real Estate": "RE", "Utilities": "Util"}
LIM = 0.6        # colour scale clipped at +-LIM
MIN_RUN = 8      # label sector runs at least this long


def load_window(window):
    """Residual correlations, sector of each stock, and the best basin's levels 0 and 1."""
    returns = pd.read_parquet(DATA / "log_returns.parquet").loc[slice(*WINDOWS[window])]
    stocks = returns.drop(columns="SPY")
    corr = spy_residuals(stocks, returns["SPY"]).corr().values
    meta = pd.read_csv(DATA / "universe.csv", index_col=0).loc[stocks.columns]
    starts = pickle.load(gzip.open(ROOT / "results/posterior" / window / "basins.pkl.gz", "rb"))["starts"]
    best = min(starts, key=lambda s: s["dl"])
    b0 = np.unique(best["bs"][0], return_inverse=True)[1]
    b1 = np.unique(np.asarray(best["bs"][1])[best["bs"][0]], return_inverse=True)[1]
    sec = np.array([SECTORS.index(s) for s in meta["sector"]])
    return corr, sec, b0, b1, meta


def stock_order(corr, sec, b0, b1):
    """Order: level-1 group (tightest first), then block by omega_rr, then sector."""
    omega = cmp.block_mean_matrix(corr, b0)
    w_rr = np.nan_to_num(np.diag(omega), nan=-1.0)             # singletons last
    group_tight = pd.Series(w_rr[b0]).groupby(b1).mean()       # mean over the group's stocks
    group_rank = group_tight.rank(ascending=False, method="first").to_numpy()[b1]
    block_rank = pd.Series(-w_rr).rank(method="first").to_numpy()[b0]
    order = np.lexsort((sec, block_rank, group_rank))          # last key is the primary one
    return order, b0[order], b1[order]


def runs(x):
    """(start, length, value) of each run of equal consecutive values."""
    edges = np.flatnonzero(np.diff(x)) + 1
    starts = np.r_[0, edges]
    lengths = np.diff(np.r_[starts, len(x)])
    return [(s, n, x[s]) for s, n in zip(starts, lengths)]


def panel(ax, corr, sec, b0, b1, title, subtitle):
    order, blk, grp = stock_order(corr, sec, b0, b1)
    im = ax.imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-LIM, vmax=LIM,
                   interpolation="nearest", rasterized=True)
    for pos in np.flatnonzero(np.diff(blk)) + 0.5:
        ax.axhline(pos, color="#9a9994", lw=0.15)
        ax.axvline(pos, color="#9a9994", lw=0.15)
    for pos in np.flatnonzero(np.diff(grp)) + 0.5:
        ax.axhline(pos, color="#2b2b2a", lw=0.5)
        ax.axvline(pos, color="#2b2b2a", lw=0.5)
    # Rows and columns share one order, so the same sector strip runs down the left and along the bottom.
    cmap = ListedColormap([SECTOR_COLOURS[s] for s in SECTORS])
    kw = dict(aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(SECTORS) - 0.5, interpolation="nearest")
    left_strip = ax.inset_axes([-0.045, 0, 0.035, 1])
    left_strip.imshow(sec[order][:, None], **kw)
    bottom_strip = ax.inset_axes([0, -0.045, 1, 0.035])
    bottom_strip.imshow(sec[order][None, :], **kw)
    for strip in (left_strip, bottom_strip):
        strip.set_axis_off()
    for start, n, s in runs(sec[order]):
        if n >= MIN_RUN:
            mid = (start + n / 2) / len(order)
            label = SECTOR_ABBR[SECTORS[s]]
            ax.text(-0.055, 1 - mid, label, transform=ax.transAxes, ha="right", va="center",
                    fontsize=4.5, color="#52514e")
            ax.text(mid, -0.055, label, transform=ax.transAxes, ha="center", va="top", rotation=90,
                    fontsize=4.5, color="#52514e")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.01, 1.11, title, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
    ax.text(-0.01, 1.012, subtitle, transform=ax.transAxes, fontsize=5.8, color="#52514e", va="bottom",
            linespacing=1.15)
    return im


def subtitle(window, b0):
    diag = json.load(open(ROOT / "results/posterior" / window / "best_diagnostics.json"))
    mix = pd.read_csv(ROOT / "results/posterior/sector_mixing_summary.csv")
    other = mix.query("window == @window and basin_rank == 0")["blockmates_other_sector"].item()
    assert diag["B"] == b0.max() + 1
    return (f"{diag['B']} groups, agreement with sectors {diag['nmi_sector']:.2f}\n"
            f"group-mates from other sectors {other:.0%}")


def figure(years, ncols, path, width=6.45):
    """Panels laid out in inches: a title band above each row, a legend band at the bottom."""
    nrows = int(np.ceil(len(years) / ncols))
    left, right, gap, title_h, row_gap, bottom_h = 0.42, 0.05, 0.42, 0.42, 0.75, 0.85
    pw = (width - left - right - (ncols - 1) * gap) / ncols
    height = title_h + nrows * pw + (nrows - 1) * row_gap + bottom_h
    fig = plt.figure(figsize=(width, height), facecolor="white")
    for k, (window, label) in enumerate(years):
        i, j = divmod(k, ncols)
        x0 = left + j * (pw + gap)
        y0 = height - title_h - i * (pw + row_gap) - pw
        ax = fig.add_axes([x0 / width, y0 / height, pw / width, pw / height])
        corr, sec, b0, b1, _ = load_window(window)
        im = panel(ax, corr, sec, b0, b1, label, subtitle(window, b0))
    cax = fig.add_axes([left / width, 0.2 / height, 1.6 / width, 0.07 / height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-LIM, -0.3, 0, 0.3, LIM])
    cb.set_label(f"market-residual correlation (clipped at \u00b1{LIM})", fontsize=6, labelpad=1)
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cb.outline.set_linewidth(0.3)
    from matplotlib.patches import Patch
    handles = [Patch(color=SECTOR_COLOURS[s], label=f"{s} ({SECTOR_ABBR[s]})") for s in SECTORS]
    fig.legend(handles=handles, loc="lower right", ncol=3, fontsize=5.5, handlelength=0.9,
               columnspacing=0.8, handletextpad=0.3, frameon=False,
               bbox_to_anchor=(1 - right / width, 0.02 / height))
    savefig(fig, path, dpi=600, bbox_inches=None)
    plt.close(fig)


def main():
    figure(YEARS, 3, ROOT / "figures" / "posterior_heatmaps_years")
    figure([YEARS[0], YEARS[1], YEARS[3]], 3, ROOT / "figures" / "posterior_heatmaps_shocks")
    for window, label in YEARS:
        corr, sec, b0, b1, _ = load_window(window)
        print(label, subtitle(window, b0).replace("\n", ", "), f"level-1 groups {b1.max() + 1}")


if __name__ == "__main__":
    main()
