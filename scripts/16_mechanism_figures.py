"""Figures for the mechanism: which sectors fragment, which ones merge, and where stocks go.

1. Sector cohesion by year: for each GICS sector, the share of same-sector stock pairs that
   share a block, averaged over all 30 annealed basins of each calendar year. A row for all
   pairs gives the baseline expected from block sizes alone.
2. Cross-sector co-assignment, change vs 2019: for sectors S, T the share of (S, T) stock
   pairs that share a block (basin-averaged), shown as 2020 - 2019 and 2022 - 2019.
3. Alluvial diagram, 2019 -> 2020 -> 2021 -> 2022, for the stocks of the two largest 2020
   cross-sector blocks (reopening/travel and cyclical/energy), using each year's best basin.
   Each band is one stock, coloured by GICS sector; boxes are blocks, labelled by the
   block's most common sub-industry in the full partition.

4. Mixing summary per year, every basin: share of same-sector pairs sharing a block, share of
   cross-sector pairs sharing a block, and the average share of a stock's block-mates that
   come from other sectors.

Usage:  python scripts/16_mechanism_figures.py
Outputs: figures/mechanism_sector_cohesion, figures/mechanism_cross_sector_change,
         figures/mechanism_alluvial, figures/mechanism_mixing_summary (PNG + PDF);
         results/posterior/sector_cohesion.csv, results/posterior/sector_mixing_summary.csv
"""
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import DATA  # noqa: E402
from src.plotting import SECTOR_COLOURS, plt, savefig  # noqa: E402

YEARS = [("year_2019", "2019"), ("stress_2020", "2020"), ("year_2021", "2021"),
         ("year_2022", "2022"), ("calm_2023", "2023"), ("calm_2024", "2024")]


def load():
    returns = pd.read_parquet(DATA / "log_returns.parquet")
    tickers = list(returns.drop(columns="SPY").columns)
    meta = pd.read_csv(DATA / "universe.csv", index_col=0).loc[tickers]
    parts = {}
    for w, _ in YEARS:
        starts = pickle.load(gzip.open(ROOT / "results/posterior" / w / "basins.pkl.gz", "rb"))["starts"]
        starts = sorted(starts, key=lambda s: s["dl"])
        parts[w] = [np.unique(s["bs"][0], return_inverse=True)[1] for s in starts]   # best first
    return tickers, meta, parts


def coassignment(b, sec_idx, n_sec):
    """n_sec x n_sec matrix: share of (S, T) stock pairs (i != j) that share a block."""
    same = (b[:, None] == b[None, :]).astype(float)
    np.fill_diagonal(same, np.nan)
    M = np.zeros((n_sec, n_sec))
    for s in range(n_sec):
        for t in range(n_sec):
            M[s, t] = np.nanmean(same[np.ix_(sec_idx == s, sec_idx == t)])
    return M


def fig_cohesion_and_change(meta, parts):
    sectors = sorted(meta["sector"].unique())
    sec_idx = np.array([sectors.index(x) for x in meta["sector"]])
    mean_M = {}
    for w, _ in YEARS:
        mean_M[w] = np.mean([coassignment(b, sec_idx, len(sectors)) for b in parts[w]], axis=0)
    # --- Figure 1: cohesion (diagonal) -------------------------------------------------
    coh = pd.DataFrame({lab: np.diag(mean_M[w]) for w, lab in YEARS}, index=sectors)
    baseline = {lab: np.mean([np.mean([(b[:, None] == b[None, :])[np.triu_indices(len(b), 1)].mean()])
                              for b in parts[w]]) for w, lab in YEARS}
    coh.loc["(all pairs: size baseline)"] = pd.Series(baseline)
    coh["n stocks"] = list(pd.Series(sec_idx).value_counts().reindex(range(len(sectors))).values) + [len(sec_idx)]
    coh.to_csv(ROOT / "results/posterior/sector_cohesion.csv", float_format="%.3f")
    order = coh.drop(index="(all pairs: size baseline)").sort_values("2019", ascending=False).index.tolist()
    order.append("(all pairs: size baseline)")
    C = coh.loc[order, [lab for _, lab in YEARS]]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(C.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            v = C.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.6 else "#0b0b0b")
    ax.set_xticks(range(C.shape[1]), C.columns)
    ax.set_yticks(range(C.shape[0]), [f"{s} ({int(coh.loc[s, 'n stocks'])})" for s in order], fontsize=8)
    ax.axhline(len(order) - 1.5, color="k", lw=0.8)
    for j, lab in enumerate(C.columns):
        if lab in ("2020", "2022"):
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(order), fill=False,
                                   edgecolor="#c4384b" if lab == "2020" else "#eda100", lw=1.6))
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="share of same-sector pairs in the same group")
    fig.tight_layout()
    savefig(fig, ROOT / "figures" / "mechanism_sector_cohesion", dpi=150)
    # --- Figure 2: cross-sector change vs 2019 ------------------------------------------
    short = [s.replace("Information Technology", "IT").replace("Communication Services", "Comm. Services")
             .replace("Consumer Discretionary", "Cons. Discretionary").replace("Consumer Staples", "Cons. Staples")
             for s in sectors]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    lim = 0.15          # colour scale capped so small cross-sector changes stay visible; values are printed
    for ax, (w, title) in zip(axes, [("stress_2020", "2020 (COVID) minus 2019"),
                                     ("year_2022", "2022 (rates) minus 2019")]):
        D = mean_M[w] - mean_M["year_2019"]
        im = ax.imshow(D, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(len(sectors)):
            for j in range(len(sectors)):
                if abs(D[i, j]) >= 0.02:
                    ax.text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(D[i, j]) > 0.8 * lim else "#0b0b0b")
        ax.set_xticks(range(len(sectors)), short, rotation=60, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(sectors)), short, fontsize=7.5)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.025, extend="both",
                 label="change in share of stock pairs sharing a group")
    savefig(fig, ROOT / "figures" / "mechanism_cross_sector_change", dpi=150)
    return coh


def bezier_band(x0, y0, x1, y1, h, colour, ax, alpha=0.55):
    """Band of height h from (x0, y0) to (x1, y1) (y = bottom edge), cubic-Bezier edges."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 + h), (xm, y1 + h), (xm, y0 + h),
             (x0, y0 + h), (x0, y0)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=colour, edgecolor="none", alpha=alpha))


def fig_alluvial(tickers, meta, parts):
    best = {w: parts[w][0] for w, _ in YEARS}
    tk = np.array(tickers)
    b20 = best["stress_2020"]
    # the two largest cross-sector 2020 blocks with high internal correlation, found by membership
    anchors = {"reopening/travel": "CCL", "cyclical/energy": "XOM"}
    chosen = sorted(set(tk[np.isin(b20, [b20[tickers.index(a)] for a in anchors.values()])]))
    idx = [tickers.index(t) for t in chosen]
    years = [("year_2019", "2019"), ("stress_2020", "2020"), ("year_2021", "2021"), ("year_2022", "2022")]
    gap, unit = 0.8, 1.0
    pos = {}             # (year, ticker) -> bottom y of the stock's band slot
    boxes = {}           # year -> list of (block, y0, height, label)
    prev_order = {t: k for k, t in enumerate(chosen)}
    for w, lab in years:
        b = best[w]
        groups = {}
        for t, i in zip(chosen, idx):
            groups.setdefault(b[i], []).append(t)
        # order blocks by the mean previous position of their stocks (fewer crossings)
        blocks = sorted(groups, key=lambda k: np.mean([prev_order[t] for t in groups[k]]))
        y, cur = 0.0, {}
        boxes[w] = []
        for blk in blocks:
            members = sorted(groups[blk], key=lambda t: prev_order[t])
            sub = meta["sub_industry"].values[b == blk]
            top = pd.Series(sub).value_counts().index[0]
            boxes[w].append((blk, y, len(members) * unit,
                             f"group of {(b == blk).sum()}; most common: {top[:24]}"))
            for t in members:
                pos[(w, t)] = y
                cur[t] = y
                y += unit
            y += gap
        prev_order = {t: cur[t] for t in chosen}
    fig, ax = plt.subplots(figsize=(14, 9))
    xs = {w: 3.2 * k for k, (w, _) in enumerate(years)}
    width = 0.25
    for (wa, _), (wb, _) in zip(years[:-1], years[1:]):
        for t in chosen:
            bezier_band(xs[wa] + width, pos[(wa, t)], xs[wb], pos[(wb, t)], unit * 0.92,
                        SECTOR_COLOURS[meta.loc[t, "sector"]], ax)
    for w, lab in years:
        for blk, y0, h, text in boxes[w]:
            ax.add_patch(Rectangle((xs[w], y0), width, h, facecolor="#2b2b2b", edgecolor="none"))
            ax.text(xs[w] + width + 0.05, y0 + h / 2, text, fontsize=6.5, va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5))
        ax.text(xs[w] + width / 2, -1.6, lab, ha="center", fontsize=10, fontweight="bold")
    # stock labels on the left (2019) column
    for t in chosen:
        ax.text(xs["year_2019"] - 0.08, pos[("year_2019", t)] + unit / 2, t, ha="right", va="center",
                fontsize=6.5, color=SECTOR_COLOURS[meta.loc[t, "sector"]])
    handles = [Rectangle((0, 0), 1, 1, color=c) for s, c in SECTOR_COLOURS.items()
               if s in set(meta.loc[chosen, "sector"])]
    labels = [s for s in SECTOR_COLOURS if s in set(meta.loc[chosen, "sector"])]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(labels),
              fontsize=7.5, frameon=False)
    ax.set_xlim(-1.2, xs["year_2022"] + 2.6)
    ax.set_ylim(max(pos.values()) + 2, -2.5)
    ax.axis("off")
    savefig(fig, ROOT / "figures" / "mechanism_alluvial", dpi=150)
    return chosen


def fig_mixing_summary(meta, parts):
    sec = meta["sector"].to_numpy(dtype=object)
    same_sec = sec[:, None] == sec[None, :]
    iu = np.triu_indices(len(sec), 1)
    rows = []
    for w, lab in YEARS:
        for k, b in enumerate(parts[w]):
            same = b[:, None] == b[None, :]
            np.fill_diagonal(same, False)
            n = same.sum(1)
            other = (same & ~same_sec).sum(1)
            rows.append({"window": w, "year": lab, "basin_rank": k,
                         "within_sector_coassigned": same[iu][same_sec[iu]].mean(),
                         "cross_sector_coassigned": same[iu][~same_sec[iu]].mean(),
                         "blockmates_other_sector": np.mean(other[n > 0] / n[n > 0])})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/posterior/sector_mixing_summary.csv", index=False, float_format="%.4f")
    panels = [("within_sector_coassigned", "Same-sector pairs\nsharing a group"),
              ("cross_sector_coassigned", "Cross-sector pairs\nsharing a group"),
              ("blockmates_other_sector", "Share of a stock's group-mates\nfrom other sectors")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (col, title) in zip(axes, panels):
        for i, (w, lab) in enumerate(YEARS):
            x = d[d.window == w]
            c = "#c4384b" if lab == "2020" else "#eda100" if lab == "2022" else "#2a78d6"
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(x)), x[col], s=12, color=c, alpha=0.6, lw=0)
            ax.scatter(i, x[x.basin_rank == 0][col], s=55, facecolor="none", edgecolor="k", lw=1.1, zorder=3)
        ax.set_xticks(range(len(YEARS)), [lab for _, lab in YEARS])
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    savefig(fig, ROOT / "figures" / "mechanism_mixing_summary", dpi=150)
    return d.groupby("year", sort=False)[[c for c, _ in panels]].agg(["mean", "min", "max"])


def main():
    tickers, meta, parts = load()
    coh = fig_cohesion_and_change(meta, parts)
    print(coh.round(2).to_string())
    chosen = fig_alluvial(tickers, meta, parts)
    print(fig_mixing_summary(meta, parts).round(4).to_string())
    print(f"\nalluvial: {len(chosen)} stocks: {' '.join(chosen)}")


if __name__ == "__main__":
    main()
