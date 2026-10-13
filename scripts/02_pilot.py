"""Pilot: full SBM pipeline on two windows (calm 2023-24, stress 2020).

For each window:
  A. model selection by description length (DL) across SBM variants,
     network constructions and market-mode treatments;
  B. comparison of the best SBM partition with GICS, Louvain and hierarchical
     clustering (NMI / ARI, and DL of each partition under the same model);
  C. the inter-block mixing pattern (assortative or not).

Usage:  python scripts/02_pilot.py [--refit]   (fits take ~5 minutes on 16 cores;
        without --refit, cached fits in results/pilot_fits.json are reused)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # we parallelise across fits instead

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
from src.network import dense_graph, knn_graph  # noqa: E402
from src.plotting import plt  # noqa: E402
from src.preprocess import preprocess  # noqa: E402
from src.sbm import fit_sbm, hierarchy_sizes, partition  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
WINDOWS = {"calm_2023_24": ("2023-01-01", "2024-12-31"),
           "stress_2020": ("2020-01-01", "2020-12-31")}
K_NN = 10

# (name, market treatment, graph, SBM keyword arguments)
CONFIGS = [
    ("dense-W-NDC",        "raw", "dense", dict(weighted=True, deg_corr=False)),
    ("dense-W-NDC-nested", "raw", "dense", dict(weighted=True, deg_corr=False, nested=True)),
    ("dense-W-NDC",        "pc1", "dense", dict(weighted=True, deg_corr=False)),
    ("dense-W-NDC-nested", "pc1", "dense", dict(weighted=True, deg_corr=False, nested=True)),
    ("dense-W-NDC",        "spy", "dense", dict(weighted=True, deg_corr=False)),
    ("dense-W-DC",         "spy", "dense", dict(weighted=True, deg_corr=True)),
    ("dense-W-NDC-nested", "spy", "dense", dict(weighted=True, deg_corr=False, nested=True)),
    ("knn-PP-DC",          "spy", "knn",   dict(weighted=False, deg_corr=True, assortative=True)),
    ("knn-NDC",            "spy", "knn",   dict(weighted=False, deg_corr=False)),
    ("knn-DC",             "spy", "knn",   dict(weighted=False, deg_corr=True)),
    ("knn-DC-nested",      "spy", "knn",   dict(weighted=False, deg_corr=True, nested=True)),
    ("knn-W-DC",           "spy", "knn",   dict(weighted=True, deg_corr=True)),
    ("knn-W-DC-nested",    "spy", "knn",   dict(weighted=True, deg_corr=True, nested=True)),
]


def load_window(start, end):
    returns = pd.read_parquet(DATA / "log_returns.parquet").loc[start:end]
    meta = pd.read_csv(DATA / "universe.csv", index_col=0)
    stocks = returns.drop(columns="SPY")
    return stocks, returns["SPY"], meta.loc[stocks.columns]


def build_graph(corr, kind):
    return dense_graph(corr) if kind == "dense" else knn_graph(corr, K_NN)


def run_config(job):
    """Worker: fit one SBM variant on one window; return plain data only."""
    window, name, mode, kind, kwargs = job
    stocks, market, _ = load_window(*WINDOWS[window])
    corr = preprocess(stocks, market, mode).corr().values
    state, dl = fit_sbm(build_graph(corr, kind), **kwargs)
    out = {"window": window, "model": name, "market": mode, "graph": kind,
           "dl_nats": dl, "labels": partition(state).tolist()}
    if kwargs.get("nested"):
        out["levels"] = hierarchy_sizes(state)
        out["labels_level1"] = np.unique(
            state.project_level(1).get_blocks().a, return_inverse=True)[1].tolist()
    return out


def plot_window(window, corr, labels, sectors, M, path):
    """Residual correlation matrix ordered by SBM block, with GICS colour strip."""
    order = np.lexsort((sectors, labels))
    sector_names = sorted(set(sectors))
    sector_idx = np.array([sector_names.index(s) for s in sectors])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    lim = np.nanpercentile(np.abs(corr[~np.eye(len(corr), dtype=bool)]), 99)
    im = ax.imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    bounds = np.flatnonzero(np.diff(labels[order])) + 0.5
    for b in bounds:
        ax.axhline(b, color="k", lw=0.4)
        ax.axvline(b, color="k", lw=0.4)
    strip = ax.inset_axes([-0.06, 0, 0.03, 1])
    strip.imshow(sector_idx[order][:, None], aspect="auto", cmap="tab20",
                 vmin=0, vmax=19, interpolation="nearest")
    strip.set_axis_off()
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("Market-residual correlations, ordered by SBM block\n"
                 "(left strip = GICS sector)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    # Contingency table: SBM blocks x GICS sectors
    table = pd.crosstab(labels, sectors)
    ax = axes[1]
    ax.imshow(table.values, cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index, fontsize=6)
    ax.set_ylabel("SBM block"); ax.grid(False)
    ax.set_title("Stocks per (SBM block, GICS sector)", fontsize=10)
    fig.suptitle(window, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_block_matrix(M_resid, M_raw, resid_names, path):
    """Block-mean correlation matrices: residual SBM (rows labelled by dominant
    sector) and raw-correlation SBM (blocks sorted by mean correlation)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    panels = [(axes[0], M_resid, "SBM on market-residual correlations\n"
               "(blocks grouped by dominant GICS sector)"),
              (axes[1], M_raw, "SBM on raw correlations\n"
               "(blocks sorted by mean correlation with others)")]
    for ax, M, title in panels:
        lim = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(f"Block-mean correlation: {title}", fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045)
    axes[0].set_yticks(range(len(resid_names)))
    axes[0].set_yticklabels(resid_names, fontsize=6)
    axes[0].set_xticks([])
    axes[1].set_xlabel("block"); axes[1].set_ylabel("block")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    cache = RESULTS / "pilot_fits.json"
    if cache.exists() and "--refit" not in sys.argv:
        fits = json.loads(cache.read_text())
    else:
        jobs = [(w, *cfg) for w in WINDOWS for cfg in CONFIGS]
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() - 2)) as pool:
            fits = list(pool.map(run_config, jobs))
        cache.write_text(json.dumps(fits))

    rows, summary = [], {}
    for window in WINDOWS:
        stocks, market, meta = load_window(*WINDOWS[window])
        sector = cmp.labels_from(meta["sector"])
        sub_ind = cmp.labels_from(meta["sub_industry"])
        for f in (f for f in fits if f["window"] == window):
            lab = np.array(f["labels"])
            rows.append({
                "window": window, "market": f["market"], "model": f["model"],
                "DL_nats": round(f["dl_nats"], 1), "B": lab.max() + 1,
                "levels": f.get("levels", ""),
                "NMI_sector": round(cmp.nmi(lab, sector), 3),
                "NMI_subind": round(cmp.nmi(lab, sub_ind), 3),
            })

        # --- B. compare the chosen SBM with baselines (SPY residuals) -------
        resid = preprocess(stocks, market, "spy")
        corr = resid.corr().values
        best = min((f for f in fits if f["window"] == window and f["market"] == "spy"
                    and f["graph"] == "dense"), key=lambda f: f["dl_nats"])
        sbm = np.array(best["labels"])
        B = sbm.max() + 1
        nested = next(f for f in fits if f["window"] == window and f["market"] == "spy"
                      and f["model"] == "dense-W-NDC-nested")
        partitions = {
            f"SBM ({best['model']}, B={B})": sbm,
            "SBM nested level 1": np.array(nested["labels_level1"]),
            "GICS sector": sector,
            "GICS sub-industry": sub_ind,
            "Louvain (dense, w>0)": cmp.louvain_dense(corr),
            "Louvain (kNN)": cmp.louvain_knn(corr, K_NN),
            "Hierarchical K=11": cmp.hierarchical(corr, 11),
            f"Hierarchical K={B}": cmp.hierarchical(corr, B),
        }
        g = dense_graph(corr)
        comp = []
        for name, lab in partitions.items():
            M = cmp.block_mean_matrix(corr, lab)
            mix = cmp.assortativity_summary(M, np.bincount(lab))
            comp.append({
                "partition": name, "K": lab.max() + 1,
                "NMI_vs_GICS": round(cmp.nmi(lab, sector), 3),
                "ARI_vs_GICS": round(cmp.ari(lab, sector), 3),
                "NMI_vs_SBM": round(cmp.nmi(lab, sbm), 3),
                "DL_minus_SBM_nats": round(cmp.dl_of_partition(g, lab) -
                                           cmp.dl_of_partition(g, sbm), 1),
                "frac_assortative": round(mix["frac_assortative_blocks"], 2),
                "within_corr": round(mix["mean_within"], 3),
                "between_corr": round(mix["mean_between"], 3),
            })
        comp = pd.DataFrame(comp)
        comp.to_csv(RESULTS / f"pilot_partitions_{window}.csv", index=False)
        print(f"\n=== {window}: partition comparison ===\n{comp.to_string(index=False)}")

        # --- C. mixing pattern of the residual SBM and of the raw-data SBM --
        M_resid = cmp.block_mean_matrix(corr, sbm)
        raw_fit = next(f for f in fits if f["window"] == window and f["market"] == "raw"
                       and f["model"] == "dense-W-NDC")
        raw_lab = np.array(raw_fit["labels"])
        raw_corr = stocks.corr().values
        M_raw = cmp.block_mean_matrix(raw_corr, raw_lab)
        summary[window] = {
            "best_model": best["model"], "B": int(B),
            "resid_mixing": cmp.assortativity_summary(M_resid, np.bincount(sbm)),
            "raw_mixing": cmp.assortativity_summary(M_raw, np.bincount(raw_lab)),
            "raw_B": int(raw_lab.max() + 1),
            "raw_block_mean_corr_sorted": np.round(np.sort(np.diag(M_raw)), 3).tolist(),
        }
        # Blocks in the raw-correlation SBM: does block 'coreness' (mean corr
        # with everyone else) explain the between-block correlations?
        # Rank-1 check: corr(M_rs, c_r * c_s) over off-diagonal pairs.
        c = np.nanmean(np.where(np.eye(len(M_raw), dtype=bool), np.nan, M_raw), axis=1)
        off = ~np.eye(len(M_raw), dtype=bool)
        summary[window]["raw_rank1_fit_corr"] = float(
            np.corrcoef(M_raw[off], np.outer(c, c)[off])[0, 1])

        plot_window(window, corr, sbm, meta["sector"].values, M_resid,
                    FIGURES / f"pilot_{window}_blocks.png")
        # order blocks by their dominant sector for a readable block matrix
        dom = pd.Series(meta["sector"].values).groupby(sbm).agg(lambda s: s.mode()[0])
        order = np.argsort(dom.values, kind="stable")
        sizes = np.bincount(sbm)
        names = [f"{dom.values[b][:18]} (n={sizes[b]})" for b in order]
        raw_order = np.argsort(-c)
        plot_block_matrix(M_resid[np.ix_(order, order)], M_raw[np.ix_(raw_order, raw_order)],
                          names, FIGURES / f"pilot_{window}_block_matrix.png")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "pilot_model_selection.csv", index=False)
    print("\n=== model selection (compare DL only within the same market/graph) ===")
    print(table.to_string(index=False))
    (RESULTS / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
