"""Describe the best basin found by 08_basin_search.py for one window.

1. Local uncertainty: two short beta = 1 chains started from the best partition. How
   much does the partition move inside its basin, and which stocks are unsettled?
2. Liu et al. (2023) interaction types between the best partition's blocks (level 0) and
   between its level-1 groups, with each pair's type probability over the local samples.
3. A block table (size, omega_rr, dominant sector and sub-industry, members, number of
   assortative / core-periphery partners) and a table of the non-assortative pairs.

Usage:  python scripts/09_best_basin.py calm_2023_24
Outputs: results/posterior/<window>/best_*.csv|json,
         figures/posterior_<window>_best_types.{png,pdf}, figures/posterior_<window>_best_blocks.{png,pdf}
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gzip
import importlib.util
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import graph_tool.all as gt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import compare as cmp  # noqa: E402
from src import interactions as ix  # noqa: E402
from src import posterior as post  # noqa: E402
from src.data import DATA  # noqa: E402
from src.preprocess import spy_residuals  # noqa: E402

_spec = importlib.util.spec_from_file_location("s06", ROOT / "scripts" / "06_posterior_sample.py")
s06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s06)
_spec = importlib.util.spec_from_file_location("s02", ROOT / "scripts" / "02_pilot.py")
s02 = importlib.util.module_from_spec(_spec)          # reuse the pilot's block-ordered plot
_spec.loader.exec_module(s02)

N_LOCAL, N_LOCAL_CHAINS = 150, 2      # calls of multiflip_mcmc_sweep(beta=1, niter=10) per chain


def best_basin(window):
    starts = pickle.load(gzip.open(ROOT / "results/posterior" / window / "basins.pkl.gz", "rb"))["starts"]
    return min(starts, key=lambda s: s["dl"])


def local_chain(job):
    window, seed = job
    g, _ = s06.window_graph(window)
    best = best_basin(window)
    state = gt.NestedBlockState(g, bs=[np.array(b) for b in best["bs"]], base_state=gt.WeightedBlockState,
                                base_state_args=dict(deg_corr=False, rec=[g.ep["w"]], rec_types=["real-normal"]))
    gt.seed_rng(seed)
    dl, parts = [], []
    for _ in range(N_LOCAL):
        state.multiflip_mcmc_sweep(beta=1, niter=10)
        dl.append(state.entropy())
        parts.append(np.array(state.get_bs()[0], dtype=np.int32))
    return {"seed": seed, "dl": np.array(dl), "parts": parts}


def describe(labels, corr, meta, tickers):
    """Block table for one partition (labels 0..K-1)."""
    omega, T, fr = ix.classify(corr, labels)
    sizes = np.bincount(labels)
    sec = pd.crosstab(labels, meta["sector"].values)
    sub = pd.crosstab(labels, meta["sub_industry"].values)
    tab = pd.DataFrame({
        "block": range(len(sizes)), "size": sizes, "omega_rr": np.diag(omega),
        "top_sector": sec.idxmax(axis=1).values, "sector_share": (sec.max(axis=1) / sizes).values,
        "top_sub_industry": sub.idxmax(axis=1).values, "sub_industry_share": (sub.max(axis=1) / sizes).values,
        "n_assortative_partners": np.nansum(T == ix.A, axis=1),
        "n_core_periphery_partners": np.nansum(T == ix.CP, axis=1),
        "members": [" ".join(np.array(tickers)[labels == r]) for r in range(len(sizes))]})
    return omega, T, fr, tab


def main(window):
    res = ROOT / "results" / "posterior" / window
    g, tickers = s06.window_graph(window)
    returns = pd.read_parquet(DATA / "log_returns.parquet").loc[slice(*s06.WINDOWS[window])]
    corr = spy_residuals(returns.drop(columns="SPY"), returns["SPY"]).corr().values
    meta = pd.read_csv(DATA / "universe.csv", index_col=0).loc[tickers]
    best = best_basin(window)
    b0 = np.unique(best["bs"][0], return_inverse=True)[1]
    level1 = np.unique(np.asarray(best["bs"][1])[best["bs"][0]], return_inverse=True)[1]

    # ---- 1. Local uncertainty ---------------------------------------------------------
    with ProcessPoolExecutor(N_LOCAL_CHAINS) as pool:
        chains = list(pool.map(local_chain, [(window, 500 + i) for i in range(N_LOCAL_CHAINS)]))
    local = [np.unique(p, return_inverse=True)[1] for c in chains for p in c["parts"]]
    aligned = np.array([gt.align_partition_labels(p, b0) for p in local])
    stay = (aligned == b0[None, :]).mean(axis=0)            # share of samples in the best block
    local_nmi = [post.nmi(p, b0) for p in local[::5]]
    diag = {"best_seed": best["seed"], "best_dl": best["dl"], "B": int(b0.max() + 1),
            "level1_groups": int(level1.max() + 1),
            "local_dl_mean": float(np.mean([c["dl"].mean() for c in chains])),
            "local_dl_min": float(min(c["dl"].min() for c in chains)),
            "local_nmi_to_best_mean": float(np.mean(local_nmi)),
            "local_nmi_to_best_min": float(np.min(local_nmi)),
            "share_stocks_always_in_best_block": float((stay == 1).mean()),
            "share_stocks_below_0.9": float((stay < 0.9).mean()),
            "nmi_sector": cmp.nmi(b0, cmp.labels_from(meta["sector"])),
            "nmi_sub_industry": cmp.nmi(b0, cmp.labels_from(meta["sub_industry"]))}

    # ---- 2-3. Interaction types --------------------------------------------------------
    omega, T, fr, tab = describe(b0, corr, meta, tickers)
    counts, npair = ix.aligned_type_probabilities(corr, local[::3], b0)
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = counts / npair[..., None]
    order = [r for r in np.argsort(-np.diag(omega)) if tab["size"][r] >= 2]
    tab = tab.set_index("block").loc[order].reset_index()
    tab.to_csv(res / "best_blocks.csv", index=False, float_format="%.3f")
    pairs = []
    for i, r in enumerate(order):
        for s in order[i + 1:]:
            t = int(T[r, s])
            core, peri = (r, s) if omega[r, r] >= omega[s, s] else (s, r)
            pairs.append({"r": r, "s": s, "type": ix.TYPES[t], "p_type": prob[r, s, t],
                          "omega_rr": omega[r, r], "omega_ss": omega[s, s], "omega_rs": omega[r, s],
                          "core_or_denser": core, "periphery_or_sparser": peri})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(res / "best_pairs.csv", index=False, float_format="%.3f")
    _, _, fr1, tab1 = describe(level1, corr, meta, tickers)
    tab1.to_csv(res / "best_level1_groups.csv", index=False, float_format="%.3f")
    diag.update({f"pairs_{k}": v for k, v in fr.items()})
    diag.update({f"level1_pairs_{k}": v for k, v in fr1.items()})
    diag["median_p_type"] = float(pairs.p_type.median())
    diag["share_pairs_p_type_below_0.9"] = float((pairs.p_type < 0.9).mean())
    json.dump(diag, open(res / "best_diagnostics.json", "w"), indent=1)
    pd.DataFrame({"ticker": tickers, "block": b0, "level1": level1, "stay_share": stay,
                  "sector": meta["sector"].values, "sub_industry": meta["sub_industry"].values}
                 ).sort_values("stay_share").to_csv(res / "best_stock_stability.csv", index=False, float_format="%.3f")

    labels = {r: f"{r} {tab.set_index('block').top_sub_industry[r][:22]} ({tab.set_index('block')['size'][r]})"
              for r in order}
    s02.plot_window(window, corr, b0, meta["sector"].values, omega,
                    ROOT / "figures" / f"posterior_{window}_best_blocks")
    ix.plot_type_matrix(omega, T, order, labels, prob=prob,
                        path=ROOT / "figures" / f"posterior_{window}_best_types.png")

    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in diag.items()}, indent=1))
    print("\nnon-assortative pairs:")
    nonA = pairs[pairs.type != "assortative"].merge(
        tab[["block", "top_sub_industry", "size"]].rename(columns={"block": "core_or_denser"}), on="core_or_denser"
    ).merge(tab[["block", "top_sub_industry", "size"]].rename(columns={"block": "periphery_or_sparser"}),
            on="periphery_or_sparser", suffixes=("_core", "_periphery"))
    print(nonA[["type", "p_type", "omega_rr", "omega_ss", "omega_rs", "top_sub_industry_core", "size_core",
                "top_sub_industry_periphery", "size_periphery"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
