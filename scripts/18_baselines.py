"""Compare each window's best-basin SBM partition with GICS and clustering baselines.

For every window, the best basin's level-0 partition is compared with GICS sectors and
sub-industries, Louvain modularity (dense positive-weight graph and kNN graph, seeded),
average-linkage hierarchical clustering at K = 11 and K = B, and the best basin's own
level-1 partition. For each partition:
  - NMI and ARI with GICS sectors, NMI with sub-industries, NMI with the SBM partition;
  - Delta DL: the description length of the flat weighted (real-normal, non-degree-corrected)
    SBM with that partition held fixed, minus the same quantity for the SBM partition
    (the measure used in 02_pilot.py). Positive = the baseline compresses the data worse;
  - the share of blocks passing the per-block assortativity test, and the mean within- and
    between-block residual correlation.

Delta DL is measured under the SBM's own objective, which the baselines were not built to
minimise, so it shows how far each baseline is from the SBM's posterior mode, not that the
baseline is "wrong" by its own criterion.

Usage:  python scripts/18_baselines.py
Output: results/posterior/baselines.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gzip
import pickle
import sys
from pathlib import Path

import graph_tool.all as gt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import compare as cmp  # noqa: E402
from src.data import DATA  # noqa: E402
from src.network import dense_graph  # noqa: E402
from src.preprocess import spy_residuals  # noqa: E402

WINDOWS = {"calm_2023_24": ("2023-01-01", "2024-12-31"), "stress_2020": ("2020-01-01", "2020-12-31"),
           "calm_2023": ("2023-01-01", "2023-12-31"), "calm_2024": ("2024-01-01", "2024-12-31"),
           "year_2019": ("2019-01-01", "2019-12-31"), "year_2021": ("2021-01-01", "2021-12-31"),
           "year_2022": ("2022-01-01", "2022-12-31")}
K_NN = 10


def main():
    gt.seed_rng(42)        # the DL evaluations are deterministic; seeded anyway (0 would mean random)
    rows = []
    for window, span in WINDOWS.items():
        returns = pd.read_parquet(DATA / "log_returns.parquet").loc[slice(*span)]
        stocks = returns.drop(columns="SPY")
        corr = spy_residuals(stocks, returns["SPY"]).corr().values
        meta = pd.read_csv(DATA / "universe.csv", index_col=0).loc[stocks.columns]
        sector, sub = cmp.labels_from(meta["sector"]), cmp.labels_from(meta["sub_industry"])
        starts = pickle.load(gzip.open(ROOT / "results/posterior" / window / "basins.pkl.gz", "rb"))["starts"]
        best = min(starts, key=lambda s: s["dl"])
        sbm = np.unique(best["bs"][0], return_inverse=True)[1]
        level1 = np.unique(np.asarray(best["bs"][1])[best["bs"][0]], return_inverse=True)[1]
        B = sbm.max() + 1
        partitions = {"SBM best basin": sbm, "SBM level 1": level1,
                      "GICS sector": sector, "GICS sub-industry": sub,
                      "Louvain (dense, w>0)": cmp.louvain_dense(corr),
                      "Louvain (kNN)": cmp.louvain_knn(corr, K_NN),
                      "Hierarchical K=11": cmp.hierarchical(corr, 11),
                      "Hierarchical K=B": cmp.hierarchical(corr, B)}
        g = dense_graph(corr)
        dl_sbm = cmp.dl_of_partition(g, sbm)
        for name, lab in partitions.items():
            lab = np.unique(lab, return_inverse=True)[1]
            mix = cmp.assortativity_summary(cmp.block_mean_matrix(corr, lab), np.bincount(lab))
            rows.append({"window": window, "partition": name, "K": int(lab.max() + 1),
                         "NMI_sector": cmp.nmi(lab, sector), "ARI_sector": cmp.ari(lab, sector),
                         "NMI_sub_industry": cmp.nmi(lab, sub), "NMI_vs_SBM": cmp.nmi(lab, sbm),
                         "DL_minus_SBM_nats": cmp.dl_of_partition(g, lab) - dl_sbm,
                         "frac_blocks_assortative": mix["frac_assortative_blocks"],
                         "within_corr": mix["mean_within"], "between_corr": mix["mean_between"]})
        print(window, "done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/posterior/baselines.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200):
        print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
