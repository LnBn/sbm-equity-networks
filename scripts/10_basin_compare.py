"""Compare the structure of every basin found by 08_basin_search.py for one window.

For each annealed start: DL, number of blocks, NMI with GICS sectors / sub-industries,
the per-block assortative share (pilot definition), the Liu et al. per-pair type
fractions, and NMI with the best basin and with the pilot partition. If these agree
across basins, the reported structure does not depend on which local optimum we hit.

Usage:  python scripts/10_basin_compare.py stress_2020
Output: results/posterior/<window>/basins_summary.csv
"""
import gzip
import importlib.util
import json
import pickle
import sys
from pathlib import Path

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


def main(window):
    res = ROOT / "results" / "posterior" / window
    r = pd.read_parquet(DATA / "log_returns.parquet").loc[slice(*s06.WINDOWS[window])]
    corr = spy_residuals(r.drop(columns="SPY"), r["SPY"]).corr().values
    tickers = list(r.drop(columns="SPY").columns)
    meta = pd.read_csv(DATA / "universe.csv", index_col=0).loc[tickers]
    sector, subind = cmp.labels_from(meta["sector"]), cmp.labels_from(meta["sub_industry"])
    starts = pickle.load(gzip.open(res / "basins.pkl.gz", "rb"))["starts"]
    parts = {s["seed"]: np.unique(s["bs"][0], return_inverse=True)[1] for s in starts}
    best_seed = min(starts, key=lambda s: s["dl"])["seed"]
    # The pilot partition exists only for the two pilot windows.
    fits = json.load(open(ROOT / "results" / "pilot_fits.json"))
    pilot_fit = next((f for f in fits if f["window"] == window and f["market"] == "spy"
                      and f["model"] == "dense-W-NDC-nested"), None)
    pilot = np.array(pilot_fit["labels"]) if pilot_fit else None
    pilot_dl = pilot_fit["dl_nats"] if pilot_fit else float("nan")
    rows = []
    for s in starts:
        b = parts[s["seed"]]
        blk = cmp.assortativity_summary(cmp.block_mean_matrix(corr, b), np.bincount(b))
        fr = ix.classify(corr, b)[2]
        rows.append({"seed": s["seed"], "dl": s["dl"], "B": b.max() + 1,
                     "nmi_sector": cmp.nmi(b, sector), "nmi_subind": cmp.nmi(b, subind),
                     "blocks_assortative": blk["frac_assortative_blocks"],
                     "pairs_A": fr["assortative"], "pairs_CP": fr["core-periphery"],
                     "pairs_D": fr["disassortative"],
                     "nmi_vs_best": post.nmi(b, parts[best_seed]),
                     "nmi_vs_pilot": post.nmi(b, pilot) if pilot is not None else np.nan})
    d = pd.DataFrame(rows).sort_values("dl")
    d.to_csv(res / "basins_summary.csv", index=False, float_format="%.4f")
    print(f"== {window}: {len(d)} basins; pilot partition DL {pilot_dl:.1f} ==")
    print(d.round(3).to_string(index=False))
    shown = d.drop(columns="seed").dropna(axis=1, how="all")   # no pilot column for non-pilot windows
    print("\ntop 5 vs all (min-max):")
    print(pd.DataFrame({"top5": shown.head(5).agg(["min", "max"]).round(3).astype(str).agg("-".join),
                        "all": shown.agg(["min", "max"]).round(3).astype(str).agg("-".join)}).to_string())
    print("\nSpearman with DL:", shown.corr(method="spearman")["dl"].round(2).to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
