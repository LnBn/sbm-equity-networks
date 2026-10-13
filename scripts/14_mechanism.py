"""Name the structures behind the calendar-year results (best basin of each year).

For each year: (1) cross-sector blocks, i.e. tightly correlated blocks
(omega_rr >= 0.3) whose dominant GICS sector covers less than half the members, and
(2) the core-periphery pairs with the largest margin omega_rs - min(omega_rr, omega_ss),
each block described by its members and their sectors.

Usage:  python scripts/14_mechanism.py
Output: results/posterior/mechanism.md, results/posterior/mechanism_cross_sector.csv,
        results/posterior/mechanism_cp_pairs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import DATA  # noqa: E402

YEARS = [("year_2019", "2019"), ("stress_2020", "2020 (COVID crash)"), ("year_2021", "2021"),
         ("year_2022", "2022 (rate shock)"), ("calm_2023", "2023"), ("calm_2024", "2024")]
MIN_OMEGA, MAX_SECTOR_SHARE, TOP_CP = 0.3, 0.5, 6

SHORT = {"Information Technology": "IT", "Consumer Discretionary": "Disc", "Consumer Staples": "Stap",
         "Communication Services": "Comm", "Health Care": "HC", "Real Estate": "RE",
         "Financials": "Fin", "Industrials": "Ind", "Materials": "Mat", "Utilities": "Util",
         "Energy": "Ener"}


def describe(members, sector):
    """'BA (Ind), CCL (Disc), ...' for a space-separated member list."""
    return ", ".join(f"{t} ({SHORT.get(sector[t], sector[t])})" for t in members.split())


def main():
    sector = pd.read_csv(DATA / "universe.csv", index_col=0)["sector"]
    cross_rows, cp_rows, md = [], [], ["# Named structures in each year's best basin\n"]
    for w, label in YEARS:
        res = ROOT / "results" / "posterior" / w
        blocks = pd.read_csv(res / "best_blocks.csv").set_index("block")
        pairs = pd.read_csv(res / "best_pairs.csv")
        cross = blocks[(blocks.omega_rr >= MIN_OMEGA) & (blocks.sector_share < MAX_SECTOR_SHARE)]
        cp = pairs[pairs.type == "core-periphery"].copy()
        cp["margin"] = cp.omega_rs - np.minimum(cp.omega_rr, cp.omega_ss)
        top = cp.sort_values("margin", ascending=False).head(TOP_CP)
        md.append(f"## {label}\n")
        md.append(f"{len(blocks)} blocks; {len(cross)} cross-sector blocks with ω_rr ≥ {MIN_OMEGA}; "
                  f"{len(cp)}/{len(pairs)} core–periphery pairs ({(cp.margin > 0.05).sum()} with margin > 0.05).\n")
        md.append("**Cross-sector blocks**\n")
        for b, r in cross.sort_values("omega_rr", ascending=False).iterrows():
            md.append(f"- block {b} (ω_rr {r.omega_rr:.2f}, {r['size']} stocks): {describe(r.members, sector)}")
            cross_rows.append({"window": w, "block": b, "omega_rr": r.omega_rr, "size": r["size"],
                               "sector_share": r.sector_share, "members": r.members})
        md.append("\n**Largest-margin core–periphery pairs** (core = denser block)\n")
        for _, r in top.iterrows():
            core, peri = blocks.loc[r.core_or_denser], blocks.loc[r.periphery_or_sparser]
            md.append(f"- margin {r.margin:.3f} (ω_core {core.omega_rr:.2f}, ω_periphery {peri.omega_rr:.2f}, "
                      f"ω_between {r.omega_rs:.2f}): core [{describe(core.members, sector)}] → "
                      f"periphery [{describe(peri.members, sector)}]")
            cp_rows.append({"window": w, "margin": r.margin, "omega_core": core.omega_rr,
                            "omega_periphery": peri.omega_rr, "omega_between": r.omega_rs,
                            "core_members": core.members, "periphery_members": peri.members})
        md.append("")
    (ROOT / "results" / "posterior" / "mechanism.md").write_text("\n".join(md))
    pd.DataFrame(cross_rows).to_csv(ROOT / "results" / "posterior" / "mechanism_cross_sector.csv",
                                    index=False, float_format="%.3f")
    pd.DataFrame(cp_rows).to_csv(ROOT / "results" / "posterior" / "mechanism_cp_pairs.csv",
                                 index=False, float_format="%.3f")
    print("\n".join(md))


if __name__ == "__main__":
    main()
