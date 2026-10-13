"""Summarise the posterior samples for one window (output of 06_posterior_sample.py).

1. Convergence: DL traces, per-chain DL mean/sd, Gelman-Rubin R-hat on DL and B.
2. Agreement: mean pairwise NMI / overlap of sampled partitions, within vs between chains,
   and of the pilot's MAP partition with the samples.
3. Modes: cluster the pooled samples into partition modes; for the main mode, per-stock
   membership marginals (which stocks are uncertain) and the consensus partition.
4. Structure on the posterior: NMI with GICS sectors / sub-industries and the share of
   assortative blocks, computed for every sample (so each comes with a credible interval).

Usage:  python scripts/07_posterior_summary.py calm_2023_24
Outputs: results/posterior/<window>/*.csv|json, figures/posterior_<window>_*.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gzip
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
from src.plotting import plt  # noqa: E402
from src.network import dense_graph  # noqa: E402
from src.preprocess import spy_residuals  # noqa: E402

THIN = 2            # use every 2nd sample in the pairwise / mode analysis


def rhat(chains):
    """Gelman-Rubin potential scale reduction for a list of 1-d traces."""
    x = np.array(chains, dtype=float)
    m, n = x.shape
    B = n * x.mean(axis=1).var(ddof=1)
    W = x.var(axis=1, ddof=1).mean()
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def plot_interactions(window, omega, T, prob, order, block_tab):
    """Left: omega (mean residual correlation) between consensus blocks; right: the
    Liu et al. interaction type of each pair, shaded by its posterior probability."""
    from matplotlib.colors import ListedColormap
    o = np.array(order)
    W, TT = omega[np.ix_(o, o)], T[np.ix_(o, o)]
    P = np.take_along_axis(prob[np.ix_(o, o)], np.nan_to_num(TT).astype(int)[..., None], 2)[..., 0]
    labels = [f"{r} {block_tab.top_sector[r][:14]} ({block_tab['size'][r]})" for r in o]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    lim = np.nanmax(np.abs(W))
    im = axes[0].imshow(W, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=axes[0], fraction=0.04, label="mean residual correlation $\\omega_{rs}$")
    cols = ["#2a78d6", "#eda100", "#c4384b"]                  # A, CP, D
    rgba = np.ones(TT.shape + (4,))
    for code, c in enumerate(cols):
        m = TT == code
        rgb = np.array([int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        rgba[m, :3] = rgb
        rgba[m, 3] = 0.25 + 0.75 * np.nan_to_num(P[m])
    axes[1].imshow(rgba, interpolation="nearest")
    for ax in axes:
        ax.set_xticks(range(len(o)))
        ax.set_yticks(range(len(o)), labels, fontsize=6)
        ax.set_xticklabels([str(r) for r in o], fontsize=6, rotation=90)
        ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    axes[1].legend(handles, ix.TYPES, loc="upper right", fontsize=7, frameon=True)
    axes[0].set_title("Block densities $\\omega$ (consensus partition, blocks by $\\omega_{rr}$ descending)", fontsize=9)
    axes[1].set_title("Interaction types (Liu et al. 2023); opacity = posterior probability", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"posterior_{window}_interaction_types.png", dpi=150)


def main(window):
    res = ROOT / "results" / "posterior" / window
    with gzip.open(res / "chains.pkl.gz", "rb") as f:
        run = pickle.load(f)
    chains, tickers = run["chains"], run["tickers"]
    meta = pd.read_csv(DATA / "universe.csv", index_col=0).loc[tickers]
    sector = cmp.labels_from(meta["sector"])
    subind = cmp.labels_from(meta["sub_industry"])
    returns = pd.read_parquet(DATA / "log_returns.parquet")
    lo, hi = {"calm_2023_24": ("2023-01-01", "2024-12-31"),
              "stress_2020": ("2020-01-01", "2020-12-31")}[window]
    r = returns.loc[lo:hi]
    corr = spy_residuals(r.drop(columns="SPY"), r["SPY"]).corr().values
    g = dense_graph(corr)

    # ---- 1. Convergence ------------------------------------------------------
    parts = [[s[0] for s in c["samples"]] for c in chains]       # level-0, per chain
    Bs = [[post.n_blocks(b) for b in p] for p in parts]
    conv = pd.DataFrame([{
        "seed": c["seed"], "start": c["start"], "dl_start_fit": c["dl_start"],
        "dl_mean": c["dl"].mean(), "dl_sd": c["dl"].std(), "dl_min": c["dl"].min(),
        "B_mean": np.mean(b), "B_sd": np.std(b), "minutes": c["seconds"] / 60}
        for c, b in zip(chains, Bs)])
    conv.to_csv(res / "convergence.csv", index=False, float_format="%.2f")
    diag = {"rhat_dl": rhat([c["dl"] for c in chains]), "rhat_B": rhat(Bs)}

    # ---- 2. Agreement within vs between chains -------------------------------
    thinned = [p[::THIN] for p in parts]
    within = np.mean([post.mean_pairwise(p, p, post.nmi, 500, seed=i) for i, p in enumerate(thinned)])
    between = np.mean([post.mean_pairwise(thinned[i], thinned[j], post.nmi, 500, seed=10 * i + j)
                       for i in range(len(thinned)) for j in range(i + 1, len(thinned))])
    within_ov = np.mean([post.mean_pairwise(p, p, post.overlap, 500, seed=i) for i, p in enumerate(thinned)])
    between_ov = np.mean([post.mean_pairwise(thinned[i], thinned[j], post.overlap, 500, seed=10 * i + j)
                          for i in range(len(thinned)) for j in range(i + 1, len(thinned))])
    pilot = json.load(open(ROOT / "results" / "pilot_fits.json"))
    map_labels = np.array(next(f["labels"] for f in pilot if f["window"] == window
                               and f["model"] == "dense-W-NDC-nested" and f["market"] == "spy"))
    pooled = [b for p in thinned for b in p]
    map_nmi = np.mean([post.nmi(map_labels, b) for b in pooled[::5]])
    diag.update(nmi_within_chain=within, nmi_between_chains=between,
                overlap_within_chain=within_ov, overlap_between_chains=between_ov,
                nmi_map_vs_samples=map_nmi, map_B=int(map_labels.max() + 1),
                map_dl=float(chains[0]["dl_start"]),
                sampled_dl_mean=float(np.mean([c["dl"].mean() for c in chains])))

    # ---- 3. Modes -------------------------------------------------------------
    modes = post.partition_modes(g, pooled)
    main_mode = modes[0]
    P = main_mode["marginals"]
    pmax = P.max(axis=1)
    ent = -np.nansum(np.where(P > 0, P * np.log(P), 0), axis=1)
    stock = pd.DataFrame({"ticker": tickers, "sector": meta["sector"].values,
                          "sub_industry": meta["sub_industry"].values,
                          "consensus_block": main_mode["partition"],
                          "map_block": map_labels, "p_max": pmax, "entropy": ent})
    stock.sort_values("p_max").to_csv(res / "stock_membership.csv", index=False, float_format="%.3f")
    mode_tab = pd.DataFrame([{"mode": k, "weight": m["weight"], "B": m["B"],
                              "nmi_with_main": post.nmi(m["partition"], main_mode["partition"])}
                             for k, m in enumerate(modes)])
    mode_tab.to_csv(res / "modes.csv", index=False, float_format="%.3f")
    diag.update(n_modes=len(modes), main_mode_weight=main_mode["weight"],
                share_stocks_pmax_below_090=float((pmax < 0.9).mean()),
                share_stocks_pmax_below_050=float((pmax < 0.5).mean()))

    # ---- 4. Structure, with posterior uncertainty ----------------------------------
    rows = []
    for b in pooled[::2]:
        b = np.unique(b, return_inverse=True)[1]       # labels 0..B-1
        M = cmp.block_mean_matrix(corr, b)
        sizes = np.bincount(b)
        a = cmp.assortativity_summary(M, sizes)
        rows.append({"B": post.n_blocks(b), "nmi_sector": cmp.nmi(b, sector),
                     "nmi_subind": cmp.nmi(b, subind),
                     "blocks_assortative": a["frac_assortative_blocks"]})
    struct = pd.DataFrame(rows)
    q = struct.quantile([0.025, 0.5, 0.975]).T
    q.columns = ["q025", "median", "q975"]
    M = cmp.block_mean_matrix(corr, map_labels)
    q.loc["B", "map"] = map_labels.max() + 1
    q.loc["nmi_sector", "map"] = cmp.nmi(map_labels, sector)
    q.loc["nmi_subind", "map"] = cmp.nmi(map_labels, subind)
    q.loc["blocks_assortative", "map"] = cmp.assortativity_summary(M, np.bincount(map_labels))["frac_assortative_blocks"]
    # Liu et al. (2023) pairwise interaction types, per sample and for the MAP.
    fr = pd.DataFrame([ix.classify(corr, b)[2] for b in pooled[::2]])
    for t in ix.TYPES:
        q.loc[f"pairs_{t}", ["q025", "median", "q975"]] = fr[t].quantile([0.025, 0.5, 0.975]).values
    map_fr = ix.classify(corr, map_labels)[2]
    for t in ix.TYPES:
        q.loc[f"pairs_{t}", "map"] = map_fr[t]
    q.to_csv(res / "structure_posterior.csv", float_format="%.3f")

    # Interaction-type matrix for the main mode's consensus partition, with the
    # posterior probability of each cell's type (samples aligned to the consensus labels).
    cons = np.unique(main_mode["partition"], return_inverse=True)[1]
    omega, T, cons_fr = ix.classify(corr, cons)
    counts, npair = ix.aligned_type_probabilities(corr, pooled[::4], cons)
    with np.errstate(invalid="ignore"):
        prob = counts / npair[..., None]
    sizes = np.bincount(cons)
    order = [r for r in np.argsort(-np.diag(omega)) if sizes[r] >= 2]   # Liu et al.: omega_rr descending
    dom = pd.crosstab(cons, meta["sector"].values)
    block_tab = pd.DataFrame({"block": range(len(sizes)), "size": sizes, "omega_rr": np.diag(omega),
                              "top_sector": dom.idxmax(axis=1).values,
                              "top_sector_share": (dom.max(axis=1) / sizes).values,
                              "members": [" ".join(np.array(tickers)[cons == r]) for r in range(len(sizes))]})
    block_tab.iloc[order].to_csv(res / "consensus_blocks.csv", index=False, float_format="%.3f")
    pairs = []
    for i, r in enumerate(order):
        for s in order[i + 1:]:
            pairs.append({"r": r, "s": s, "omega_rr": omega[r, r], "omega_ss": omega[s, s],
                          "omega_rs": omega[r, s], "type": ix.TYPES[int(T[r, s])],
                          "p_type": prob[r, s, int(T[r, s])], "n_samples": int(npair[r, s])})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(res / "consensus_pairs.csv", index=False, float_format="%.3f")
    diag.update({f"consensus_pairs_{k}": v for k, v in cons_fr.items()})
    diag["consensus_median_p_type"] = float(pairs.p_type.median())
    plot_interactions(window, omega, T, prob, order, block_tab)
    json.dump(diag, open(res / "diagnostics.json", "w"), indent=1)

    print(f"== {window} ==")
    print(conv.round(2).to_string(index=False))
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in diag.items()}, indent=1))
    print(mode_tab.head(8).round(3).to_string(index=False))
    print(q.round(3).to_string())
    print("\nleast certain stocks (main mode):")
    print(stock.sort_values("p_max").head(12)[["ticker", "sector", "sub_industry", "p_max"]].round(2).to_string(index=False))

    # ---- Figures ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for c in chains:
        tr = np.concatenate([c["dl_burn"], c["dl"]])
        axes[0].plot(tr, lw=0.7, label=f"seed {c['seed']} ({c['start']})")
    axes[0].axvline(run["n_burn"], color="k", lw=0.8, ls="--")
    axes[0].axhline(chains[0]["dl_start"], color="grey", lw=0.8, ls=":", label="pilot MAP DL")
    axes[0].set(xlabel="sweep call (10 merge-split sweeps each)", ylabel="description length (nats)",
                title="DL traces (burn-in left of dashed line)")
    axes[0].legend(fontsize=7)
    for b, c in zip(Bs, chains):
        axes[1].hist(b, bins=np.arange(min(map(min, Bs)) - 0.5, max(map(max, Bs)) + 1.5),
                     histtype="step", lw=1.2, label=f"seed {c['seed']}")
    axes[1].axvline(map_labels.max() + 1, color="k", lw=1, ls="--", label="pilot MAP")
    axes[1].set(xlabel="number of blocks B (level 0)", ylabel="samples", title="Posterior of B")
    axes[1].legend(fontsize=7)
    axes[2].hist(pmax, bins=20, color="#2a78d6")
    axes[2].set(xlabel="max membership probability", ylabel="stocks",
                title="Per-stock membership certainty (main mode)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"posterior_{window}_diagnostics.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1])
