"""Shared matplotlib style: colour-blind-checked categorical palette, light axes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (never cycled): blue, orange, aqua, yellow, magenta, green, violet, red
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8985"
STRESS_SHADE = "#e9e8e4"
SECTOR_COLOURS = {   # one fixed colour per GICS sector, shared by all figures
    "Communication Services": "#6250d6", "Consumer Discretionary": "#eb6834",
    "Consumer Staples": "#1baf7a", "Energy": "#5c4033", "Financials": "#2a78d6",
    "Health Care": "#e87ba4", "Industrials": "#8a8985", "Information Technology": "#00a3c4",
    "Materials": "#eda100", "Real Estate": "#008300", "Utilities": "#c4384b"}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#b5b4ae",
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "lines.linewidth": 1.6,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
})


def savefig(fig, path, **kwargs):
    """Save a figure twice: PNG for quick viewing and PDF (vector) for papers.

    `path` may end in .png, .pdf or have no suffix; both files are written."""
    from pathlib import Path
    path = Path(path)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), **kwargs)
