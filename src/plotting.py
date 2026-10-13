"""Shared matplotlib style: colour-blind-checked categorical palette, light axes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (never cycled): blue, orange, aqua, yellow, magenta, green, violet, red
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8985"
STRESS_SHADE = "#e9e8e4"

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
