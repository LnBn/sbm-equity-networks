"""Download constituents and prices, choose the universe, save log returns.

Usage:  python scripts/01_download.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DATA, download_prices, get_constituents, select_universe  # noqa: E402

START, END = "2010-01-01", "2026-10-01"
N_STOCKS = 250

constituents = get_constituents()
tickers = constituents["ticker"].tolist() + ["SPY"]
close, volume = download_prices(tickers, START, END)
print(f"downloaded {close.shape[1]} series, {close.shape[0]} days "
      f"({close.index[0].date()} to {close.index[-1].date()})")

n_early = (constituents["date_added"] <= START).sum()
print(f"constituents in the index by {START}: {n_early} of {len(constituents)}")

universe, meta = select_universe(constituents, close, volume, START, N_STOCKS)
print(f"universe: {len(universe)} stocks")
print(meta["sector"].value_counts().to_string())

# Daily log returns of the universe plus SPY (used to remove the market mode).
prices = close[universe + ["SPY"]].dropna()
log_returns = np.log(prices).diff().dropna()
log_returns.to_parquet(DATA / "log_returns.parquet")
meta.to_csv(DATA / "universe.csv")
print(f"saved log returns: {log_returns.shape}")
