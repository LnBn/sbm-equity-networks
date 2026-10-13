"""Data acquisition: S&P 500 constituents (Wikipedia) and daily prices (yfinance).

Everything downloaded is cached under data/ (or $SBM_DATA_DIR) so the analysis is reproducible
offline once the first download has run.
"""
import io
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
# Cache directory; override with SBM_DATA_DIR to keep downloaded data elsewhere.
DATA = Path(os.environ.get("SBM_DATA_DIR", ROOT / "data")).expanduser()
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "sbm-correlation-networks research script (personal, non-commercial)"}


def get_constituents(refresh=False):
    """Current S&P 500 members with GICS sector, sub-industry and date added."""
    path = DATA / "sp500_constituents.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date_added"])
    html = requests.get(WIKI_URL, headers=HEADERS, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    df = pd.DataFrame({
        # yfinance uses '-' where Wikipedia uses '.' (e.g. BRK.B -> BRK-B)
        "ticker": table["Symbol"].str.replace(".", "-", regex=False),
        "name": table["Security"],
        "sector": table["GICS Sector"],
        "sub_industry": table["GICS Sub-Industry"],
        "date_added": pd.to_datetime(table["Date added"], errors="coerce"),
    })
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def download_prices(tickers, start, end, batch_size=50, pause=2.0, refresh=False):
    """Daily adjusted close and volume for `tickers`, cached as parquet.

    Downloads in small batches with a pause between them to stay polite
    to Yahoo's servers.
    """
    close_path, vol_path = DATA / "close.parquet", DATA / "volume.parquet"
    if close_path.exists() and vol_path.exists() and not refresh:
        return pd.read_parquet(close_path), pd.read_parquet(vol_path)
    closes, vols = [], []
    for i in range(0, len(tickers), batch_size):
        batch = list(tickers[i:i + batch_size])
        raw = yf.download(batch, start=start, end=end, auto_adjust=True,
                          progress=False, threads=True)
        closes.append(raw["Close"])
        vols.append(raw["Volume"])
        time.sleep(pause)
    close = pd.concat(closes, axis=1).sort_index()
    volume = pd.concat(vols, axis=1).sort_index()
    close.to_parquet(close_path)
    volume.to_parquet(vol_path)
    return close, volume


def select_universe(constituents, close, volume, start, n_stocks=250, liquidity_days=252):
    """Choose a fixed universe while limiting look-ahead bias.

    1. Keep only firms that were already in the S&P 500 at `start`
       (removes firms that entered later *because* they grew).
    2. Require a complete price history over the whole sample.
    3. Rank by average dollar volume over the first `liquidity_days`
       trading days only, and keep the top `n_stocks`.

    Firms that left the index before today are still missing (survivorship
    bias); see README for discussion.
    """
    start = pd.Timestamp(start)
    early = constituents[constituents["date_added"] <= start]
    tickers = [t for t in early["ticker"] if t in close.columns]
    complete = close[tickers].notna().all()
    tickers = complete[complete].index.tolist()
    dollar_vol = (close[tickers] * volume[tickers]).iloc[:liquidity_days].mean()
    chosen = dollar_vol.sort_values(ascending=False).index[:n_stocks]
    meta = constituents.set_index("ticker").loc[chosen]
    return sorted(chosen), meta.sort_index()
