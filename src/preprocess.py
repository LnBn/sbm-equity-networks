"""Return preprocessing: removing the market mode from daily log returns.

Raw equity correlations are dominated by a single 'market' factor (the top
eigenvalue of the correlation matrix is typically 30-50% of the trace), which
hides the finer group structure. Two standard ways to remove it are offered.
"""
import numpy as np
import pandas as pd


def spy_residuals(returns, market):
    """Residuals from regressing each stock on the market (OLS with intercept).

    returns : DataFrame (days x stocks), market : Series (days).
    """
    X = np.column_stack([np.ones(len(market)), market.values])
    beta, *_ = np.linalg.lstsq(X, returns.values, rcond=None)
    resid = returns.values - X @ beta
    return pd.DataFrame(resid, index=returns.index, columns=returns.columns)


def remove_top_pcs(returns, k=1):
    """Standardise returns and subtract the projection on the top-k principal components."""
    Z = (returns - returns.mean()) / returns.std()
    U, s, Vt = np.linalg.svd(Z.values, full_matrices=False)
    top = (U[:, :k] * s[:k]) @ Vt[:k]
    return pd.DataFrame(Z.values - top, index=returns.index, columns=returns.columns)


def correlation(returns):
    """Pearson correlation matrix as a DataFrame."""
    return returns.corr()


def preprocess(returns, market, mode):
    """Return the series whose correlations define the network.

    mode: 'raw' (market mode kept), 'spy' (market-regression residuals),
    or 'pc1' (top principal component removed).
    """
    if mode == "raw":
        return returns
    if mode == "spy":
        return spy_residuals(returns, market)
    if mode == "pc1":
        return remove_top_pcs(returns, k=1)
    raise ValueError(f"unknown mode {mode!r}")
