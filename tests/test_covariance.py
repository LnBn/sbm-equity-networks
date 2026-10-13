"""Checks for the covariance estimators.  Run:  python tests/test_covariance.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.covariance import (block_target_correlation, gmv_weights,  # noqa: E402
                            lw_shrink_to_block)


def simulate(T=252, N=40, K=4, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, K, N)
    factors = rng.standard_normal((T, K))
    market = rng.standard_normal((T, 1))
    X = 0.7 * factors[:, labels] + 0.5 * market + rng.standard_normal((T, N))
    return X, labels


def lw_constant_correlation_loop(X):
    """Direct, loop-based transcription of Ledoit & Wolf (2004) for comparison."""
    T, N = X.shape
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / T
    var = np.diag(S)
    R = S / np.sqrt(np.outer(var, var))
    rbar = (R.sum() - N) / (N * (N - 1))
    F = rbar * np.sqrt(np.outer(var, var))
    np.fill_diagonal(F, var)
    pi = rho = 0.0
    for i in range(N):
        for j in range(N):
            pi += ((Xc[:, i] * Xc[:, j] - S[i, j]) ** 2).mean()
            if i == j:
                rho += ((Xc[:, i] ** 2 - S[i, i]) ** 2).mean()
            else:
                th_i = ((Xc[:, i] ** 2 - S[i, i]) * (Xc[:, i] * Xc[:, j] - S[i, j])).mean()
                th_j = ((Xc[:, j] ** 2 - S[j, j]) * (Xc[:, i] * Xc[:, j] - S[i, j])).mean()
                rho += rbar / 2 * (np.sqrt(var[j] / var[i]) * th_i
                                   + np.sqrt(var[i] / var[j]) * th_j)
    gamma = ((F - S) ** 2).sum()
    return max(0.0, min(1.0, (pi - rho) / gamma / T))


def test_block_target_matches_brute_force():
    X, labels = simulate()
    R = np.corrcoef(X, rowvar=False)
    target = block_target_correlation(R, labels)
    N = len(R)
    for i, j in [(0, 1), (3, 17), (5, 5)]:
        if i == j:
            assert target[i, j] == 1.0
            continue
        mask = np.outer(labels == labels[i], labels == labels[j]) & ~np.eye(N, dtype=bool)
        assert np.isclose(target[i, j], R[mask].mean())


def test_lw_constant_correlation_intensity():
    X, _ = simulate(N=25)
    _, delta = lw_shrink_to_block(X)
    assert np.isclose(delta, lw_constant_correlation_loop(X))


def test_gmv_weights_sum_to_one():
    X, labels = simulate()
    cov, _ = lw_shrink_to_block(X, labels)
    assert np.isclose(gmv_weights(cov).sum(), 1.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
