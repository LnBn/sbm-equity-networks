"""Covariance estimators and global minimum-variance (GMV) portfolios.

All estimators take a (T x N) array of returns from the estimation window.

The block estimators share one idea: replace each sample correlation r_ij by
the average correlation between the groups of i and j (a 'block-constant'
correlation matrix), keeping each stock's own sample variance. With one group
this is the constant-correlation model of Elton & Gruber; with GICS sectors or
SBM blocks it uses that partition instead.
"""
import numpy as np
from sklearn.covariance import LedoitWolf


def sample_cov(X):
    return np.cov(X, rowvar=False)


def ledoit_wolf_identity(X):
    """Ledoit-Wolf (2004a) shrinkage towards a scaled identity matrix."""
    return LedoitWolf().fit(X).covariance_


def block_target_correlation(R, labels):
    """Block-constant correlation matrix: r_ij -> mean r_kl over k in b(i), l in b(j), k != l.

    Singleton blocks have no within-block pairs; their (single) diagonal
    entry is set to the overall mean correlation.
    """
    N = len(R)
    K = labels.max() + 1
    onehot = np.zeros((N, K))
    onehot[np.arange(N), labels] = 1
    R_off = R - np.eye(N)                       # zero the diagonal
    sums = onehot.T @ R_off @ onehot            # sum of r_ij over block pairs
    sizes = onehot.sum(axis=0)
    pairs = np.outer(sizes, sizes) - np.diag(sizes)  # exclude i == j pairs
    overall = R_off.sum() / (N * (N - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(pairs > 0, sums / pairs, overall)
    target = means[labels][:, labels]
    np.fill_diagonal(target, 1.0)
    return target


def _nearest_psd(S, floor=1e-10):
    """Clip negative eigenvalues (block averaging can, rarely, break PSD)."""
    vals, vecs = np.linalg.eigh((S + S.T) / 2)
    return (vecs * np.maximum(vals, floor * vals.max())) @ vecs.T


def block_cov(X, labels):
    """Covariance with block-constant correlations and sample variances."""
    S = sample_cov(X)
    sd = np.sqrt(np.diag(S))
    R = S / np.outer(sd, sd)
    return _nearest_psd(block_target_correlation(R, labels) * np.outer(sd, sd))


def lw_shrink_to_block(X, labels=None):
    """Ledoit-Wolf shrinkage of the sample covariance towards a block target.

    labels=None gives the original constant-correlation target of
    Ledoit & Wolf (2004, 'Honey, I shrunk the sample covariance matrix').
    With labels, the single average correlation is replaced by block-pair
    averages; the optimal-intensity formula carries over unchanged because,
    as in the original paper, the averaged correlations are treated as fixed.
    """
    T, N = X.shape
    X = X - X.mean(axis=0)
    S = X.T @ X / T                              # MLE covariance (1/T)
    var = np.diag(S)
    sd = np.sqrt(var)
    R = S / np.outer(sd, sd)
    if labels is None:
        labels = np.zeros(N, dtype=int)
    Rbar = block_target_correlation(R, labels)
    F = Rbar * np.outer(sd, sd)                  # shrinkage target

    # pi: sum of asymptotic variances of the sample covariances
    X2 = X ** 2
    pi_mat = X2.T @ X2 / T - S ** 2
    pi_hat = pi_mat.sum()
    # rho: sum of asymptotic covariances between target and sample entries
    theta = (X ** 3).T @ X / T - var[:, None] * S   # theta[i, j] = theta_ii,ij
    ratio = np.outer(1 / sd, sd)                     # sqrt(s_jj / s_ii)
    rho_off = Rbar / 2 * (ratio * theta + ratio.T * theta.T)
    np.fill_diagonal(rho_off, 0)
    rho_hat = np.trace(pi_mat) + rho_off.sum()
    # gamma: misspecification of the target
    gamma_hat = ((F - S) ** 2).sum()
    delta = max(0.0, min(1.0, (pi_hat - rho_hat) / gamma_hat / T))
    return delta * F + (1 - delta) * S, delta


def gmv_weights(cov):
    """Global minimum-variance weights w = C^-1 1 / (1' C^-1 1); shorting allowed."""
    ones = np.ones(len(cov))
    x = np.linalg.solve(cov, ones)
    return x / x.sum()
