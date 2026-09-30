"""RSA utilities with statistically valid tests.

The entries of an RDM are NOT independent (each stimulus appears in n-1
pairs), so the usual Spearman p-value is invalid. We instead use a
permutation (Mantel-style) test: shuffle the stimulus labels of one RDM
(rows and columns together) and rebuild the null distribution.
"""
import numpy as np
from scipy.stats import rankdata


def correlation_rdm(acts, center=True):
    """1 - Pearson correlation between stimulus patterns. acts: (N, features).

    center=True subtracts each feature's mean across stimuli first. For GPT-2
    this matters: a few huge-magnitude residual dimensions otherwise dominate
    the correlation and hide stimulus-specific structure.
    """
    X = np.asarray(acts, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    rdm = 1.0 - Xn @ Xn.T
    rdm = (rdm + rdm.T) / 2
    np.fill_diagonal(rdm, 0.0)
    return rdm


def _rank_matrix(rdm):
    """Symmetric matrix whose off-diagonal entries are the ranks of the RDM."""
    n = rdm.shape[0]
    iu = np.triu_indices(n, 1)
    M = np.zeros((n, n))
    M[iu] = rankdata(rdm[iu])
    return M + M.T


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return np.nan if denom == 0 else float(a @ b / denom)


def spearman_rsa(rdm_a, rdm_b):
    """Plain Spearman correlation of RDM upper triangles (no p-value)."""
    iu = np.triu_indices(rdm_a.shape[0], 1)
    return _pearson(rankdata(rdm_a[iu]), rankdata(rdm_b[iu]))


def mantel_spearman(rdm_a, rdm_b, n_perm=5000, seed=0):
    """Spearman RSA between rdm_a and rdm_b with a label-permutation p-value.

    One-sided (H1: positive alignment). Returns dict(rho, p, null95).
    """
    n = rdm_a.shape[0]
    iu = np.triu_indices(n, 1)
    ra = _rank_matrix(rdm_a)
    rb = rankdata(rdm_b[iu])
    obs = _pearson(ra[iu], rb)
    if np.isnan(obs):
        return dict(rho=np.nan, p=np.nan, null95=np.nan)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = _pearson(ra[np.ix_(perm, perm)][iu], rb)
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return dict(rho=obs, p=float(p), null95=float(np.percentile(null, 95)))


def partial_mantel_spearman(rdm_a, rdm_b, rdm_control, n_perm=5000, seed=0):
    """Partial Spearman(a, b | control) with a label-permutation p-value.

    Asks: does rdm_a share structure with rdm_b BEYOND what rdm_control
    (e.g. plain Euclidean distance) already explains? Only rdm_a is permuted.
    """
    n = rdm_a.shape[0]
    iu = np.triu_indices(n, 1)
    ra = _rank_matrix(rdm_a)
    rb = rankdata(rdm_b[iu])
    rc = rankdata(rdm_control[iu])
    r_bc = _pearson(rb, rc)

    def partial(va):
        r_ab, r_ac = _pearson(va, rb), _pearson(va, rc)
        denom = np.sqrt((1 - r_ac ** 2) * (1 - r_bc ** 2))
        if denom == 0 or np.isnan(denom):
            return np.nan
        return (r_ab - r_ac * r_bc) / denom

    obs = partial(ra[iu])
    if np.isnan(obs):
        return dict(rho=np.nan, p=np.nan, null95=np.nan)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = partial(ra[np.ix_(perm, perm)][iu])
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return dict(rho=obs, p=float(p), null95=float(np.percentile(null, 95)))


def partial_spearman(rdm_a, rdm_b, rdm_control):
    """Observed partial Spearman(a, b | control), no p-value (fast; for seed sweeps)."""
    iu = np.triu_indices(rdm_a.shape[0], 1)
    ra, rb, rc = (rankdata(r[iu]) for r in (rdm_a, rdm_b, rdm_control))
    r_ab, r_ac, r_bc = _pearson(ra, rb), _pearson(ra, rc), _pearson(rb, rc)
    denom = np.sqrt((1 - r_ac ** 2) * (1 - r_bc ** 2))
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float((r_ab - r_ac * r_bc) / denom)