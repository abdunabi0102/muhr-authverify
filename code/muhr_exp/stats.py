"""Statistics used in Paper 1 tables (no SciPy dependency)."""
from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _betainc_inv(a, b, q, iters=200):
    """Inverse regularized incomplete beta by bisection on the binomial CDF identity."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _betainc(a, b, x):
    # integer a, b: I_x(a, b) = P[Bin(a+b-1, x) >= a]
    n = int(a + b - 1)
    return sum(math.comb(n, j) * x ** j * (1 - x) ** (n - j) for j in range(int(a), n + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact CI. For large n with k = 0 the upper bound is 1 - (alpha/2)^(1/n)."""
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else (_betainc_inv(k, n - k + 1, alpha / 2) if n <= 2000 else
                             _poisson_lo(k, alpha) / n)
    if k == n:
        hi = 1.0
    elif k == 0:
        hi = 1 - (alpha / 2) ** (1 / n)
    elif n <= 2000:
        hi = _betainc_inv(k + 1, n - k, 1 - alpha / 2)
    else:
        hi = _poisson_hi(k, alpha) / n
    return lo, hi


def _poisson_cdf(k, lam):
    s, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i:
            term *= lam / i
        s += term
    return s


def _poisson_hi(k, alpha):
    lo, hi = 0.0, 10.0 * (k + 10)
    for _ in range(200):
        mid = (lo + hi) / 2
        if _poisson_cdf(k, mid) > alpha / 2:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _poisson_lo(k, alpha):
    if k == 0:
        return 0.0
    lo, hi = 0.0, 10.0 * (k + 10)
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - _poisson_cdf(k - 1, mid) < alpha / 2:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def poisson_interval(k: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Poisson CI for an observed count k (used to compare with a predicted mean)."""
    return _poisson_lo(k, alpha), _poisson_hi(k, alpha)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A ok, B fail) and c (A fail, B ok)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_mean_ci(x, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals):
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
