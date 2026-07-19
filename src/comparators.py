"""Baseline positivity methods, for an honest head-to-head against the mixture.

The point of a comparison is to find out what actually carries the result. These
are the alternatives a reviewer would reach for:

* ``gamma_mixture`` — a two-component Gamma mixture, the distribution
  [GammaGateR](https://pmc.ncbi.nlm.nih.gov/articles/PMC10541135/) uses for
  marker gating. Gamma is the textbook choice for non-negative, right-skewed
  intensity data, so this tests whether the Gaussian assumption in
  ``src/positivity.py`` was doing the work.
* ``kmeans`` — a 1-D two-cluster split, the simplest possible "find two groups".
* ``otsu`` and ``manual`` live in ``src/markers.py`` already.

None of these baselines answers "is there a positive population at all" — they
always return two groups. That question is the mixture method's actual
contribution, and the benchmark is built to show whether the *distribution*
matters or the *population test* does.
"""

from __future__ import annotations

import numpy as np
from scipy import special

_MAX_ITERS = 200
_TOLERANCE = 1e-6
_EPS = 1e-6


def _gamma_logpdf(values: np.ndarray, shape: float, scale: float) -> np.ndarray:
    shape = max(float(shape), _EPS)
    scale = max(float(scale), _EPS)
    return (
        (shape - 1.0) * np.log(values)
        - values / scale
        - shape * np.log(scale)
        - special.gammaln(shape)
    )


def gamma_mixture_threshold(values: np.ndarray) -> float:
    """Two-component Gamma mixture by EM; return the crossover threshold.

    The M-step uses method-of-moments for each component (shape = mean^2/var,
    scale = var/mean), which is the closed-form update GammaGateR relies on and
    keeps the fit deterministic. No random restarts, so the same input always
    gives the same threshold.
    """
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, _EPS)
    n = values.size
    if n < 3 or float(values.max()) <= float(values.min()):
        return float("inf")

    split = float(np.median(values))
    low, high = values[values <= split], values[values > split]
    if low.size == 0 or high.size == 0:
        return float("inf")

    def moments(sample, weights=None):
        if weights is None:
            mean, var = sample.mean(), sample.var()
        else:
            total = max(weights.sum(), _EPS)
            mean = (weights * sample).sum() / total
            var = (weights * (sample - mean) ** 2).sum() / total
        var = max(var, _EPS)
        return mean * mean / var, var / mean  # shape, scale

    low_shape, low_scale = moments(low)
    high_shape, high_scale = moments(high)
    shapes = np.array([low_shape, high_shape])
    scales = np.array([low_scale, high_scale])
    weights = np.array([low.size / n, high.size / n])

    previous = -np.inf
    for _ in range(_MAX_ITERS):
        log_comp = np.stack([
            np.log(weights[k]) + _gamma_logpdf(values, shapes[k], scales[k])
            for k in range(2)
        ])
        log_norm = np.logaddexp(log_comp[0], log_comp[1])
        loglik = float(log_norm.sum())
        resp = np.exp(log_comp - log_norm)

        for k in range(2):
            shapes[k], scales[k] = moments(values, resp[k])
        totals = resp.sum(axis=1)
        weights = np.maximum(totals, _EPS) / n

        if abs(loglik - previous) < _TOLERANCE * max(1.0, abs(previous)):
            break
        previous = loglik

    # Order components by mean (shape*scale); threshold = posterior-0.5 crossover
    # between the two means, found on a fine deterministic grid.
    means = shapes * scales
    lo, hi = float(min(means)), float(max(means))
    if hi <= lo:
        return float("inf")
    grid = np.linspace(lo, hi, 512)
    order = np.argsort(means)
    log_lo = np.log(weights[order[0]]) + _gamma_logpdf(grid, shapes[order[0]], scales[order[0]])
    log_hi = np.log(weights[order[1]]) + _gamma_logpdf(grid, shapes[order[1]], scales[order[1]])
    crossings = np.where(np.diff(np.sign(log_hi - log_lo)))[0]
    return float(grid[crossings[0]]) if crossings.size else float(0.5 * (lo + hi))


def kmeans_threshold(values: np.ndarray, iterations: int = 100) -> float:
    """1-D two-cluster k-means; threshold at the midpoint of the two centres.

    Deterministic: centres are initialised at the data's min and max.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3 or float(values.max()) <= float(values.min()):
        return float("inf")

    centres = np.array([values.min(), values.max()], dtype=np.float64)
    for _ in range(iterations):
        assignment = (np.abs(values[:, None] - centres[None, :]).argmin(axis=1))
        new = centres.copy()
        for k in range(2):
            members = values[assignment == k]
            if members.size:
                new[k] = members.mean()
        if np.allclose(new, centres):
            break
        centres = new
    return float(centres.mean())


def positive_fraction(values: np.ndarray, threshold: float) -> float:
    """Percent of objects at or above a threshold."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(threshold):
        return 0.0
    return 100.0 * float((values >= threshold).sum()) / values.size
