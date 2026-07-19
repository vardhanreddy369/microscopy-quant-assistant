"""Reproducible marker-positivity calling for standard fluorescence.

The percent-positive call — what fraction of cells carry a marker such as TUNEL,
cleaved caspase-3 or Ki67 — is one of the least reproducible steps in the
literature. A systematic review of roughly 9,000 immunofluorescence papers found
fewer than 10% report their thresholding at all, and manual thresholds introduce
documented bias (Jiang et al., *Front. Cell. Neurosci.* 2023).

This module makes that call in a way that is objective, self-assessing, and
self-documenting:

* **A two-component Gaussian mixture** over the per-object marker intensities,
  fitted by a deterministic EM, replaces a hand-set or histogram threshold. Each
  object gets a posterior probability of being positive; the threshold is the
  crossover where that posterior is 0.5.

* **A bimodality test** answers the question a percentage alone hides: is there a
  distinct positive population at all, or is the method cutting one continuous
  population in half? Model selection by BIC compares a one-component fit against
  the two-component fit. BIC is used rather than a likelihood-ratio test because
  the number of mixture components is a non-regular testing problem — under the
  null the second component's parameters are unidentifiable, so the likelihood
  ratio does not have its usual chi-squared distribution. BIC has no such
  requirement.

* **Negative-control anchoring** derives the threshold from a control image
  imaged alongside the sample, which is the field's gold standard for an
  objective cut-off, and reports which control was used.

* **A reproducibility report** records every input and parameter, so a reader
  can reproduce the number exactly — the specific thing the review found missing.

Everything here is deterministic: the EM is initialised from the data (no random
restarts), so the same input always yields the same call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A positive population is reported only when two independent, established
# criteria agree. Neither is tuned on this project's data.
#
# 1. Model selection. Kass & Raftery's convention: a BIC difference above 10 is
#    "very strong" evidence for the more complex model. This asks whether two
#    components fit better than one at all.
#
# 2. Separation. Ashman's D = sqrt(2)|m1-m0| / sqrt(s0^2 + s1^2); D > 2 is the
#    standard cut-off for two genuinely distinct populations rather than one
#    skewed one. This asks whether the two components are actually apart.
#
# Both are required because either alone is foolable: a non-Gaussian single
# population can win on BIC (two Gaussians approximate a flat or skewed shape
# better than one), and D can look moderate for a mildly skewed single
# population. Demanding evidence *and* separation is what makes an all-negative
# field read as one population.
STRONG_EVIDENCE = 10.0
MIN_SEPARATION = 2.0


def ashman_d(fit: MixtureFit) -> float:
    """Standardised separation between the two fitted components."""
    if fit.n_components != 2:
        return 0.0
    m0, m1 = fit.means
    s0, s1 = fit.sigmas
    return float(np.sqrt(2.0) * abs(m1 - m0) / np.sqrt(s0 * s0 + s1 * s1))

# EM guards. A component is not allowed to collapse to zero width (which would
# send its likelihood to infinity), and the fit stops once the log-likelihood
# settles.
_MIN_SIGMA = 1e-3
_MAX_ITERS = 200
_TOLERANCE = 1e-6


@dataclass
class MixtureFit:
    """A fitted one- or two-component Gaussian model."""

    n_components: int
    weights: np.ndarray
    means: np.ndarray
    sigmas: np.ndarray
    log_likelihood: float
    n_observations: int

    @property
    def n_parameters(self) -> int:
        # Each component contributes a mean and a sigma; a k-component mixture
        # adds k-1 free mixing weights.
        return 2 * self.n_components + (self.n_components - 1)

    @property
    def bic(self) -> float:
        """Lower is better. BIC = k ln(n) - 2 ln(L)."""
        return self.n_parameters * np.log(self.n_observations) - 2.0 * self.log_likelihood


@dataclass
class PositivityResult:
    """The outcome of calling positivity on one set of per-object intensities."""

    threshold: float
    posteriors: np.ndarray
    positive: np.ndarray
    method: str
    n_total: int
    n_positive: int
    bimodal: bool
    delta_bic: float
    fit: MixtureFit | None = None
    control_reference: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def percent_positive(self) -> float:
        return (100.0 * self.n_positive / self.n_total) if self.n_total else float("nan")


def _gaussian_logpdf(values: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), _MIN_SIGMA)
    z = (values - mean) / sigma
    return -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


def fit_one_component(values: np.ndarray) -> MixtureFit:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    sigma = max(float(values.std(ddof=0)), _MIN_SIGMA)
    log_likelihood = float(_gaussian_logpdf(values, mean, sigma).sum())
    return MixtureFit(
        n_components=1,
        weights=np.array([1.0]),
        means=np.array([mean]),
        sigmas=np.array([sigma]),
        log_likelihood=log_likelihood,
        n_observations=values.size,
    )


def fit_two_component(values: np.ndarray) -> MixtureFit:
    """Fit a two-component 1-D Gaussian mixture by EM.

    Initialised deterministically by splitting the sorted values at their median,
    so there are no random restarts and the same input always gives the same fit.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size

    split = float(np.median(values))
    low, high = values[values <= split], values[values > split]
    if low.size == 0 or high.size == 0:
        # Median coincides with the extreme; fall back to a mean split.
        split = float(values.mean())
        low, high = values[values <= split], values[values > split]
        if low.size == 0 or high.size == 0:
            return fit_one_component(values)

    means = np.array([low.mean(), high.mean()], dtype=np.float64)
    sigmas = np.array([
        max(low.std(ddof=0), _MIN_SIGMA),
        max(high.std(ddof=0), _MIN_SIGMA),
    ])
    weights = np.array([low.size / n, high.size / n], dtype=np.float64)

    previous = -np.inf
    log_likelihood = previous
    for _ in range(_MAX_ITERS):
        # E-step: responsibilities via the log-sum-exp trick.
        log_component = np.stack([
            np.log(weights[k]) + _gaussian_logpdf(values, means[k], sigmas[k])
            for k in range(2)
        ])
        log_norm = np.logaddexp(log_component[0], log_component[1])
        log_likelihood = float(log_norm.sum())
        responsibilities = np.exp(log_component - log_norm)

        # M-step.
        totals = responsibilities.sum(axis=1)
        totals = np.where(totals < 1e-12, 1e-12, totals)
        weights = totals / n
        means = (responsibilities * values).sum(axis=1) / totals
        variances = (responsibilities * (values - means[:, None]) ** 2).sum(axis=1) / totals
        sigmas = np.sqrt(np.maximum(variances, _MIN_SIGMA**2))

        if abs(log_likelihood - previous) < _TOLERANCE * max(1.0, abs(previous)):
            break
        previous = log_likelihood

    # Order the components so index 1 is always the marker-positive (brighter) one.
    order = np.argsort(means)
    return MixtureFit(
        n_components=2,
        weights=weights[order],
        means=means[order],
        sigmas=sigmas[order],
        log_likelihood=log_likelihood,
        n_observations=n,
    )


def _crossover(fit: MixtureFit) -> float:
    """Intensity where the two components are equally likely, between the means.

    Solves w0 N(x|m0,s0) = w1 N(x|m1,s1). Equal variances give a linear
    equation; otherwise a quadratic, of whose roots the one lying between the two
    means is the decision boundary.
    """
    (w0, w1), (m0, m1), (s0, s1) = fit.weights, fit.means, fit.sigmas
    lo, hi = (m0, m1) if m0 <= m1 else (m1, m0)

    a0, a1 = 1.0 / (2.0 * s0 * s0), 1.0 / (2.0 * s1 * s1)
    k = np.log(w0 / s0) - np.log(w1 / s1)

    quad_a = a0 - a1
    quad_b = -2.0 * a0 * m0 + 2.0 * a1 * m1
    quad_c = a0 * m0 * m0 - a1 * m1 * m1 - k

    if abs(quad_a) < 1e-12:
        if abs(quad_b) < 1e-12:
            return float(0.5 * (m0 + m1))
        root = -quad_c / quad_b
        return float(root if lo <= root <= hi else 0.5 * (m0 + m1))

    disc = quad_b * quad_b - 4.0 * quad_a * quad_c
    if disc < 0:
        return float(0.5 * (m0 + m1))
    sqrt_disc = np.sqrt(disc)
    roots = [(-quad_b + sqrt_disc) / (2.0 * quad_a),
             (-quad_b - sqrt_disc) / (2.0 * quad_a)]
    between = [r for r in roots if lo <= r <= hi]
    return float(between[0]) if between else float(0.5 * (m0 + m1))


def call_by_mixture(values: np.ndarray) -> PositivityResult:
    """Call positivity with the mixture model and the BIC bimodality test."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size

    if n < 3 or float(values.max()) <= float(values.min()):
        return PositivityResult(
            threshold=float("inf"),
            posteriors=np.zeros(n),
            positive=np.zeros(n, dtype=bool),
            method="mixture",
            n_total=n,
            n_positive=0,
            bimodal=False,
            delta_bic=0.0,
            notes=["too few or identical objects to fit a mixture"],
        )

    one = fit_one_component(values)
    two = fit_two_component(values)
    delta_bic = one.bic - two.bic  # positive favours two components
    separation = ashman_d(two)
    bimodal = (
        two.n_components == 2
        and delta_bic > STRONG_EVIDENCE
        and separation > MIN_SEPARATION
    )

    if not bimodal:
        # One population: nothing is called positive, and the report says why.
        if delta_bic <= STRONG_EVIDENCE:
            reason = (
                f"a one-component model fits about as well (ΔBIC {delta_bic:.1f}, "
                f"below {STRONG_EVIDENCE:.0f})"
            )
        else:
            reason = (
                f"the two fitted components are not distinct (Ashman's D "
                f"{separation:.1f}, below {MIN_SEPARATION:.0f}), so the split is "
                "one population rather than two"
            )
        return PositivityResult(
            threshold=float("inf"),
            posteriors=np.zeros(n),
            positive=np.zeros(n, dtype=bool),
            method="mixture",
            n_total=n,
            n_positive=0,
            bimodal=False,
            delta_bic=float(delta_bic),
            fit=two,
            notes=[
                f"no distinct positive population detected: {reason}. The "
                "percentage is left at zero rather than reporting a split the "
                "data does not support."
            ],
        )

    threshold = _crossover(two)
    # Posterior probability of the brighter (positive) component per object.
    log_pos = np.log(two.weights[1]) + _gaussian_logpdf(values, two.means[1], two.sigmas[1])
    log_neg = np.log(two.weights[0]) + _gaussian_logpdf(values, two.means[0], two.sigmas[0])
    posteriors = np.exp(log_pos - np.logaddexp(log_pos, log_neg))
    positive = values >= threshold

    return PositivityResult(
        threshold=float(threshold),
        posteriors=posteriors,
        positive=positive,
        method="mixture",
        n_total=n,
        n_positive=int(positive.sum()),
        bimodal=True,
        delta_bic=float(delta_bic),
        fit=two,
        notes=[
            f"two populations resolved (ΔBIC {delta_bic:.1f}); negative mean "
            f"{two.means[0]:.1f}, positive mean {two.means[1]:.1f}."
        ],
    )


def call_by_negative_control(
    values: np.ndarray,
    control_values: np.ndarray,
    percentile: float = 99.0,
    control_name: str = "negative control",
) -> PositivityResult:
    """Threshold from a negative control, the field's gold standard.

    The cut-off is a high percentile of the control's per-object intensities, so
    that only cells brighter than nearly all control cells are called positive.
    """
    values = np.asarray(values, dtype=np.float64)
    control = np.asarray(control_values, dtype=np.float64)
    n = values.size

    if control.size == 0:
        raise ValueError("negative-control thresholding needs control objects")

    threshold = float(np.percentile(control, percentile))
    positive = values >= threshold
    return PositivityResult(
        threshold=threshold,
        posteriors=(values >= threshold).astype(float),
        positive=positive,
        method="negative_control",
        n_total=n,
        n_positive=int(positive.sum()),
        bimodal=True,
        delta_bic=float("nan"),
        control_reference=control_name,
        notes=[
            f"threshold set at the {percentile:.0f}th percentile of {control.size} "
            f"objects in '{control_name}' ({threshold:.1f} on the 0-255 scale)."
        ],
    )


def call_by_manual(values: np.ndarray, threshold: float) -> PositivityResult:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= float(threshold)
    return PositivityResult(
        threshold=float(threshold),
        posteriors=(values >= float(threshold)).astype(float),
        positive=positive,
        method="manual",
        n_total=values.size,
        n_positive=int(positive.sum()),
        bimodal=True,
        delta_bic=float("nan"),
        notes=[f"threshold set manually at {float(threshold):.1f}."],
    )


def reproducibility_report(result: PositivityResult, context: dict) -> str:
    """A plain-text record complete enough to reproduce the number.

    ``context`` carries the surrounding facts the caller knows — the image name,
    the channels, the segmentation settings, software versions.
    """
    lines = [
        "MARKER POSITIVITY — REPRODUCIBILITY REPORT",
        "=" * 44,
    ]
    for key, value in context.items():
        lines.append(f"{key:24s}: {value}")

    lines += [
        f"{'method':24s}: {result.method}",
        f"{'objects measured':24s}: {result.n_total}",
        f"{'marker-positive':24s}: {result.n_positive}",
        f"{'percent positive':24s}: {result.percent_positive:.2f}%",
        f"{'threshold (0-255)':24s}: {result.threshold:.3f}",
        f"{'distinct populations':24s}: {'yes' if result.bimodal else 'no'}",
    ]
    if np.isfinite(result.delta_bic):
        lines.append(f"{'bimodality ΔBIC':24s}: {result.delta_bic:.2f}")
    if result.fit is not None and result.fit.n_components == 2:
        fit = result.fit
        lines += [
            f"{'negative component':24s}: mean {fit.means[0]:.2f}, sd {fit.sigmas[0]:.2f}, "
            f"weight {fit.weights[0]:.3f}",
            f"{'positive component':24s}: mean {fit.means[1]:.2f}, sd {fit.sigmas[1]:.2f}, "
            f"weight {fit.weights[1]:.3f}",
        ]
    if result.control_reference:
        lines.append(f"{'negative control':24s}: {result.control_reference}")
    for note in result.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)
