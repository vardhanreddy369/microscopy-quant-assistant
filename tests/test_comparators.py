"""Tests for the baseline comparators and the head-to-head result.

The comparison's headline claim — that the population test, not the distribution,
is what carries the method — is asserted here so it cannot quietly stop being
true.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import comparators, markers, positivity


def two_populations(n_neg=40, n_pos=15, neg=20.0, pos=120.0, sd=6.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(neg, sd, n_neg), rng.normal(pos, sd, n_pos)])


def one_population(n=50, mean=22.0, sd=5.0, seed=0):
    return np.abs(np.random.default_rng(seed).normal(mean, sd, n))


class TestGammaMixture:
    def test_threshold_lands_between_two_populations(self):
        values = two_populations(neg=20.0, pos=120.0)
        threshold = comparators.gamma_mixture_threshold(values)
        assert 20.0 < threshold < 120.0

    def test_recovers_the_fraction_on_clean_data(self):
        values = two_populations(n_neg=40, n_pos=15, neg=20.0, pos=120.0, sd=5.0)
        threshold = comparators.gamma_mixture_threshold(values)
        assert comparators.positive_fraction(values, threshold) == pytest.approx(
            100 * 15 / 55, abs=3.0
        )

    def test_is_deterministic(self):
        values = two_populations(seed=3)
        a = comparators.gamma_mixture_threshold(values)
        b = comparators.gamma_mixture_threshold(values)
        assert a == b

    def test_too_few_points_gives_no_threshold(self):
        assert not np.isfinite(comparators.gamma_mixture_threshold(np.array([1.0, 2.0])))


class TestKMeans:
    def test_threshold_lands_between_two_populations(self):
        values = two_populations(neg=15.0, pos=130.0)
        threshold = comparators.kmeans_threshold(values)
        assert 15.0 < threshold < 130.0

    def test_is_deterministic(self):
        values = two_populations(seed=5)
        assert comparators.kmeans_threshold(values) == comparators.kmeans_threshold(values)


class TestPositiveFraction:
    def test_counts_objects_at_or_above_threshold(self):
        values = np.array([1.0, 5.0, 10.0, 20.0])
        assert comparators.positive_fraction(values, 10.0) == pytest.approx(50.0)

    def test_infinite_threshold_gives_zero(self):
        assert comparators.positive_fraction(np.array([1.0, 2.0]), float("inf")) == 0.0


class TestTheHeadToHeadResult:
    """The benchmark's two conclusions, as assertions.

    1. On real positives, the distribution barely matters — every method is close.
    2. On one population, only the gated method abstains; the others invent a
       split. The ungated Gaussian proves it is the gate, not the distribution.
    """

    def test_all_methods_agree_on_a_real_positive_population(self):
        values = two_populations(n_neg=35, n_pos=20, neg=18.0, pos=125.0, sd=5.0)
        truth = 100 * 20 / 55

        gaussian = positivity.call_by_mixture(values).percent_positive
        gamma = comparators.positive_fraction(values, comparators.gamma_mixture_threshold(values))
        kmeans = comparators.positive_fraction(values, comparators.kmeans_threshold(values))
        otsu = comparators.positive_fraction(values, markers._exact_otsu(values))

        for value in (gaussian, gamma, kmeans, otsu):
            assert abs(value - truth) < 6.0, "methods should agree on a clear population"

    def test_only_the_gated_method_abstains_on_one_population(self):
        values = one_population(n=60, mean=22.0, sd=5.0)

        gated = positivity.call_by_mixture(values)
        assert not gated.bimodal
        assert gated.percent_positive == 0.0

        # Every ungated baseline splits the single population and reports a
        # spurious fraction.
        gamma = comparators.positive_fraction(values, comparators.gamma_mixture_threshold(values))
        kmeans = comparators.positive_fraction(values, comparators.kmeans_threshold(values))
        otsu = comparators.positive_fraction(values, markers._exact_otsu(values))
        assert gamma > 15.0
        assert kmeans > 15.0
        assert otsu > 15.0

    def test_the_gate_not_the_distribution_is_the_difference(self):
        """The same Gaussian mixture, without the gate, fails like the baselines."""
        values = one_population(n=60, mean=22.0, sd=5.0)

        gated = positivity.call_by_mixture(values).percent_positive
        fit = positivity.fit_two_component(values)
        ungated = comparators.positive_fraction(values, positivity._crossover(fit))

        assert gated == 0.0
        assert ungated >= 10.0, (
            "Gaussian-without-gate must invent a spurious positive fraction like "
            "the other ungated methods, isolating the gate as the contribution"
        )
