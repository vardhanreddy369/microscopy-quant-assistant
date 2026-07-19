"""Tests for the reproducible marker-positivity module.

The module makes a statistical claim — that it can tell a real positive
population from one continuous population split in half — so it is tested in
both directions against cases with a known answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import measurements, preprocessing, positivity, segmentation
from src.config import DEFAULTS, SAMPLE_DIR


def two_populations(n_neg=40, n_pos=15, neg=20.0, pos=120.0, sd=6.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(neg, sd, n_neg),
        rng.normal(pos, sd, n_pos),
    ])


def one_population(n=50, mean=25.0, sd=6.0, seed=0):
    return np.random.default_rng(seed).normal(mean, sd, n)


class TestMixtureFitting:
    def test_two_component_fit_recovers_the_means(self):
        values = two_populations(neg=20.0, pos=120.0, sd=5.0)
        fit = positivity.fit_two_component(values)
        assert fit.n_components == 2
        assert fit.means[0] == pytest.approx(20.0, abs=4.0)
        assert fit.means[1] == pytest.approx(120.0, abs=4.0)

    def test_components_are_ordered_dim_then_bright(self):
        fit = positivity.fit_two_component(two_populations())
        assert fit.means[0] < fit.means[1], "index 1 must be the positive component"

    def test_fit_is_deterministic(self):
        """The same input must give an identical fit every time.

        Reproducibility is the whole point; a random-restart EM would not have
        this property.
        """
        values = two_populations(seed=3)
        a = positivity.fit_two_component(values)
        b = positivity.fit_two_component(values)
        assert np.array_equal(a.means, b.means)
        assert np.array_equal(a.sigmas, b.sigmas)

    def test_bic_prefers_two_components_for_two_populations(self):
        values = two_populations()
        one = positivity.fit_one_component(values)
        two = positivity.fit_two_component(values)
        assert two.bic < one.bic

    def test_bic_prefers_one_component_for_one_population(self):
        values = one_population()
        one = positivity.fit_one_component(values)
        two = positivity.fit_two_component(values)
        assert one.bic <= two.bic + positivity.STRONG_EVIDENCE


class TestBimodalityDetection:
    def test_two_populations_are_called_bimodal(self):
        result = positivity.call_by_mixture(two_populations())
        assert result.bimodal
        assert result.delta_bic > positivity.STRONG_EVIDENCE

    def test_one_population_is_not_called_bimodal(self):
        """The failure this exists to prevent: calling a split on one population."""
        result = positivity.call_by_mixture(one_population())
        assert not result.bimodal
        assert result.n_positive == 0
        assert result.percent_positive == 0.0

    def test_a_one_population_result_explains_itself(self):
        result = positivity.call_by_mixture(one_population())
        assert result.notes
        assert "no distinct positive population" in result.notes[0].lower()

    def test_separation_gate_rejects_close_components(self):
        # Two components that are statistically favoured but not far apart must
        # not be called two populations.
        values = two_populations(neg=40.0, pos=52.0, sd=6.0, n_neg=40, n_pos=40)
        result = positivity.call_by_mixture(values)
        two = positivity.fit_two_component(values)
        if positivity.ashman_d(two) < positivity.MIN_SEPARATION:
            assert not result.bimodal

    def test_too_few_objects_is_not_bimodal(self):
        assert not positivity.call_by_mixture(np.array([1.0, 2.0])).bimodal


class TestThresholdAndAssignment:
    def test_threshold_lands_between_the_populations(self):
        result = positivity.call_by_mixture(two_populations(neg=20.0, pos=120.0))
        assert 20.0 < result.threshold < 120.0

    def test_bright_objects_are_positive_and_dim_are_not(self):
        values = two_populations(n_neg=30, n_pos=20, neg=15.0, pos=130.0, sd=4.0)
        result = positivity.call_by_mixture(values)
        assert (values[values > 100] >= result.threshold).all()
        assert (values[values < 40] < result.threshold).all()

    def test_posteriors_are_probabilities(self):
        result = positivity.call_by_mixture(two_populations())
        assert result.posteriors.min() >= 0.0
        assert result.posteriors.max() <= 1.0

    def test_recovered_count_matches_the_construction(self):
        values = two_populations(n_neg=35, n_pos=15, neg=18.0, pos=125.0, sd=5.0)
        result = positivity.call_by_mixture(values)
        assert result.n_positive == pytest.approx(15, abs=1)


class TestNegativeControl:
    def test_threshold_comes_from_the_control(self):
        control = one_population(n=60, mean=20.0, sd=4.0, seed=1)
        sample = two_populations(n_neg=30, n_pos=20, neg=20.0, pos=120.0, seed=2)
        result = positivity.call_by_negative_control(sample, control, percentile=99.0)
        assert result.method == "negative_control"
        assert result.threshold == pytest.approx(np.percentile(control, 99.0))

    def test_almost_no_control_object_is_called_positive_against_itself(self):
        control = one_population(n=100, mean=20.0, sd=4.0, seed=5)
        result = positivity.call_by_negative_control(control, control, percentile=99.0)
        # By construction only the top ~1% of the control exceeds its own 99th
        # percentile.
        assert result.percent_positive <= 2.0

    def test_empty_control_is_rejected(self):
        with pytest.raises(ValueError):
            positivity.call_by_negative_control(np.array([1.0, 2.0]), np.array([]))


class TestReproducibilityReport:
    def test_report_records_the_essentials(self):
        result = positivity.call_by_mixture(two_populations())
        report = positivity.reproducibility_report(
            result, {"image": "demo.png", "marker channel": "green"}
        )
        assert "demo.png" in report
        assert "percent positive" in report.lower()
        assert "threshold" in report.lower()
        assert f"{result.percent_positive:.2f}" in report

    def test_report_is_identical_for_identical_input(self):
        values = two_populations(seed=7)
        context = {"image": "x.png"}
        a = positivity.reproducibility_report(positivity.call_by_mixture(values), context)
        b = positivity.reproducibility_report(positivity.call_by_mixture(values), context)
        assert a == b


class TestAgainstKnownFractions:
    """The bundled series has known positive fractions from 0% to 70%."""

    @staticmethod
    def truth():
        path = SAMPLE_DIR / "marker_ground_truth.json"
        if not path.exists():
            pytest.skip("run scripts/make_sample_data.py first")
        return json.loads(path.read_text())

    @staticmethod
    def marker_intensities(filename):
        nuclear = preprocessing.prepare(SAMPLE_DIR / filename, channel="blue")
        marker = preprocessing.prepare(SAMPLE_DIR / filename, channel="green")
        result = segmentation.segment(
            nuclear.analysis,
            threshold_method="otsu",
            min_size=DEFAULTS["min_size"],
            smoothing_sigma=DEFAULTS["smoothing_sigma"],
            cleanup_radius=DEFAULTS["cleanup_radius"],
            separate_touching=True,
            peak_min_distance=DEFAULTS["peak_min_distance"],
            seeding=DEFAULTS["seeding"],
            seed_depth=DEFAULTS["seed_depth"],
        )
        return measurements.measure(result.labels, marker.intensity)["mean_intensity"].to_numpy()

    @pytest.mark.parametrize(
        "filename",
        [
            "synthetic_marker_10pct.png",
            "synthetic_marker_30pct.png",
            "synthetic_marker_50pct.png",
            "synthetic_marker_70pct.png",
            "synthetic_marker_pair.png",
        ],
    )
    def test_recovers_the_true_fraction(self, filename):
        truth = self.truth()[filename]["percent_positive"]
        result = positivity.call_by_mixture(self.marker_intensities(filename))
        assert result.bimodal, f"{filename} should have a positive population"
        assert result.percent_positive == pytest.approx(truth, abs=2.0)

    def test_all_negative_image_reports_no_population(self):
        """The important direction: 0% true positive must not be called positive."""
        result = positivity.call_by_mixture(
            self.marker_intensities("synthetic_marker_00pct.png")
        )
        assert not result.bimodal
        assert result.percent_positive == 0.0

    def test_the_measurement_is_deterministic_end_to_end(self):
        values = self.marker_intensities("synthetic_marker_30pct.png")
        first = positivity.call_by_mixture(values)
        second = positivity.call_by_mixture(values)
        assert first.threshold == second.threshold
        assert first.n_positive == second.n_positive
