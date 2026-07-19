"""Tests for two-channel marker-positive quantification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocessing, segmentation
from src.config import DEFAULTS, SAMPLE_DIR
from src.markers import _exact_otsu, choose_threshold, measure_marker, separation


def labelled_row(count=6, size=120, radius=8):
    """A row of separate round objects."""
    labels = np.zeros((size, size), dtype=np.int32)
    yy, xx = np.mgrid[0:size, 0:size]
    for index in range(count):
        cx = int((index + 0.5) * size / count)
        labels[((yy - size // 2) ** 2 + (xx - cx) ** 2) <= radius**2] = index + 1
    return labels


def marker_plane_for(labels, positive_ids, high=0.8, low=0.05):
    """Marker channel where the listed object ids are bright."""
    plane = np.full(labels.shape, 0.01, dtype=np.float32)
    for object_id in np.unique(labels[labels > 0]):
        value = high if object_id in positive_ids else low
        plane[labels == object_id] = value
    return plane


class TestExactOtsu:
    """The threshold search must not inherit an image histogram's binning."""

    def test_splits_a_clean_bimodal_set_in_the_gap(self):
        values = np.array([15.0, 16.0, 17.0, 18.0, 104.0, 110.0, 120.0])
        threshold = _exact_otsu(values)
        assert 18.0 < threshold < 104.0

    def test_beats_histogram_binning_on_few_objects(self):
        """Regression guard for a real defect.

        skimage.threshold_otsu bins into 256 bins, which is right for an image
        and wrong for a few dozen per-object means. On this distribution it
        returned 27.9 — inside the negative cluster — and misclassified two
        negatives as positive. The exact search must place the threshold in the
        gap between 28.1 and 104.6.
        """
        negatives = np.linspace(15.8, 28.1, 31)
        positives = np.linspace(104.6, 137.0, 13)
        values = np.concatenate([negatives, positives])

        threshold = _exact_otsu(values)
        assert 28.1 < threshold < 104.6
        assert int((values >= threshold).sum()) == 13
        assert int((values < threshold).sum()) == 31

    def test_single_value_is_not_split(self):
        assert not np.isfinite(_exact_otsu(np.array([5.0])))

    def test_identical_values_are_not_split(self):
        assert not np.isfinite(choose_threshold(np.full(10, 7.0)))


class TestThresholdSelection:
    def test_manual_requires_a_value(self):
        with pytest.raises(ValueError):
            choose_threshold(np.array([1.0, 2.0, 3.0]), method="manual")

    def test_manual_returns_what_was_asked_for(self):
        assert choose_threshold(np.array([1.0, 9.0]), method="manual",
                                manual_value=4.5) == pytest.approx(4.5)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError):
            choose_threshold(np.array([1.0, 2.0]), method="guess")

    def test_too_few_objects_gives_no_split(self):
        assert not np.isfinite(choose_threshold(np.array([1.0, 2.0])))


class TestMeasureMarker:
    def test_counts_the_positive_objects(self):
        labels = labelled_row(count=6)
        plane = marker_plane_for(labels, positive_ids={1, 3, 5})
        result = measure_marker(labels, plane)
        assert result.total == 6
        assert result.positive == 3
        assert result.percent == pytest.approx(50.0)

    def test_identifies_which_objects_are_positive(self):
        labels = labelled_row(count=4)
        plane = marker_plane_for(labels, positive_ids={2, 4})
        frame = measure_marker(labels, plane).frame
        positive_ids = set(frame.loc[frame["marker_positive"], "object_id"])
        assert positive_ids == {2, 4}

    def test_segmentation_channel_defines_the_denominator(self):
        """Every segmented object is counted, including the negative ones.

        Segmenting on the marker channel instead would find only the positive
        cells and make the percentage meaningless.
        """
        labels = labelled_row(count=8)
        plane = marker_plane_for(labels, positive_ids={1})
        result = measure_marker(labels, plane)
        assert result.total == 8
        assert result.positive == 1
        assert result.percent == pytest.approx(12.5)

    def test_manual_threshold_changes_the_call(self):
        labels = labelled_row(count=4)
        plane = marker_plane_for(labels, positive_ids={1, 2}, high=0.8, low=0.4)
        # 0.4 -> 102 and 0.8 -> 204 on the 0-255 scale.
        below = measure_marker(labels, plane, method="manual", manual_threshold=50.0)
        between = measure_marker(labels, plane, method="manual", manual_threshold=150.0)
        assert below.positive == 4
        assert between.positive == 2

    def test_no_objects_gives_an_empty_result(self):
        labels = np.zeros((40, 40), dtype=np.int32)
        result = measure_marker(labels, np.zeros((40, 40), dtype=np.float32))
        assert result.total == 0
        assert result.positive == 0
        assert np.isnan(result.fraction)

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            measure_marker(np.zeros((10, 10), dtype=np.int32),
                           np.zeros((20, 20), dtype=np.float32))

    def test_intensity_is_reported_on_the_0_255_scale(self):
        labels = labelled_row(count=3)
        plane = marker_plane_for(labels, positive_ids={1}, high=0.5, low=0.1)
        frame = measure_marker(labels, plane).frame
        brightest = frame["marker_mean"].max()
        assert brightest == pytest.approx(127.5, abs=1.0)


class TestSeparation:
    def test_well_separated_populations_score_high(self):
        # The measure is a gap fraction in [0, 1], so a clean split approaches 1.
        labels = labelled_row(count=6)
        plane = marker_plane_for(labels, positive_ids={1, 2, 3}, high=0.9, low=0.05)
        value = separation(measure_marker(labels, plane).frame)
        assert 0.5 < value <= 1.0

    def test_a_genuine_split_separates_better_than_a_spurious_one(self):
        """Two real populations must score above one population cut in half.

        Asserted as a comparison rather than against a fixed cut-off: with only
        a few dozen objects, a single population can show a sizeable gap purely
        from sparse sampling, so no absolute threshold would be defensible.
        """
        labels = labelled_row(count=8)
        genuine = measure_marker(
            labels, marker_plane_for(labels, positive_ids={1, 2, 3},
                                     high=0.9, low=0.05)
        )

        rng = np.random.default_rng(0)
        single = np.full(labels.shape, 0.01, dtype=np.float32)
        for object_id in np.unique(labels[labels > 0]):
            single[labels == object_id] = rng.uniform(0.30, 0.34)
        spurious = measure_marker(labels, single)

        assert spurious.positive > 0  # an automatic split is still produced
        assert separation(genuine.frame) > separation(spurious.frame)


class TestAgainstKnownTruth:
    """The bundled two-channel sample has a known positive fraction."""

    @staticmethod
    def load():
        path = SAMPLE_DIR / "marker_ground_truth.json"
        if not path.exists():
            pytest.skip("run scripts/make_sample_data.py first")
        return json.loads(path.read_text())["synthetic_marker_pair.png"]

    @staticmethod
    def run_pipeline():
        nuclear = preprocessing.prepare(
            SAMPLE_DIR / "synthetic_marker_pair.png", channel="blue"
        )
        marker = preprocessing.prepare(
            SAMPLE_DIR / "synthetic_marker_pair.png", channel="green"
        )
        result = segmentation.segment(
            nuclear.analysis,
            threshold_method="otsu",
            min_size=DEFAULTS["min_size"],
            smoothing_sigma=DEFAULTS["smoothing_sigma"],
            cleanup_radius=DEFAULTS["cleanup_radius"],
            separate_touching=True,
            peak_min_distance=DEFAULTS["peak_min_distance"],
        )
        return result, measure_marker(result.labels, marker.intensity)

    def test_every_nucleus_is_found(self):
        truth = self.load()
        result, _ = self.run_pipeline()
        assert result.n_objects == truth["total_nuclei"]

    def test_the_positive_count_is_exact(self):
        truth = self.load()
        _, marker = self.run_pipeline()
        assert marker.positive == truth["marker_positive"]

    def test_the_percentage_is_within_one_point(self):
        truth = self.load()
        _, marker = self.run_pipeline()
        assert marker.percent == pytest.approx(truth["percent_positive"], abs=1.0)
