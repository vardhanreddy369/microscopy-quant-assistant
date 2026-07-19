"""Measurement correctness tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import measurements
from src.measurements import COLUMNS, circularity_from


def labelled_disk(size=200, centre=(100, 100), radius=30):
    yy, xx = np.mgrid[0:size, 0:size]
    mask = ((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) <= radius**2
    return mask.astype(np.int32)


class TestCircularity:
    def test_perfect_circle_is_near_one(self):
        labels = labelled_disk(radius=40)
        frame = measurements.measure(labels, labels.astype(np.float32))
        assert frame.loc[0, "circularity"] == pytest.approx(1.0, abs=0.06)

    def test_elongated_shape_scores_lower_than_a_circle(self):
        labels = np.zeros((200, 200), dtype=np.int32)
        labels[90:110, 40:160] = 1  # long thin bar
        bar = measurements.measure(labels, labels.astype(np.float32))
        disk = measurements.measure(
            labelled_disk(radius=40), labelled_disk(radius=40).astype(np.float32)
        )
        # Pull the scalars out through numpy rather than pandas .loc: a pandas
        # scalar is typed as a broad union that makes strict type checkers
        # explode a simple `<` into every possible type pairing. .to_numpy()
        # lands in concrete numeric territory.
        bar_circularity = bar["circularity"].to_numpy()[0]
        disk_circularity = disk["circularity"].to_numpy()[0]
        assert bar_circularity < disk_circularity

    def test_never_exceeds_one(self):
        # Discretisation can push the raw formula above 1 for tiny objects.
        values = circularity_from(np.array([4.0, 9.0, 1.0]), np.array([4.0, 6.0, 2.0]))
        assert np.nanmax(values) <= 1.0

    def test_zero_perimeter_is_nan_not_a_crash(self):
        assert np.isnan(circularity_from(np.array([5.0]), np.array([0.0]))[0])


class TestMeasurementFrame:
    def test_all_documented_columns_exist(self):
        labels = labelled_disk()
        frame = measurements.measure(labels, labels.astype(np.float32))
        for column in COLUMNS:
            assert column in frame.columns

    def test_one_row_per_object(self):
        labels = np.zeros((200, 200), dtype=np.int32)
        labels[20:50, 20:50] = 1
        labels[100:140, 100:140] = 2
        labels[150:180, 20:60] = 3
        frame = measurements.measure(labels, labels.astype(np.float32) / 3.0)
        assert len(frame) == 3
        assert sorted(frame["object_id"]) == [1, 2, 3]

    def test_area_matches_pixel_count(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:30, 10:40] = 1  # 20 * 30 = 600 pixels
        frame = measurements.measure(labels, labels.astype(np.float32))
        assert frame.loc[0, "area_pixels"] == 600

    def test_centroid_x_is_the_column_not_the_row(self):
        # A regression guard: regionprops reports (row, col), and swapping them
        # silently mirrors every coordinate in the exported CSV.
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:20, 70:80] = 1  # near the top-right: small y, large x
        frame = measurements.measure(labels, labels.astype(np.float32))
        assert frame.loc[0, "centroid_x"] == pytest.approx(74.5)
        assert frame.loc[0, "centroid_y"] == pytest.approx(14.5)

    def test_empty_labels_give_an_empty_frame_with_columns(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        frame = measurements.measure(labels, labels.astype(np.float32))
        assert frame.empty
        for column in COLUMNS:
            assert column in frame.columns

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            measurements.measure(
                np.zeros((10, 10), dtype=np.int32), np.zeros((20, 20), dtype=np.float32)
            )


class TestIntensity:
    def test_intensity_is_reported_on_the_0_255_scale(self):
        labels = labelled_disk()
        intensity = np.where(labels > 0, 0.5, 0.0).astype(np.float32)
        frame = measurements.measure(labels, intensity)
        assert frame.loc[0, "mean_intensity"] == pytest.approx(127.5, abs=0.5)

    def test_min_and_max_bracket_the_mean(self):
        labels = labelled_disk()
        rng = np.random.default_rng(0)
        intensity = (labels * rng.uniform(0.2, 0.9, labels.shape)).astype(np.float32)
        row = measurements.measure(labels, intensity).loc[0]
        assert row["minimum_intensity"] <= row["mean_intensity"] <= row["maximum_intensity"]


class TestBorderFlag:
    def test_object_touching_the_edge_is_flagged(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[0:20, 0:20] = 1  # hard against the top-left corner
        frame = measurements.measure(labels, labels.astype(np.float32))
        assert bool(frame.loc[0, "touches_border"]) is True

    def test_interior_object_is_not_flagged(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[40:60, 40:60] = 1
        frame = measurements.measure(labels, labels.astype(np.float32))
        assert bool(frame.loc[0, "touches_border"]) is False


class TestPixelScale:
    def test_no_scale_means_no_micrometre_columns(self):
        labels = labelled_disk()
        frame = measurements.measure(labels, labels.astype(np.float32))
        assert "area_um2" not in frame.columns

    def test_area_scales_with_the_square_of_pixel_size(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:30, 10:40] = 1  # 600 pixels
        frame = measurements.measure(labels, labels.astype(np.float32), pixel_size_um=0.5)
        assert frame.loc[0, "area_um2"] == pytest.approx(600 * 0.25)
        assert frame.loc[0, "centroid_x_um"] == pytest.approx(
            frame.loc[0, "centroid_x"] * 0.5
        )


class TestSummary:
    def test_summary_of_empty_frame_reports_zero_count(self):
        summary = measurements.summarize(measurements.measure(
            np.zeros((10, 10), dtype=np.int32), np.zeros((10, 10), dtype=np.float32)
        ))
        assert summary["count"] == 0

    def test_summary_counts_match_the_frame(self):
        labels = np.zeros((200, 200), dtype=np.int32)
        labels[20:50, 20:50] = 1
        labels[100:140, 100:140] = 2
        frame = measurements.measure(labels, labels.astype(np.float32) / 2)
        summary = measurements.summarize(frame)
        assert summary["count"] == len(frame) == 2
