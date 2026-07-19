"""Segmentation tests, including counting accuracy against known truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocessing, segmentation
from src.config import DEFAULTS, SAMPLE_DIR


def disk_image(size=200, centres=((100, 100),), radius=25, brightness=1.0):
    """Synthetic image with filled bright disks on a dark background."""
    image = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for cy, cx in centres:
        image[((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2] = brightness
    return image


def segment_with_defaults(plane, **overrides):
    params = {
        "threshold_method": "otsu",
        "min_size": DEFAULTS["min_size"],
        "smoothing_sigma": DEFAULTS["smoothing_sigma"],
        "peak_min_distance": DEFAULTS["peak_min_distance"],
        "separate_touching": DEFAULTS["separate_touching"],
    }
    params.update(overrides)
    return segmentation.segment(plane, **params)


class TestBasicCounting:
    def test_counts_separated_objects(self):
        plane = disk_image(centres=((50, 50), (50, 150), (150, 50), (150, 150)))
        assert segment_with_defaults(plane).n_objects == 4

    def test_blank_image_finds_nothing(self):
        result = segment_with_defaults(np.zeros((100, 100), dtype=np.float32))
        assert result.n_objects == 0
        assert result.empty
        assert not result.mask.any()

    def test_uniform_image_finds_nothing(self):
        # A flat non-zero image has no bimodal histogram; Otsu must not invent
        # a foreground covering the whole frame.
        result = segment_with_defaults(np.full((100, 100), 0.5, dtype=np.float32))
        assert result.n_objects == 0

    def test_min_size_removes_small_objects(self):
        plane = disk_image(centres=((50, 50), (150, 150)), radius=25)
        plane += disk_image(centres=((50, 150),), radius=3)
        big_only = segment_with_defaults(plane, min_size=500)
        assert big_only.n_objects == 2

    def test_min_size_boundary_is_inclusive(self):
        # An object of exactly min_size pixels must be kept, since the control is
        # labelled "minimum object size".
        plane = np.zeros((60, 60), dtype=np.float32)
        plane[10:20, 10:20] = 1.0  # exactly 100 pixels
        assert segment_with_defaults(plane, min_size=100, smoothing_sigma=0,
                                     cleanup_radius=0).n_objects == 1
        assert segment_with_defaults(plane, min_size=101, smoothing_sigma=0,
                                     cleanup_radius=0).n_objects == 0

    def test_labels_are_sequential(self):
        plane = disk_image(centres=((50, 50), (50, 150), (150, 100)))
        result = segment_with_defaults(plane)
        present = sorted(np.unique(result.labels[result.labels > 0]))
        assert present == list(range(1, result.n_objects + 1))


class TestWatershedSeparation:
    """The watershed step is the core claim: touching objects become separate."""

    @staticmethod
    def touching_pair(radius=25, gap_factor=0.85):
        offset = int(radius * 2 * gap_factor)
        centres = ((100, 100 - offset // 2), (100, 100 + offset // 2))
        return disk_image(size=200, centres=centres, radius=radius)

    def test_touching_objects_merge_without_watershed(self):
        plane = self.touching_pair()
        assert segment_with_defaults(plane, separate_touching=False).n_objects == 1

    def test_watershed_splits_touching_objects(self):
        plane = self.touching_pair()
        assert segment_with_defaults(plane, separate_touching=True).n_objects == 2

    def test_watershed_does_not_split_a_single_object(self):
        plane = disk_image(centres=((100, 100),), radius=30)
        assert segment_with_defaults(plane, separate_touching=True).n_objects == 1

    def test_watershed_preserves_total_foreground_area(self):
        # Splitting must partition the mask, not erode it.
        plane = self.touching_pair()
        merged = segment_with_defaults(plane, separate_touching=False)
        split = segment_with_defaults(plane, separate_touching=True)
        assert split.mask.sum() == pytest.approx(merged.mask.sum(), rel=0.02)


class TestThresholding:
    def test_manual_threshold_requires_a_value(self):
        with pytest.raises(ValueError):
            segmentation.compute_threshold(disk_image(), "manual", None)

    def test_manual_threshold_too_high_finds_nothing(self):
        plane = disk_image(brightness=0.5)
        result = segment_with_defaults(
            plane, threshold_method="manual", manual_threshold=0.99
        )
        assert result.n_objects == 0

    def test_manual_threshold_too_low_merges_everything(self):
        # A real image has a non-zero background. A threshold under the
        # background floor makes the entire frame one connected object.
        plane = disk_image(centres=((50, 50), (150, 150))) + 0.2
        result = segment_with_defaults(
            plane, threshold_method="manual", manual_threshold=0.05,
            separate_touching=False,
        )
        assert result.n_objects == 1
        assert result.mask.all()

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError):
            segmentation.compute_threshold(disk_image(), "not-a-method")

    @pytest.mark.parametrize("method", ["otsu", "li", "yen", "triangle"])
    def test_automatic_methods_find_the_objects(self, method):
        plane = disk_image(centres=((50, 50), (150, 150)))
        assert segment_with_defaults(plane, threshold_method=method).n_objects == 2


class TestGroundTruthAccuracy:
    """Counting accuracy on samples whose true object count is known."""

    @staticmethod
    def load_truth():
        path = SAMPLE_DIR / "ground_truth.json"
        if not path.exists():
            pytest.skip("run scripts/make_sample_data.py first")
        return json.loads(path.read_text())

    def count(self, filename):
        prepared = preprocessing.prepare(SAMPLE_DIR / filename)
        return segment_with_defaults(prepared.analysis).n_objects

    def test_easy_sample_is_exact(self):
        truth = self.load_truth()
        assert self.count("synthetic_easy.png") == truth["synthetic_easy.png"]

    def test_touching_sample_is_exact(self):
        # This is the case the watershed step exists for, so it must be exact,
        # not merely close.
        truth = self.load_truth()
        assert self.count("synthetic_moderate.png") == truth["synthetic_moderate.png"]

    def test_difficult_sample_is_a_known_failure(self):
        """The difficult sample is expected to under-count.

        Asserted as a range so the documented limitation stays honest: if a
        future change silently made this perfect, or much worse, the claim in
        the README would need updating too.
        """
        truth = self.load_truth()
        expected = truth["synthetic_difficult.png"]
        detected = self.count("synthetic_difficult.png")
        assert 0.4 * expected <= detected <= 0.9 * expected, (
            f"detected {detected} of {expected}; the README documents this as a "
            "partial-recall failure case"
        )


class TestIlluminationCorrectionInSegmentation:
    """Uneven illumination defeats a single global threshold.

    This is a distinct failure mode from touching objects, and unlike touching
    objects it *is* fixable. It shows up in two ways at once: dim objects on the
    dark side fall below the threshold, and the bright side of the background
    rises above it and fragments into spurious detections.
    """

    @staticmethod
    def unevenly_lit_disks(size=240, radius=16):
        """Three identical disks on a background that fades across the frame."""
        image = np.zeros((size, size), dtype=np.float32)
        yy, xx = np.mgrid[0:size, 0:size]
        image += 0.70 * (xx / size)
        for cx in (40, 120, 200):
            image[((yy - 120) ** 2 + (xx - cx) ** 2) <= radius**2] += 0.20
        return np.clip(image, 0, 1)

    def test_uncorrected_threshold_misses_the_dim_object(self):
        plane = self.unevenly_lit_disks()
        result = segment_with_defaults(plane, background_radius=0)
        assert result.labels[120, 40] == 0, (
            "expected the disk on the dark side to fall below a global threshold"
        )

    def test_uncorrected_threshold_also_swallows_the_background(self):
        """The bright end of the gradient rises above the global threshold.

        Measured as the foreground fraction rather than an object count: what
        goes wrong is that half the image becomes "signal", and whether that
        lands as one blob or twenty is an accident of the seeding.
        """
        plane = self.unevenly_lit_disks()
        result = segment_with_defaults(plane, background_radius=0)
        foreground_fraction = result.mask.sum() / plane.size
        assert foreground_fraction > 0.30, (
            "expected the lit half of the background to be taken as foreground"
        )

    def test_correction_recovers_the_dim_object(self):
        plane = self.unevenly_lit_disks()
        result = segment_with_defaults(plane, background_radius=30)
        assert result.labels[120, 40] > 0

    def test_correction_finds_exactly_the_three_real_objects(self):
        plane = self.unevenly_lit_disks()
        assert segment_with_defaults(plane, background_radius=30).n_objects == 3

    def test_radius_smaller_than_objects_corrupts_their_area_silently(self):
        """A too-small radius keeps the count but hollows the objects out.

        A top-hat removes structure larger than its radius, so a radius under
        the object size strips each object's flat interior and leaves only a
        rim. The count still comes out right, which is exactly what makes it
        dangerous: the areas are wrong by a factor of five and nothing in the
        output signals it. Hence the control's help text saying to set the
        radius larger than the objects.
        """
        plane = self.unevenly_lit_disks()
        good = segment_with_defaults(plane, background_radius=30)
        bad = segment_with_defaults(plane, background_radius=3)

        assert good.n_objects == 3
        assert bad.mask.sum() < 0.5 * good.mask.sum(), (
            "expected a too-small radius to erode the objects"
        )

    def test_default_leaves_segmentation_unchanged(self):
        plane = disk_image(centres=((50, 50), (150, 150)))
        assert (
            segment_with_defaults(plane).n_objects
            == segment_with_defaults(plane, background_radius=0).n_objects
        )
