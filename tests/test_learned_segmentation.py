"""Tests for the optional learned-model segmentation path.

Cellpose is an optional dependency, so everything that needs it skips cleanly
when it is absent. The availability guard itself is always tested, because the
application relies on it to decide whether to offer the mode at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import learned_segmentation
from src.segmentation import SegmentationResult

CELLPOSE = learned_segmentation.is_available()
needs_cellpose = pytest.mark.skipif(not CELLPOSE, reason="cellpose not installed")


def disks(size=200, centres=((60, 60), (60, 140), (140, 100)), radius=20):
    image = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for cy, cx in centres:
        image[((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2] = 1.0
    return image


class TestAvailability:
    """These run whether or not Cellpose is installed."""

    def test_availability_is_a_boolean(self):
        assert isinstance(learned_segmentation.is_available(), bool)

    def test_reason_is_empty_when_available(self):
        if learned_segmentation.is_available():
            assert learned_segmentation.unavailable_reason() == ""

    def test_reason_explains_how_to_install_when_missing(self):
        if not learned_segmentation.is_available():
            reason = learned_segmentation.unavailable_reason().lower()
            assert "cellpose" in reason
            assert "install" in reason

    def test_importing_the_module_does_not_require_cellpose(self):
        # The module must import cleanly in a base install, or the app would
        # fail to start wherever the optional dependency is absent.
        assert hasattr(learned_segmentation, "segment")


@needs_cellpose
class TestSegmentation:
    def test_returns_the_same_result_type_as_the_classical_path(self):
        result = learned_segmentation.segment(disks())
        assert isinstance(result, SegmentationResult)

    def test_finds_separated_objects(self):
        result = learned_segmentation.segment(disks())
        assert result.n_objects == 3

    def test_labels_are_sequential(self):
        result = learned_segmentation.segment(disks())
        present = sorted(np.unique(result.labels[result.labels > 0]))
        assert present == list(range(1, result.n_objects + 1))

    def test_mask_matches_the_labels(self):
        result = learned_segmentation.segment(disks())
        assert (result.mask == (result.labels > 0)).all()

    def test_method_is_recorded(self):
        assert learned_segmentation.segment(disks()).method == "cellpose"

    def test_threshold_is_not_a_number(self):
        """A learned model produces no intensity threshold, so it must not
        report one that downstream code might display as if meaningful."""
        assert np.isnan(learned_segmentation.segment(disks()).threshold)

    def test_minimum_size_is_honoured(self):
        """The size control must mean the same thing in both modes."""
        image = disks(centres=((60, 60), (60, 140), (140, 100)), radius=20)
        kept = learned_segmentation.segment(image, min_size=10)
        dropped = learned_segmentation.segment(image, min_size=100_000)
        assert kept.n_objects == 3
        assert dropped.n_objects == 0

    def test_rejects_a_non_two_dimensional_plane(self):
        with pytest.raises(ValueError):
            learned_segmentation.segment(np.zeros((10, 10, 3), dtype=np.float32))

    def test_blank_image_finds_nothing(self):
        result = learned_segmentation.segment(np.zeros((120, 120), dtype=np.float32))
        assert result.n_objects == 0
