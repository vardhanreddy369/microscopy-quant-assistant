"""Tests for the scoring code itself.

A bug here would produce authoritative-looking but meaningless accuracy
numbers, so the metrics are checked against cases with hand-computable answers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation import (
    DEFAULT_THRESHOLDS,
    DatasetScore,
    decode_colored_mask,
    iou_matrix,
    score_image,
)


def boxes(shape, specs):
    """Label image from (label, row_slice, col_slice) specs."""
    out = np.zeros(shape, dtype=np.int32)
    for label, rows, cols in specs:
        out[rows, cols] = label
    return out


class TestColorMaskDecoding:
    def test_touching_objects_of_different_colours_stay_separate(self):
        # Two adjacent blocks sharing an edge, coloured 1 and 2. Connected
        # components on the raw colours would merge them; decoding must not.
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 2:10] = 1
        mask[5:15, 10:18] = 2
        assert decode_colored_mask(mask).max() == 2

    def test_same_colour_objects_apart_are_counted_separately(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:10, 2:8] = 1
        mask[5:10, 20:26] = 1  # same colour, not touching
        assert decode_colored_mask(mask).max() == 2

    def test_rgba_input_uses_the_first_channel(self):
        mask = np.zeros((20, 20, 4), dtype=np.uint8)
        mask[..., 3] = 255
        mask[5:15, 5:15, 0] = 3
        assert decode_colored_mask(mask).max() == 1

    def test_empty_mask_decodes_to_nothing(self):
        assert decode_colored_mask(np.zeros((10, 10), dtype=np.uint8)).max() == 0

    def test_labels_are_contiguous(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        mask[2:8, 12:18] = 2
        mask[20:26, 2:8] = 3
        labels = decode_colored_mask(mask)
        present = sorted(np.unique(labels[labels > 0]))
        assert present == [1, 2, 3]


class TestIouMatrix:
    def test_identical_objects_score_one(self):
        a = boxes((20, 20), [(1, slice(5, 15), slice(5, 15))])
        assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)

    def test_disjoint_objects_score_zero(self):
        a = boxes((20, 40), [(1, slice(5, 15), slice(0, 10))])
        b = boxes((20, 40), [(1, slice(5, 15), slice(20, 30))])
        assert iou_matrix(a, b)[0, 0] == 0.0

    def test_half_overlap_matches_hand_calculation(self):
        # 10x10 predicted vs 10x10 true, offset by 5 columns.
        # intersection 10*5=50, union 100+100-50=150 -> 1/3
        a = boxes((20, 30), [(1, slice(0, 10), slice(0, 10))])
        b = boxes((20, 30), [(1, slice(0, 10), slice(5, 15))])
        assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)

    def test_matrix_shape_is_predictions_by_truth(self):
        a = boxes((30, 30), [(1, slice(0, 5), slice(0, 5)),
                             (2, slice(10, 15), slice(10, 15))])
        b = boxes((30, 30), [(1, slice(0, 5), slice(0, 5))])
        assert iou_matrix(a, b).shape == (2, 1)

    def test_empty_inputs_give_an_empty_matrix(self):
        empty = np.zeros((10, 10), dtype=np.int32)
        a = boxes((10, 10), [(1, slice(2, 5), slice(2, 5))])
        assert iou_matrix(a, empty).size == 0
        assert iou_matrix(empty, a).size == 0


class TestScoring:
    @staticmethod
    def three_objects():
        return boxes((40, 60), [
            (1, slice(2, 12), slice(2, 12)),
            (2, slice(2, 12), slice(20, 30)),
            (3, slice(2, 12), slice(40, 50)),
        ])

    def test_perfect_prediction_scores_one(self):
        truth = self.three_objects()
        result = DatasetScore(images=[score_image(truth, truth)])
        assert result.f1_at(0.5) == pytest.approx(1.0)
        assert result.average_precision() == pytest.approx(1.0)
        assert result.mean_matched_iou == pytest.approx(1.0)

    def test_completely_disjoint_prediction_scores_zero(self):
        truth = self.three_objects()
        predicted = boxes((40, 60), [(1, slice(25, 35), slice(2, 12))])
        result = DatasetScore(images=[score_image(predicted, truth)])
        assert result.f1_at(0.5) == 0.0

    def test_missing_one_object_gives_a_false_negative(self):
        truth = self.three_objects()
        predicted = boxes((40, 60), [
            (1, slice(2, 12), slice(2, 12)),
            (2, slice(2, 12), slice(20, 30)),
        ])
        tp, fp, fn = DatasetScore(images=[score_image(predicted, truth)]).totals_at(0.5)
        assert (tp, fp, fn) == (2, 0, 1)

    def test_spurious_object_gives_a_false_positive(self):
        truth = self.three_objects()
        predicted = self.three_objects()
        predicted[30:35, 30:35] = 4
        tp, fp, fn = DatasetScore(images=[score_image(predicted, truth)]).totals_at(0.5)
        assert (tp, fp, fn) == (3, 1, 0)

    def test_one_prediction_cannot_match_two_true_objects(self):
        """Optimal assignment must not let a single blob claim both nuclei."""
        truth = boxes((20, 40), [(1, slice(5, 15), slice(2, 12)),
                                 (2, slice(5, 15), slice(14, 24))])
        predicted = boxes((20, 40), [(1, slice(5, 15), slice(2, 24))])
        tp, fp, fn = DatasetScore(images=[score_image(predicted, truth)]).totals_at(0.5)
        assert tp <= 1
        assert fn >= 1

    def test_threshold_is_applied_strictly(self):
        # IoU exactly 1/3, so it passes at 0.3 and fails at 0.5.
        truth = boxes((20, 30), [(1, slice(0, 10), slice(5, 15))])
        predicted = boxes((20, 30), [(1, slice(0, 10), slice(0, 10))])
        result = DatasetScore(
            images=[score_image(predicted, truth, thresholds=(0.3, 0.5))],
            thresholds=(0.3, 0.5),
        )
        assert result.totals_at(0.3)[0] == 1
        assert result.totals_at(0.5)[0] == 0

    def test_empty_prediction_counts_every_object_as_missed(self):
        truth = self.three_objects()
        empty = np.zeros_like(truth)
        tp, fp, fn = DatasetScore(images=[score_image(empty, truth)]).totals_at(0.5)
        assert (tp, fp, fn) == (0, 0, 3)


class TestErrorAttribution:
    def test_splitting_one_object_in_two_is_recorded_as_a_split(self):
        truth = boxes((30, 30), [(1, slice(5, 25), slice(5, 25))])
        predicted = boxes((30, 30), [(1, slice(5, 15), slice(5, 25)),
                                     (2, slice(15, 25), slice(5, 25))])
        result = score_image(predicted, truth)
        assert result.split_errors == 1
        assert result.merge_errors == 0

    def test_merging_two_objects_is_recorded_as_a_merge(self):
        truth = boxes((20, 40), [(1, slice(5, 15), slice(2, 12)),
                                 (2, slice(5, 15), slice(14, 24))])
        predicted = boxes((20, 40), [(1, slice(5, 15), slice(2, 24))])
        result = score_image(predicted, truth)
        assert result.merge_errors == 1
        assert result.split_errors == 0

    def test_a_correct_segmentation_records_neither(self):
        truth = boxes((20, 40), [(1, slice(5, 15), slice(2, 12)),
                                 (2, slice(5, 15), slice(14, 24))])
        result = score_image(truth, truth)
        assert result.split_errors == 0
        assert result.merge_errors == 0


class TestAggregation:
    def test_counts_sum_across_images(self):
        truth = boxes((20, 20), [(1, slice(2, 10), slice(2, 10))])
        result = DatasetScore(images=[score_image(truth, truth) for _ in range(4)])
        assert result.total_true == 4
        assert result.n_images == 4
        assert result.f1_at(0.5) == pytest.approx(1.0)

    def test_count_error_is_signed(self):
        truth = boxes((30, 30), [(1, slice(2, 10), slice(2, 10))])
        predicted = boxes((30, 30), [(1, slice(2, 10), slice(2, 10)),
                                     (2, slice(15, 23), slice(15, 23))])
        assert score_image(predicted, truth).count_error == 1
        assert score_image(truth, predicted).count_error == -1

    def test_average_precision_lies_between_zero_and_one(self):
        truth = boxes((40, 60), [(1, slice(2, 12), slice(2, 12)),
                                 (2, slice(2, 12), slice(20, 30))])
        predicted = boxes((40, 60), [(1, slice(3, 13), slice(3, 13))])
        value = DatasetScore(images=[score_image(predicted, truth)]).average_precision()
        assert 0.0 < value < 1.0


class TestThresholdSweep:
    """The sweep is a named external standard, so it must actually match it."""

    def test_uses_the_full_ten_threshold_standard(self):
        # IoU 0.50 to 0.95 in steps of 0.05. np.arange excludes its endpoint, so
        # a stop of 0.95 silently yields nine thresholds and an average that is
        # not comparable to published Data Science Bowl or COCO numbers.
        assert len(DEFAULT_THRESHOLDS) == 10
        assert float(DEFAULT_THRESHOLDS[0]) == pytest.approx(0.50)
        assert float(DEFAULT_THRESHOLDS[-1]) == pytest.approx(0.95)

    def test_thresholds_are_evenly_spaced(self):
        steps = np.diff([float(t) for t in DEFAULT_THRESHOLDS])
        assert np.allclose(steps, 0.05)

    def test_average_precision_spans_every_threshold(self):
        truth = boxes((40, 60), [(1, slice(2, 12), slice(2, 12))])
        result = DatasetScore(images=[score_image(truth, truth)])
        assert set(result.images[0].per_threshold) == {
            float(t) for t in DEFAULT_THRESHOLDS
        }
