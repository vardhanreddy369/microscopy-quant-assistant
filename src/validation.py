"""Instance-segmentation scoring against ground-truth masks.

Counting accuracy alone is a weak claim: a method can get the count right by
splitting one nucleus and merging two others. These metrics compare objects to
objects, so a correct count built from wrong objects still scores badly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

# The standard instance-segmentation sweep: IoU 0.50 to 0.95 in steps of 0.05,
# ten thresholds, as used by the 2018 Data Science Bowl and COCO. Averaging over
# thresholds stops a method looking good purely because it is scored at a single
# lenient overlap. The stop value is 1.0 rather than 0.95 because np.arange
# excludes its endpoint, and stopping at 0.90 would quietly report a nine-
# threshold average while claiming to match the ten-threshold standard.
DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))

# Fraction of a ground-truth object a prediction must cover before it counts as
# having claimed part of it, when attributing split and merge errors.
FRAGMENT_FRACTION = 0.25


def decode_colored_mask(mask: np.ndarray) -> np.ndarray:
    """Decode a BBBC039-style colour-coded mask into per-object labels.

    The published masks are *not* labelled by object id. They are 4-coloured so
    that any two touching nuclei carry different colours; individual objects are
    recovered by connected components computed separately within each colour.
    Reading the colour channel as an object id instead collapses the dataset
    from roughly 23,000 nuclei to a few hundred.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]

    labels = np.zeros(mask.shape, dtype=np.int32)
    next_label = 0
    for color in np.unique(mask):
        if color == 0:
            continue
        component, count = ndi.label(mask == color)
        labels[component > 0] = component[component > 0] + next_label
        next_label += count

    return labels


def intersection_matrix(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pixel overlap between every predicted and every true object."""
    n_pred, n_true = int(predicted.max()), int(truth.max())
    if n_pred == 0 or n_true == 0:
        return np.zeros((n_pred, n_true), dtype=np.int64)

    paired = predicted.astype(np.int64).ravel() * (n_true + 1) + truth.astype(
        np.int64
    ).ravel()
    counts = np.bincount(paired, minlength=(n_pred + 1) * (n_true + 1))
    return counts.reshape(n_pred + 1, n_true + 1)[1:, 1:]


def iou_matrix(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Intersection-over-union between every predicted and every true object."""
    n_pred, n_true = int(predicted.max()), int(truth.max())
    if n_pred == 0 or n_true == 0:
        return np.zeros((n_pred, n_true), dtype=np.float64)

    intersection = intersection_matrix(predicted, truth)
    pred_area = np.bincount(predicted.ravel(), minlength=n_pred + 1)[1:][:, None]
    true_area = np.bincount(truth.ravel(), minlength=n_true + 1)[1:][None, :]
    union = pred_area + true_area - intersection
    return intersection / np.maximum(union, 1)


@dataclass
class ImageScore:
    """Scores for a single image."""

    name: str = ""
    n_true: int = 0
    n_predicted: int = 0
    per_threshold: dict[float, tuple[int, int, int]] = field(default_factory=dict)
    matched_ious: list[float] = field(default_factory=list)
    split_errors: int = 0
    merge_errors: int = 0

    @property
    def count_error(self) -> int:
        return self.n_predicted - self.n_true


def score_image(
    predicted: np.ndarray,
    truth: np.ndarray,
    thresholds=DEFAULT_THRESHOLDS,
    name: str = "",
) -> ImageScore:
    """Match predicted objects to true objects and tally errors.

    Matching maximises total IoU (an optimal assignment) rather than taking the
    best available greedily, so one prediction cannot claim several true objects.
    """
    n_pred, n_true = int(predicted.max()), int(truth.max())
    result = ImageScore(name=name, n_true=n_true, n_predicted=n_pred)

    if n_pred == 0 or n_true == 0:
        for threshold in thresholds:
            result.per_threshold[float(threshold)] = (0, n_pred, n_true)
        return result

    overlaps = iou_matrix(predicted, truth)
    rows, cols = linear_sum_assignment(-overlaps)
    pair_ious = overlaps[rows, cols]

    for threshold in thresholds:
        true_positives = int((pair_ious >= threshold).sum())
        result.per_threshold[float(threshold)] = (
            true_positives,
            n_pred - true_positives,
            n_true - true_positives,
        )

    result.matched_ious = [float(v) for v in pair_ious if v > 0]

    # Attribute *how* the segmentation went wrong, not just that it did.
    intersection = intersection_matrix(predicted, truth)
    true_area = np.bincount(truth.ravel(), minlength=n_true + 1)[1:]
    claimed = intersection >= (FRAGMENT_FRACTION * true_area[None, :])
    result.split_errors = int((claimed.sum(axis=0) >= 2).sum())
    result.merge_errors = int((claimed.sum(axis=1) >= 2).sum())

    return result


@dataclass
class DatasetScore:
    """Aggregated scores over a set of images."""

    images: list[ImageScore] = field(default_factory=list)
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def total_true(self) -> int:
        return sum(image.n_true for image in self.images)

    @property
    def total_predicted(self) -> int:
        return sum(image.n_predicted for image in self.images)

    def totals_at(self, threshold: float) -> tuple[int, int, int]:
        key = float(threshold)
        tp = fp = fn = 0
        for image in self.images:
            a, b, c = image.per_threshold[key]
            tp, fp, fn = tp + a, fp + b, fn + c
        return tp, fp, fn

    def f1_at(self, threshold: float) -> float:
        tp, fp, fn = self.totals_at(threshold)
        denominator = 2 * tp + fp + fn
        return (2 * tp / denominator) if denominator else 0.0

    def precision_at(self, threshold: float) -> float:
        tp, fp, _ = self.totals_at(threshold)
        return (tp / (tp + fp)) if (tp + fp) else 0.0

    def recall_at(self, threshold: float) -> float:
        tp, _, fn = self.totals_at(threshold)
        return (tp / (tp + fn)) if (tp + fn) else 0.0

    def average_precision(self) -> float:
        """Mean over thresholds of TP / (TP + FP + FN), the Data Science Bowl metric."""
        values = []
        for threshold in self.thresholds:
            tp, fp, fn = self.totals_at(threshold)
            total = tp + fp + fn
            values.append(tp / total if total else 0.0)
        return float(np.mean(values)) if values else 0.0

    @property
    def mean_matched_iou(self) -> float:
        every = [v for image in self.images for v in image.matched_ious]
        return float(np.mean(every)) if every else 0.0

    @property
    def split_errors(self) -> int:
        return sum(image.split_errors for image in self.images)

    @property
    def merge_errors(self) -> int:
        return sum(image.merge_errors for image in self.images)

    @property
    def count_mape(self) -> float:
        """Mean absolute percentage error of the per-image object count."""
        errors = [
            abs(image.count_error) / image.n_true
            for image in self.images
            if image.n_true > 0
        ]
        return float(np.mean(errors)) if errors else 0.0

    def summary(self) -> dict[str, float | int]:
        return {
            "images": self.n_images,
            "true_objects": self.total_true,
            "predicted_objects": self.total_predicted,
            "f1_at_50": self.f1_at(0.5),
            "precision_at_50": self.precision_at(0.5),
            "recall_at_50": self.recall_at(0.5),
            "f1_at_75": self.f1_at(0.75),
            "average_precision": self.average_precision(),
            "mean_matched_iou": self.mean_matched_iou,
            "split_errors": self.split_errors,
            "merge_errors": self.merge_errors,
            "count_mape": self.count_mape,
        }
