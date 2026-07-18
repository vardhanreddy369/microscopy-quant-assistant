"""Classical segmentation of bright objects on a dark background.

This is deliberately *not* machine learning. The pipeline is threshold ->
morphological cleanup -> distance transform -> watershed, which is the standard
approach for separating touching convex objects such as nuclei.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_li, threshold_otsu, threshold_triangle, threshold_yen
from skimage.morphology import closing, disk, opening, remove_small_objects
from skimage.segmentation import relabel_sequential, watershed

from .preprocessing import smooth

THRESHOLD_METHODS = ("otsu", "li", "yen", "triangle", "manual")


def _drop_small(array: np.ndarray, min_size: int) -> np.ndarray:
    """Remove objects with an area below ``min_size``.

    scikit-image 0.26 replaced ``min_size`` (drops area < value) with
    ``max_size`` (drops area <= value), so the threshold shifts by one to keep
    "minimum size" meaning what the user was told it means.
    """
    if not min_size or min_size <= 0:
        return array
    with warnings.catch_warnings():
        # scikit-image warns when a label image happens to contain exactly one
        # object, in case a boolean mask was intended. Here both are valid input.
        warnings.filterwarnings(
            "ignore", message="Only one label was provided", category=UserWarning
        )
        return remove_small_objects(array, max_size=int(min_size) - 1)

_AUTO_THRESHOLDS = {
    "otsu": threshold_otsu,
    "li": threshold_li,
    "yen": threshold_yen,
    "triangle": threshold_triangle,
}


@dataclass
class SegmentationResult:
    """Outcome of one segmentation run."""

    labels: np.ndarray
    mask: np.ndarray
    threshold: float
    n_objects: int
    method: str
    separated: bool

    @property
    def empty(self) -> bool:
        return self.n_objects == 0


def compute_threshold(
    plane: np.ndarray, method: str = "otsu", manual_value: float | None = None
) -> float:
    """Return the intensity cut-off separating foreground from background."""
    if method not in THRESHOLD_METHODS:
        raise ValueError(f"method must be one of {THRESHOLD_METHODS}, got {method!r}")

    if method == "manual":
        if manual_value is None:
            raise ValueError("manual thresholding requires manual_value")
        return float(manual_value)

    finite = plane[np.isfinite(plane)]
    # A flat image has no bimodal histogram; every automatic method either
    # raises or returns something meaningless, so short-circuit.
    if finite.size == 0 or float(finite.max()) <= float(finite.min()):
        return float("inf")

    try:
        return float(_AUTO_THRESHOLDS[method](finite))
    except (ValueError, RuntimeError):
        return float("inf")


def build_mask(
    plane: np.ndarray,
    threshold: float,
    min_size: int = 50,
    cleanup_radius: int = 1,
    fill_holes: bool = True,
) -> np.ndarray:
    """Threshold and clean up a binary foreground mask."""
    mask = plane > threshold

    if cleanup_radius and cleanup_radius > 0:
        footprint = disk(int(cleanup_radius))
        # Opening removes speckle, closing seals the pinholes that opening and
        # noise leave behind inside otherwise solid objects.
        mask = opening(mask, footprint)
        mask = closing(mask, footprint)

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    return _drop_small(mask, min_size)


def separate_touching_objects(
    mask: np.ndarray, peak_min_distance: int = 7, distance_smoothing: float = 1.0
) -> np.ndarray:
    """Split touching objects with a distance-transform watershed.

    Each object's centre is the point furthest from any background pixel, so
    peaks in the distance transform act as one seed per object and the watershed
    grows those seeds back out to the mask boundary.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(mask)

    # Smoothing the distance map suppresses the multiple near-equal peaks a
    # ragged boundary produces, which would otherwise shatter one object into
    # several fragments.
    if distance_smoothing and distance_smoothing > 0:
        distance = smooth(distance, distance_smoothing)

    coords = peak_local_max(
        distance,
        min_distance=max(1, int(peak_min_distance)),
        labels=mask,
        exclude_border=False,
    )

    if coords.size == 0:
        return ndi.label(mask)[0].astype(np.int32)

    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = ndi.label(markers)

    labels = watershed(-distance, markers, mask=mask)
    return labels.astype(np.int32)


def segment(
    plane: np.ndarray,
    threshold_method: str = "otsu",
    manual_threshold: float | None = None,
    min_size: int = 50,
    smoothing_sigma: float = 1.0,
    cleanup_radius: int = 1,
    fill_holes: bool = True,
    separate_touching: bool = True,
    peak_min_distance: int = 7,
) -> SegmentationResult:
    """Run the full segmentation pipeline on a prepared analysis plane.

    ``plane`` must be the 2-D float image produced by :func:`preprocessing.prepare`.
    """
    plane = np.asarray(plane, dtype=np.float32)
    if plane.ndim != 2:
        raise ValueError(f"expected a 2-D analysis plane, got shape {plane.shape}")

    blurred = smooth(plane, smoothing_sigma)
    threshold = compute_threshold(blurred, threshold_method, manual_threshold)

    if not np.isfinite(threshold):
        empty = np.zeros(plane.shape, dtype=np.int32)
        return SegmentationResult(
            labels=empty,
            mask=empty.astype(bool),
            threshold=float("nan"),
            n_objects=0,
            method=threshold_method,
            separated=False,
        )

    mask = build_mask(blurred, threshold, min_size, cleanup_radius, fill_holes)

    if not mask.any():
        empty = np.zeros(plane.shape, dtype=np.int32)
        return SegmentationResult(
            labels=empty,
            mask=empty.astype(bool),
            threshold=threshold,
            n_objects=0,
            method=threshold_method,
            separated=separate_touching,
        )

    if separate_touching:
        labels = separate_touching_objects(mask, peak_min_distance)
    else:
        labels = ndi.label(mask)[0].astype(np.int32)

    # Watershed can carve off slivers that are below the size floor the user
    # asked for, so the filter is applied again after splitting.
    labels = _drop_small(labels, min_size)

    labels = relabel_sequential(labels)[0].astype(np.int32)
    n_objects = int(labels.max())

    return SegmentationResult(
        labels=labels,
        mask=labels > 0,
        threshold=float(threshold),
        n_objects=n_objects,
        method=threshold_method,
        separated=separate_touching,
    )
