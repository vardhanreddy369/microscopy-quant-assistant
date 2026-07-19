"""Two-channel marker-positive quantification.

Counting objects is rarely the endpoint of a fluorescence experiment. The
reported result is usually a *fraction*: percent TUNEL-positive nuclei, percent
cleaved-caspase-3-positive, percent Ki67-positive. That is a two-channel
measurement:

    segment nuclei in the nuclear channel (DAPI/Hoechst)
        -> measure the marker channel inside each nucleus mask
        -> decide positive or negative per object
        -> report the percentage

The nuclear channel defines *where the cells are*; the marker channel decides
*which of them count*. Segmenting on the marker channel instead would measure
only the cells that are already positive and make the denominator meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from . import positivity
from .measurements import INTENSITY_SCALE

# "mixture" is the recommended method: a two-component model that also reports
# whether a positive population exists at all (see src/positivity.py). "otsu" is
# the exact per-object split, kept as a simpler alternative; "manual" takes the
# threshold from the researcher, ideally a negative control.
THRESHOLD_METHODS = ("mixture", "otsu", "manual")


def _exact_otsu(values: np.ndarray) -> float:
    """Exhaustive 1-D Otsu over sorted values.

    ``skimage.filters.threshold_otsu`` bins its input into a 256-bin histogram.
    That is right for an image of millions of pixels and wrong for a few dozen
    per-object means: on a cleanly bimodal set of 44 objects, with a gap between
    28 and 105, it returned 27.9 — inside the negative cluster — and
    misclassified two negatives as positive.

    Searching every split point of the sorted values is exact, has no binning
    artefact, and at this size costs nothing.
    """
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = ordered.size
    if n < 2:
        return float("inf")

    best_threshold, best_variance = float("inf"), -1.0
    for index in range(1, n):
        if ordered[index] == ordered[index - 1]:
            continue
        weight_low, weight_high = index / n, (n - index) / n
        gap = ordered[:index].mean() - ordered[index:].mean()
        variance = weight_low * weight_high * gap * gap
        if variance > best_variance:
            best_variance = variance
            best_threshold = 0.5 * (ordered[index - 1] + ordered[index])
    return float(best_threshold)


def separation(frame: pd.DataFrame) -> float:
    """How cleanly the positive and negative populations separate, in [0, 1].

    Measured as the empty gap between the dimmest positive and the brightest
    negative, as a fraction of the full intensity range.

    A standardised mean difference such as Cohen's d is the wrong tool here: it
    is scale-invariant, so splitting one tight cluster in half yields a *large*
    d precisely in the case the diagnostic is meant to catch. The gap is what
    distinguishes two populations from one, so the gap is what is measured.

    Treat this as informative, not decisive. With a few dozen objects a single
    population can show a respectable gap purely from sparse sampling, so it
    cannot carry a fixed pass/fail cut-off. Read it alongside the marker
    intensity histogram.
    """
    positives = frame.loc[frame["marker_positive"], "marker_mean"].to_numpy()
    negatives = frame.loc[~frame["marker_positive"], "marker_mean"].to_numpy()
    if positives.size == 0 or negatives.size == 0:
        return float("nan")

    full_range = float(frame["marker_mean"].max() - frame["marker_mean"].min())
    if full_range <= 0:
        return 0.0
    gap = float(positives.min() - negatives.max())
    return max(0.0, gap / full_range)


@dataclass
class MarkerResult:
    """Outcome of scoring one marker channel against a set of nuclei."""

    frame: pd.DataFrame
    threshold: float
    method: str
    # Populated by the mixture method: whether a distinct positive population
    # was found at all, the strength of that evidence, and any explanation.
    bimodal: bool = True
    delta_bic: float = float("nan")
    notes: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return int(len(self.frame))

    @property
    def positive(self) -> int:
        return int(self.frame["marker_positive"].sum()) if self.total else 0

    @property
    def negative(self) -> int:
        return self.total - self.positive

    @property
    def fraction(self) -> float:
        """Positive objects as a fraction of all objects."""
        return (self.positive / self.total) if self.total else float("nan")

    @property
    def percent(self) -> float:
        return self.fraction * 100.0


def _object_means(labels: np.ndarray, marker_plane: np.ndarray) -> pd.DataFrame:
    table = regionprops_table(
        labels, intensity_image=marker_plane,
        properties=("label", "intensity_mean", "intensity_max"),
    )
    return pd.DataFrame(
        {
            "object_id": table["label"].astype("int64"),
            "marker_mean": table["intensity_mean"] * INTENSITY_SCALE,
            "marker_max": table["intensity_max"] * INTENSITY_SCALE,
        }
    )


def choose_threshold(
    object_means: np.ndarray,
    method: str = "otsu",
    manual_value: float | None = None,
) -> float:
    """Pick the intensity above which an object counts as marker-positive.

    ``otsu``
        Split the per-object mean intensities into two groups. Appropriate when
        the field genuinely contains both populations. Check the reported
        separation: an automatic split is meaningless when every cell is in
        fact negative.
    ``manual``
        Whatever value the researcher sets. This is the rigorous option, because
        the defensible way to fix a positivity threshold is from a negative
        control stained and imaged alongside the sample, not from the sample
        itself.
    """
    if method not in THRESHOLD_METHODS:
        raise ValueError(f"method must be one of {THRESHOLD_METHODS}, got {method!r}")

    if method == "manual":
        if manual_value is None:
            raise ValueError("manual thresholding requires manual_value")
        return float(manual_value)

    values = np.asarray(object_means, dtype=np.float64)
    if values.size < 3 or float(values.max()) <= float(values.min()):
        return float("inf")
    return _exact_otsu(values)


def measure_marker(
    labels: np.ndarray,
    marker_plane: np.ndarray,
    method: str = "otsu",
    manual_threshold: float | None = None,
) -> MarkerResult:
    """Score every segmented object against a marker channel.

    ``labels`` come from segmenting the nuclear channel. ``marker_plane`` is the
    un-normalised plane of the marker channel, so intensities stay comparable
    between images in a batch.
    """
    labels = np.asarray(labels)
    marker_plane = np.asarray(marker_plane, dtype=np.float32)

    if labels.shape != marker_plane.shape:
        raise ValueError(
            f"labels {labels.shape} and marker plane {marker_plane.shape} "
            "must have the same shape"
        )

    if labels.max() == 0:
        empty = pd.DataFrame(
            {
                "object_id": pd.Series(dtype="int64"),
                "marker_mean": pd.Series(dtype="float64"),
                "marker_max": pd.Series(dtype="float64"),
                "marker_positive": pd.Series(dtype="bool"),
            }
        )
        return MarkerResult(frame=empty, threshold=float("nan"), method=method)

    frame = _object_means(labels, marker_plane)

    if method == "mixture":
        # The mixture model both sets the threshold and decides whether a
        # positive population exists. When it does not, every object is negative.
        result = positivity.call_by_mixture(frame["marker_mean"].to_numpy())
        frame["marker_positive"] = result.positive
        return MarkerResult(
            frame=frame,
            threshold=result.threshold,
            method="mixture",
            bimodal=result.bimodal,
            delta_bic=result.delta_bic,
            notes=tuple(result.notes),
        )

    threshold = choose_threshold(
        frame["marker_mean"].to_numpy(),
        method=method,
        manual_value=manual_threshold,
    )
    frame["marker_positive"] = frame["marker_mean"] >= threshold
    return MarkerResult(frame=frame, threshold=threshold, method=method)


def summarize(result: MarkerResult) -> dict[str, float | int]:
    """Headline numbers for a marker experiment."""
    if not result.total:
        return {
            "total": 0, "positive": 0, "negative": 0,
            "percent_positive": float("nan"), "threshold": float("nan"),
            "mean_positive_intensity": float("nan"),
            "mean_negative_intensity": float("nan"),
            "separation": float("nan"),
            "bimodal": result.bimodal, "delta_bic": result.delta_bic,
        }

    frame = result.frame
    positives = frame.loc[frame["marker_positive"], "marker_mean"]
    negatives = frame.loc[~frame["marker_positive"], "marker_mean"]
    return {
        "total": result.total,
        "positive": result.positive,
        "negative": result.negative,
        "percent_positive": result.percent,
        "threshold": result.threshold,
        "mean_positive_intensity": float(positives.mean()) if len(positives) else float("nan"),
        "mean_negative_intensity": float(negatives.mean()) if len(negatives) else float("nan"),
        "separation": separation(frame),
        "bimodal": result.bimodal,
        "delta_bic": result.delta_bic,
    }
