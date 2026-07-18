"""Per-object shape and intensity measurements.

All spatial measurements are in pixels unless a pixel size is supplied. No
pixel-to-micrometre conversion is guessed from the file: an unverified scale
produces confidently wrong numbers, which is worse than plain pixels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

_REGION_PROPERTIES = (
    "label",
    "centroid",
    "area",
    "perimeter",
    "perimeter_crofton",
    "equivalent_diameter_area",
    "eccentricity",
    "solidity",
    "axis_major_length",
    "axis_minor_length",
    "intensity_mean",
    "intensity_max",
    "intensity_min",
    "bbox",
)

COLUMNS = (
    "object_id",
    "centroid_x",
    "centroid_y",
    "area_pixels",
    "perimeter_pixels",
    "equivalent_diameter_pixels",
    "circularity",
    "eccentricity",
    "solidity",
    "major_axis_pixels",
    "minor_axis_pixels",
    "mean_intensity",
    "maximum_intensity",
    "minimum_intensity",
    "touches_border",
)

# Intensity is stored internally as a float in [0, 1]. Reporting on the familiar
# 8-bit scale keeps the CSV readable without implying more precision than the
# source image carries.
INTENSITY_SCALE = 255.0


def _empty_frame(pixel_size_um: float | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({name: pd.Series(dtype="float64") for name in COLUMNS})
    frame["object_id"] = frame["object_id"].astype("int64")
    frame["touches_border"] = frame["touches_border"].astype("bool")
    if pixel_size_um:
        for name in _SCALED_COLUMNS:
            frame[name] = pd.Series(dtype="float64")
    return frame


_SCALED_COLUMNS = (
    "area_um2",
    "perimeter_um",
    "equivalent_diameter_um",
    "major_axis_um",
    "minor_axis_um",
    "centroid_x_um",
    "centroid_y_um",
)


def circularity_from(area: np.ndarray, perimeter: np.ndarray) -> np.ndarray:
    """4*pi*area / perimeter**2, clipped to [0, 1].

    A perfect circle scores 1. Values above 1 are a pixel-discretisation
    artefact on very small objects rather than a real measurement, so they are
    clipped rather than reported.
    """
    area = np.asarray(area, dtype=np.float64)
    perimeter = np.asarray(perimeter, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        value = 4.0 * np.pi * area / np.square(perimeter)

    value = np.where(perimeter > 0, value, np.nan)
    return np.clip(value, 0.0, 1.0)


def measure(
    labels: np.ndarray,
    intensity_plane: np.ndarray,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """Measure every labelled object.

    ``intensity_plane`` should be the un-normalised plane from
    :class:`preprocessing.PreparedImage` so intensities stay comparable between
    images in a batch.
    """
    labels = np.asarray(labels)
    intensity_plane = np.asarray(intensity_plane, dtype=np.float32)

    if labels.shape != intensity_plane.shape:
        raise ValueError(
            f"labels {labels.shape} and intensity plane {intensity_plane.shape} "
            "must have the same shape"
        )

    if labels.max() == 0:
        return _empty_frame(pixel_size_um)

    table = regionprops_table(
        labels, intensity_image=intensity_plane, properties=_REGION_PROPERTIES
    )

    height, width = labels.shape
    # regionprops reports centroid as (row, column); x is the column.
    frame = pd.DataFrame(
        {
            "object_id": table["label"].astype("int64"),
            "centroid_x": table["centroid-1"],
            "centroid_y": table["centroid-0"],
            "area_pixels": table["area"],
            "perimeter_pixels": table["perimeter"],
            "equivalent_diameter_pixels": table["equivalent_diameter_area"],
            # Crofton perimeter is a less biased estimator on digitised
            # boundaries, so circularity uses it even though the plain perimeter
            # is the one reported.
            "circularity": circularity_from(
                table["area"], table["perimeter_crofton"]
            ),
            "eccentricity": table["eccentricity"],
            "solidity": table["solidity"],
            "major_axis_pixels": table["axis_major_length"],
            "minor_axis_pixels": table["axis_minor_length"],
            "mean_intensity": table["intensity_mean"] * INTENSITY_SCALE,
            "maximum_intensity": table["intensity_max"] * INTENSITY_SCALE,
            "minimum_intensity": table["intensity_min"] * INTENSITY_SCALE,
            # Objects clipped by the field of view have truncated area and
            # shape. They are flagged rather than dropped so the user decides.
            "touches_border": (
                (table["bbox-0"] <= 0)
                | (table["bbox-1"] <= 0)
                | (table["bbox-2"] >= height)
                | (table["bbox-3"] >= width)
            ),
        }
    )

    if pixel_size_um and pixel_size_um > 0:
        scale = float(pixel_size_um)
        frame["area_um2"] = frame["area_pixels"] * scale**2
        frame["perimeter_um"] = frame["perimeter_pixels"] * scale
        frame["equivalent_diameter_um"] = frame["equivalent_diameter_pixels"] * scale
        frame["major_axis_um"] = frame["major_axis_pixels"] * scale
        frame["minor_axis_um"] = frame["minor_axis_pixels"] * scale
        frame["centroid_x_um"] = frame["centroid_x"] * scale
        frame["centroid_y_um"] = frame["centroid_y"] * scale

    return frame.reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    """Headline numbers for the summary cards."""
    if frame.empty:
        return {
            "count": 0,
            "mean_area": float("nan"),
            "median_area": float("nan"),
            "mean_intensity": float("nan"),
            "mean_circularity": float("nan"),
            "border_objects": 0,
        }

    return {
        "count": int(len(frame)),
        "mean_area": float(frame["area_pixels"].mean()),
        "median_area": float(frame["area_pixels"].median()),
        "mean_intensity": float(frame["mean_intensity"].mean()),
        "mean_circularity": float(frame["circularity"].mean(skipna=True)),
        "border_objects": int(frame["touches_border"].sum()),
    }
