"""Annotated overlays, mask renderings, and distribution charts."""

from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")  # No GUI backend: figures are rendered to buffers only.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

BOUNDARY_COLOR = (255, 235, 59)  # amber, readable over both dark and bright signal
LABEL_COLOR = (255, 255, 255)
LABEL_OUTLINE = (0, 0, 0)


def to_display_rgb(plane: np.ndarray) -> np.ndarray:
    """Convert any analysis plane or loaded image to 8-bit RGB for display."""
    array = np.asarray(plane)

    if array.ndim == 3 and array.shape[-1] >= 3:
        rgb = array[..., :3]
        if rgb.dtype == np.uint8:
            return np.ascontiguousarray(rgb)
        rgb = rgb.astype(np.float32)
        peak = float(rgb.max()) if rgb.size else 0.0
        if peak > 1.0:
            rgb = rgb / peak
        return np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

    if array.ndim == 3:
        array = array[..., 0]

    array = array.astype(np.float32)
    peak = float(array.max()) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    gray = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(np.dstack([gray, gray, gray]))


# Above this count, ID digits are smaller than the objects they label and turn
# the overlay into noise. The table still carries every ID.
MAX_OBJECTS_FOR_IDS = 100
MIN_DIAMETER_FOR_IDS = 12.0


def _median_diameter(labels: np.ndarray) -> float:
    values = labels[labels > 0]
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    return float(np.sqrt(np.median(counts) * 4.0 / np.pi))


def _auto_font_scale(median_diameter: float) -> float:
    """Size ID text relative to the objects, so labels stay legible but fit.

    Scaling by image width alone puts huge numbers on a sparse image and
    illegible ones on a dense field; median object size is the better guide.
    """
    if median_diameter <= 0:
        return 0.4
    return float(np.clip(median_diameter / 30.0, 0.32, 0.90))


def should_show_ids(labels: np.ndarray) -> bool:
    """Whether ID numbers will be readable on this particular result."""
    n_objects = int(np.asarray(labels).max())
    if n_objects == 0 or n_objects > MAX_OBJECTS_FOR_IDS:
        return False
    return _median_diameter(labels) >= MIN_DIAMETER_FOR_IDS


def annotate(
    base: np.ndarray,
    labels: np.ndarray,
    show_ids: bool | str = "auto",
    show_centroids: bool = False,
    boundary_color: tuple[int, int, int] = BOUNDARY_COLOR,
) -> np.ndarray:
    """Draw object boundaries and ID numbers over an image.

    ``show_ids="auto"`` suppresses the numbers on dense results, where they
    would overlap into an unreadable smear rather than identify anything.
    """
    canvas = to_display_rgb(base).copy()
    labels = np.asarray(labels)

    if labels.max() == 0:
        return canvas

    boundaries = find_boundaries(labels, mode="outer")
    canvas[boundaries] = boundary_color

    if show_ids == "auto":
        show_ids = should_show_ids(labels)

    if not (show_ids or show_centroids):
        return canvas

    font_scale = _auto_font_scale(_median_diameter(labels))
    thickness = 1 if font_scale < 0.55 else 2

    from skimage.measure import regionprops_table

    table = regionprops_table(labels, properties=("label", "centroid"))
    for object_id, row, col in zip(
        table["label"], table["centroid-0"], table["centroid-1"]
    ):
        centre = (int(round(col)), int(round(row)))
        if show_centroids:
            cv2.circle(canvas, centre, max(1, thickness), boundary_color, -1)
        if show_ids:
            text = str(int(object_id))
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            origin = (centre[0] - text_w // 2, centre[1] + text_h // 2)
            # Dark outline first so white digits stay readable on bright objects.
            cv2.putText(
                canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                LABEL_OUTLINE, thickness + 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                LABEL_COLOR, thickness, cv2.LINE_AA,
            )

    return canvas


def mask_to_image(mask: np.ndarray) -> np.ndarray:
    """Binary mask as an 8-bit black-and-white image."""
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def labels_to_color(labels: np.ndarray) -> np.ndarray:
    """Colour each object differently, which makes split objects obvious."""
    labels = np.asarray(labels)
    if labels.max() == 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    colored = label2rgb(labels, bg_label=0)
    return (np.clip(colored, 0, 1) * 255).astype(np.uint8)


# Chart palette, matching the application theme. Charts sit inside a dark page,
# so a default white figure would punch a bright hole in it and make the
# micrographs beside it harder to read.
_PANEL = "#141B23"
_INK = "#E6EAF0"
_INK_DIM = "#8A97A8"
_RULE = "#2A3542"
_SIGNAL = "#F2C14E"   # amber, the measurement accent
_DAPI = "#7FB3D5"     # blue, the intensity channel

# Numbers in the charts are set in the same monospace face as the readout.
_MONO = ["SF Mono", "Menlo", "DejaVu Sans Mono", "monospace"]


def _histogram(
    values: pd.Series, title: str, xlabel: str, color: str
) -> plt.Figure:
    figure, axes = plt.subplots(figsize=(6, 3.2), dpi=120)
    figure.patch.set_facecolor(_PANEL)
    axes.set_facecolor(_PANEL)

    clean = pd.Series(values).dropna()

    if clean.empty:
        axes.text(0.5, 0.5, "No objects detected", ha="center", va="center",
                  transform=axes.transAxes, color=_INK_DIM)
        axes.set_xticks([])
        axes.set_yticks([])
    else:
        bins = int(np.clip(np.sqrt(len(clean)) * 1.5, 6, 40))
        axes.hist(clean, bins=bins, color=color, edgecolor=_PANEL, linewidth=0.7)
        axes.axvline(clean.median(), color=_INK, linestyle="--", linewidth=1.3,
                     label=f"median {clean.median():.1f}")
        legend = axes.legend(frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(_INK_DIM)
            text.set_fontfamily(_MONO)

    axes.set_title(title, fontsize=10, color=_INK, pad=10)
    axes.set_xlabel(xlabel, fontsize=9, color=_INK_DIM)
    axes.set_ylabel("Object count", fontsize=9, color=_INK_DIM)
    axes.tick_params(colors=_INK_DIM, labelsize=8)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontfamily(_MONO)
    axes.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_RULE)
    figure.tight_layout()
    return figure


def area_histogram(frame: pd.DataFrame) -> plt.Figure:
    column = "area_um2" if "area_um2" in frame.columns else "area_pixels"
    unit = "µm²" if column == "area_um2" else "pixels"
    values = frame[column] if not frame.empty else pd.Series(dtype=float)
    return _histogram(values, "Object size distribution", f"Area ({unit})", _SIGNAL)


def intensity_histogram(frame: pd.DataFrame) -> plt.Figure:
    values = frame["mean_intensity"] if not frame.empty else pd.Series(dtype=float)
    return _histogram(
        values, "Mean intensity distribution", "Mean intensity (0-255)", _DAPI
    )
