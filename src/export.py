"""Serialisation helpers for downloads and command-line output."""

from __future__ import annotations

import io
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd


def dataframe_to_csv_bytes(frame: pd.DataFrame, float_format: str = "%.3f") -> bytes:
    """Encode measurements as UTF-8 CSV bytes."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, float_format=float_format)
    return buffer.getvalue().encode("utf-8")


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode an image array as PNG bytes."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        peak = float(array.max()) if array.size else 0.0
        array = array.astype(np.float32) / peak if peak > 1.0 else array.astype(np.float32)
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    return iio.imwrite("<bytes>", array, extension=".png")


def figure_to_png_bytes(figure) -> bytes:
    """Encode a matplotlib figure as PNG bytes."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    return buffer.getvalue()


def save_outputs(
    directory: str | Path,
    stem: str,
    frame: pd.DataFrame,
    annotated: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Path]:
    """Write CSV, annotated image, and mask to ``directory``. Returns the paths."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "csv": out_dir / f"{stem}_measurements.csv",
        "annotated": out_dir / f"{stem}_annotated.png",
        "mask": out_dir / f"{stem}_mask.png",
    }

    paths["csv"].write_bytes(dataframe_to_csv_bytes(frame))
    paths["annotated"].write_bytes(image_to_png_bytes(annotated))
    paths["mask"].write_bytes(image_to_png_bytes(mask))
    return paths
