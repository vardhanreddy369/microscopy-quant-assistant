"""Image loading, channel selection, and intensity preparation.

The analysis image produced here is always a 2-D float32 array in [0, 1] where
high values mean "signal". Downstream segmentation assumes that convention, so
light-background images are inverted at this stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import imageio.v3 as iio
from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import gaussian

CHANNELS = ("grayscale", "red", "green", "blue")
BACKGROUNDS = ("dark", "light")

_CHANNEL_INDEX = {"red": 0, "green": 1, "blue": 2}


class ImageLoadError(ValueError):
    """Raised when an input cannot be interpreted as a 2-D image."""


def load_image(source) -> np.ndarray:
    """Load an image from a path, raw bytes, file-like object, or array.

    Multi-page TIFFs and z-stacks are reduced to a single plane with a maximum
    intensity projection, which is the conventional way to flatten a stack for
    2-D object counting.
    """
    if isinstance(source, np.ndarray):
        image = source
    else:
        if hasattr(source, "read"):
            source = source.read()
        try:
            image = iio.imread(source)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as-is
            raise ImageLoadError(f"Could not read the image: {exc}") from exc

    image = np.asarray(image)

    if image.ndim == 2:
        return image
    if image.ndim == 3:
        # (H, W, 3|4) is a colour image; anything else is treated as a stack.
        if image.shape[-1] in (3, 4):
            return image
        return image.max(axis=0)
    if image.ndim == 4:
        # (Z, H, W, C) stack -> project over Z, keep colour.
        return image.max(axis=0)

    raise ImageLoadError(
        f"Unsupported image with {image.ndim} dimensions and shape {image.shape}."
    )


def _to_float(image: np.ndarray) -> np.ndarray:
    """Scale an image to float32 in [0, 1].

    Integer images are converted against the *dtype* range, not their own
    observed range. Scaling by the observed peak would stretch every image to
    full brightness independently, which silently destroys the absolute
    intensity differences between images that a batch comparison depends on.
    A dim image must stay dim here; :func:`normalize_contrast` handles making
    it visible for segmentation.
    """
    image = np.asarray(image)

    if image.dtype == bool:
        return image.astype(np.float32)

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        span = float(info.max) - float(info.min)
        if span <= 0:
            return np.zeros(image.shape, dtype=np.float32)
        return ((image.astype(np.float32) - float(info.min)) / span).astype(np.float32)

    out = image.astype(np.float32, copy=True)
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return np.zeros_like(out)
    lo, hi = float(finite.min()), float(finite.max())
    # A float image already in [0, 1] is assumed to use the conventional range.
    # Anything else has no declared scale, so its own range is all there is.
    if lo >= 0.0 and hi <= 1.0:
        return out
    if hi <= lo:
        return np.zeros_like(out)
    return (out - lo) / (hi - lo)


def select_channel(image: np.ndarray, channel: str = "grayscale") -> np.ndarray:
    """Reduce an image to a single 2-D analysis plane.

    A single channel of a fluorescence image is usually the right analysis
    target; converting RGB to luminance mixes unrelated stains together.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")

    image = np.asarray(image)

    if image.ndim == 2:
        return _to_float(image)

    rgb = image[..., :3]
    if channel == "grayscale":
        return _to_float(rgb2gray(_to_float(rgb)))
    return _to_float(rgb[..., _CHANNEL_INDEX[channel]])


def normalize_contrast(
    plane: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.5
) -> np.ndarray:
    """Percentile contrast stretch.

    Percentiles rather than min/max so a handful of hot pixels cannot compress
    the rest of the histogram into a narrow band.
    """
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros_like(plane, dtype=np.float32)

    lo, hi = np.percentile(finite, (low_percentile, high_percentile))
    if hi <= lo:
        return np.zeros_like(plane, dtype=np.float32)
    return exposure.rescale_intensity(
        plane, in_range=(float(lo), float(hi)), out_range=(0.0, 1.0)
    ).astype(np.float32)


def smooth(plane: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur. sigma <= 0 is a no-op."""
    if sigma is None or sigma <= 0:
        return plane.astype(np.float32, copy=False)
    return gaussian(plane, sigma=float(sigma), preserve_range=True).astype(np.float32)


@dataclass
class PreparedImage:
    """The three views of an input image the rest of the pipeline needs."""

    original: np.ndarray
    """The image exactly as loaded, for display."""

    analysis: np.ndarray
    """2-D float32 in [0, 1], signal high, contrast-stretched. Segmentation input."""

    intensity: np.ndarray
    """2-D float32 in [0, 1], signal high, *not* contrast-stretched.

    Intensity statistics are measured on this plane rather than on ``analysis``
    because the percentile stretch is computed per image. Measuring stretched
    values would rescale every image onto the same range and destroy exactly the
    brightness differences a batch comparison is trying to detect.
    """


def prepare(
    source,
    channel: str = "grayscale",
    background: str = "dark",
    normalize: bool = True,
) -> PreparedImage:
    """Load an image and derive the display, segmentation, and intensity planes."""
    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {BACKGROUNDS}, got {background!r}")

    original = load_image(source)
    plane = select_channel(original, channel)

    if background == "light":
        plane = 1.0 - plane

    analysis = normalize_contrast(plane) if normalize else plane
    return PreparedImage(original=original, analysis=analysis, intensity=plane)
