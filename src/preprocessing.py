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
from skimage.morphology import disk, white_tophat

CHANNELS = ("grayscale", "red", "green", "blue")
BACKGROUNDS = ("dark", "light")

_CHANNEL_INDEX = {"red": 0, "green": 1, "blue": 2}


class ImageLoadError(ValueError):
    """Raised when an input cannot be interpreted as a 2-D image."""


def _read_all_frames(source) -> tuple[np.ndarray, bool]:
    """Read every frame in a file, not just the first.

    ``imageio.imread`` returns only page 0 of a multi-page file. For microscopy
    that is a silent data-loss bug: a confocal z-stack would be analysed as its
    first slice alone, which is often the most out-of-focus one, and the user
    would see a plausible count with no warning.

    ``index=...`` asks for every frame and always returns a leading frame axis,
    including for single-frame formats, so the caller can collapse it
    explicitly rather than having to guess whether axis 0 is frames or rows.

    Returns ``(array, has_frame_axis)``.
    """
    try:
        return np.asarray(iio.imread(source, index=...)), True
    except Exception:  # noqa: BLE001 - plugin may not support frame indexing
        return np.asarray(iio.imread(source)), False


def load_image(source) -> np.ndarray:
    """Load an image from a path, raw bytes, file-like object, or array.

    Multi-page TIFFs and z-stacks are reduced to a single plane with a maximum
    intensity projection, which is the conventional way to flatten a stack for
    2-D object counting.
    """
    if isinstance(source, np.ndarray):
        image = np.asarray(source)
    else:
        if hasattr(source, "read"):
            source = source.read()
        try:
            image, has_frame_axis = _read_all_frames(source)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as-is
            raise ImageLoadError(f"Could not read the image: {exc}") from exc

        if has_frame_axis:
            # Axis 0 is known to be frames here, so collapse it directly instead
            # of inferring from the shape, which would misread a 3-pixel-wide
            # image as a colour image.
            image = image[0] if image.shape[0] == 1 else image.max(axis=0)

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


def correct_illumination(plane: np.ndarray, radius: int) -> np.ndarray:
    """Flatten a slowly varying background (a white top-hat, or "rolling ball").

    A single global threshold assumes the background is uniform. When one corner
    of the field is brighter than another, any threshold that keeps the dim
    corner floods the bright one, and any threshold that suits the bright corner
    drops real objects in the dim one. Subtracting a morphological opening
    removes structure larger than ``radius`` while leaving the objects.

    ``radius`` must be larger than the objects being measured, or it removes
    them too. ``radius <= 0`` is a no-op.

    This is off by default. It costs an order of magnitude more time than the
    rest of the pipeline, and on evenly illuminated images it buys nothing.
    """
    if not radius or radius <= 0:
        return plane.astype(np.float32, copy=False)

    flattened = white_tophat(plane, disk(int(radius)))
    span = float(np.ptp(flattened))
    if span <= 0:
        return np.zeros_like(flattened, dtype=np.float32)
    return ((flattened - flattened.min()) / span).astype(np.float32)


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
