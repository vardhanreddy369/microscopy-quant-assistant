"""Shared analysis defaults.

The Streamlit app and the command-line script both read these so the demo
behaves identically in either place. The values are tuned against
sample_data/public_human_mitosis.png; see scripts/tune_defaults.py.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"
OUTPUT_DIR = ROOT / "outputs"

# Interactive size limits. These live here rather than in app.py so the single
# image and batch paths cannot drift apart: the batch loop previously had no
# guard at all, and one oversized file could stall an entire run.
MAX_PIXELS = 40_000_000
SLOW_PIXELS = 6_000_000

SIZE_OK = "ok"
SIZE_SLOW = "slow"
SIZE_TOO_LARGE = "too_large"


def size_verdict(height: int, width: int) -> tuple[str, str]:
    """Classify an image by pixel count. Returns ``(verdict, message)``."""
    pixels = int(height) * int(width)
    if pixels > MAX_PIXELS:
        return SIZE_TOO_LARGE, (
            f"{width}x{height} pixels exceeds the {MAX_PIXELS:,} pixel limit for "
            "interactive analysis. Crop or downscale the image first."
        )
    if pixels > SLOW_PIXELS:
        return SIZE_SLOW, f"Large image ({width}x{height}). Analysis may take a few seconds."
    return SIZE_OK, ""


DEFAULTS = {
    "channel": "grayscale",
    "background": "dark",
    "threshold_method": "otsu",
    "manual_threshold": 0.35,
    # These four were chosen by grid search on the BBBC039 *training* split
    # (100 real fluorescence images, ~11,000 hand-annotated nuclei) and then
    # confirmed on its held-out test split: see scripts/tune_on_bbbc039.py and
    # docs/VALIDATION_DATA.md.
    #
    # They replace an earlier set picked by eye on a single crop. Light
    # smoothing with no morphological dilation keeps object boundaries tight,
    # which is what lifted mean matched IoU from 0.862 to 0.886. The synthetic
    # samples score identically either way, so only the annotated data could
    # distinguish them.
    "min_size": 60,
    "smoothing_sigma": 0.6,
    "cleanup_radius": 0,
    "fill_holes": True,
    "separate_touching": True,
    "peak_min_distance": 9,
    # Off by default. It is ~10x slower than the rest of the pipeline and on the
    # evenly illuminated BBBC039 benchmark it is worth at most +0.004 F1, inside
    # noise. It matters a great deal on unevenly lit images: it takes the
    # difficult synthetic sample from 72 to 90 of 110 objects.
    "background_radius": 0,
    "pixel_size_um": None,
}

# Ordered for the demo: the real fluorescence image first, so the app opens on
# the case the pipeline handles best.
SAMPLE_IMAGES = (
    (
        "Human mitosis (real, fluorescence)",
        "public_human_mitosis.png",
        "Real public fluorescence image. Bright nuclei on a dark background, "
        "with both separated and touching nuclei.",
    ),
    (
        "Synthetic - easy",
        "synthetic_easy.png",
        "Well separated simulated nuclei, high signal-to-noise.",
    ),
    (
        "Synthetic - touching",
        "synthetic_moderate.png",
        "Clustered and touching simulated nuclei. This is the case the "
        "watershed step exists to handle.",
    ),
    (
        "Synthetic - difficult (known failure case)",
        "synthetic_difficult.png",
        "Dense, noisy, unevenly illuminated. The classical pipeline does not "
        "handle this well; it is included to show the limitation honestly.",
    ),
    (
        "Immunohistochemistry (real, brightfield)",
        "public_immunohistochemistry.png",
        "Real public H-DAB stained tissue, RGB and light background. Exercises "
        "channel selection. The pipeline is not validated for brightfield "
        "histology.",
    ),
)

# Per-sample overrides where the tuned global defaults are not appropriate.
SAMPLE_OVERRIDES = {
    "public_immunohistochemistry.png": {"channel": "blue", "background": "light"},
}

# Samples whose output must not be read as a valid measurement. Reporting a
# tidy object count and an area column for a result this wrong is precisely the
# "confidently wrong number" this project sets out to avoid, so the app says so
# on screen rather than leaving it to the caption.
SAMPLE_CAVEATS = {
    "public_immunohistochemistry.png": (
        "These numbers are not valid nucleus measurements. This pipeline "
        "thresholds bright objects on a dark background; on brightfield stained "
        "tissue it instead outlines whole tissue regions and cuts them into "
        "arbitrary watershed polygons. Here roughly 58% of the image is treated "
        "as foreground and the average 'object' is about 89 pixels across, "
        "where a real nucleus would be 10-20. Brightfield histology needs stain "
        "colour deconvolution, which this tool does not do. The sample is kept "
        "to show what running a method outside its domain looks like."
    ),
}


def sample_path(filename: str) -> Path:
    return SAMPLE_DIR / filename
