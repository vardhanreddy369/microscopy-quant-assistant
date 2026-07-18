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


def sample_path(filename: str) -> Path:
    return SAMPLE_DIR / filename
