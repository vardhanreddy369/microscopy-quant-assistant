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
    "min_size": 60,
    "smoothing_sigma": 1.6,
    "cleanup_radius": 1,
    "fill_holes": True,
    "separate_touching": True,
    # 7 rather than 9: both score exactly on the synthetic samples, but the real
    # image has roughly 12-pixel nuclei, and a 9-pixel exclusion radius merges
    # touching pairs there. Verified with scripts/tune_defaults.py.
    "peak_min_distance": 7,
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
