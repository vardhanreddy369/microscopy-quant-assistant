"""One narrated demo: the process, the pictures, and the result, in order.

    python scripts/demo.py

Walks through a real analysis step by step — narrating what is happening,
showing the image at each stage, and printing the result ("the take") — then
does the same for the two-channel marker workflow. Built to be run live while
talking. Real images on iTerm2/Warp, colour block-art elsewhere.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import termimage
from src import markers, measurements, preprocessing, segmentation, visualization
from src.config import DEFAULTS, SAMPLE_DIR

RULE = "=" * 70


def step(number: str, title: str, what_happens: str) -> None:
    print("\n" + RULE)
    print(f" STEP {number} — {title}")
    print(RULE)
    print(what_happens + "\n")


def take(text: str) -> None:
    """Print the result of a step — 'the take'."""
    print(f"\n   >> RESULT: {text}")


def tint(plane: np.ndarray, channel: int) -> np.ndarray:
    """A grayscale plane shown in a single colour (0=red, 1=green, 2=blue)."""
    gray = visualization.to_display_rgb(plane)[..., 0]
    out = np.zeros((*gray.shape, 3), dtype=np.uint8)
    out[..., channel] = gray
    return out


def segment(plane):
    return segmentation.segment(
        plane, threshold_method="otsu", min_size=DEFAULTS["min_size"],
        smoothing_sigma=DEFAULTS["smoothing_sigma"], cleanup_radius=DEFAULTS["cleanup_radius"],
        separate_touching=True, peak_min_distance=DEFAULTS["peak_min_distance"],
        seeding=DEFAULTS["seeding"], seed_depth=DEFAULTS["seed_depth"],
    )


def figure_to_rgb(figure) -> np.ndarray:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=90, bbox_inches="tight",
                   facecolor=figure.get_facecolor())
    buffer.seek(0)
    return iio.imread(buffer)


def main() -> int:
    print(RULE)
    print(" BIOMEDICAL MICROSCOPY QUANTIFICATION — a walk through one analysis")
    print(RULE)

    # STEP 1 -----------------------------------------------------------------
    step("1", "LOAD THE IMAGE", "A microscopy image is just a grid of brightness "
         "numbers. Reading it off disk —\nno intelligence involved yet.")
    prepared = preprocessing.prepare(SAMPLE_DIR / "public_human_mitosis.png")
    termimage.show(visualization.to_display_rgb(prepared.analysis), cols=48,
                   caption="   the raw fluorescence image (grayscale — bright nuclei on dark):")
    take(f"{prepared.analysis.shape[1]}x{prepared.analysis.shape[0]} pixels loaded.")

    # STEP 2 -----------------------------------------------------------------
    step("2", "FIND THE CELLS", "Now it separates the nuclei: threshold the bright "
         "parts, find each cell's\ncentre, and 'flood' outward to split touching "
         "cells. Classic image processing.")
    t0 = time.perf_counter()
    result = segment(prepared.analysis)
    dt = (time.perf_counter() - t0) * 1000
    termimage.show_side_by_side(
        visualization.to_display_rgb(prepared.analysis),
        visualization.annotate(prepared.analysis, result.labels, show_ids=False),
        labels=("what the microscope saw", "what the tool found (yellow outlines)"),
        cols=42,
    )
    take(f"{result.n_objects} nuclei detected in {dt:.0f} milliseconds, "
         "touching cells separated.")

    # STEP 3 -----------------------------------------------------------------
    step("3", "MEASURE EVERY CELL", "For each nucleus it computes area, shape and "
         "brightness — the spreadsheet\na researcher actually wants.")
    frame = measurements.measure(result.labels, prepared.intensity)
    termimage.show(figure_to_rgb(visualization.area_histogram(frame)), cols=60,
                   caption="   distribution of cell sizes:")
    row = frame.iloc[0]
    take(f"{len(frame)} cells measured. e.g. cell #1: area {row['area_pixels']:.0f}px, "
         f"circularity {row['circularity']:.2f}.")

    # STEP 4 -----------------------------------------------------------------
    step("4", "PERCENT MARKER-POSITIVE  (the part for a marker-based lab)",
         "A two-channel image: nuclei in one colour, a marker in another. The tool\n"
         "measures the marker INSIDE each nucleus and reports what fraction are "
         "positive.\nThis is the shape of a TUNEL or caspase-3 readout.")
    nuclear = preprocessing.prepare(SAMPLE_DIR / "synthetic_marker_pair.png", channel="blue")
    marker = preprocessing.prepare(SAMPLE_DIR / "synthetic_marker_pair.png", channel="green")
    seg = segment(nuclear.analysis)
    marker_result = markers.measure_marker(seg.labels, marker.intensity, method="mixture")

    print("   The two channels, shown separately so each is unmistakable:")
    termimage.show_side_by_side(
        tint(nuclear.analysis, 2),   # nuclei in pure blue
        tint(marker.analysis, 1),    # marker in pure green
        labels=("NUCLEI channel (blue)", "MARKER channel (green)"),
        cols=42,
    )
    print("\n   The call — a yellow outline means marker-positive, a grey outline "
          "means negative\n   (negatives are kept because they are the denominator):")
    raw_pair = preprocessing.load_image(SAMPLE_DIR / "synthetic_marker_pair.png")
    termimage.show(
        visualization.annotate_marker(raw_pair, seg.labels, marker_result.frame),
        cols=48,
    )
    take(f"{marker_result.positive} of {marker_result.total} cells positive "
         f"= {marker_result.percent:.1f}%.")

    # STEP 5 -----------------------------------------------------------------
    step("5", "HOW IT CHOSE THE THRESHOLD  (the reproducible part)",
         "It did not eyeball a cutoff. It fitted two populations to the marker\n"
         "brightness and put the line where they cross — the same number every run —\n"
         "and first tested that a positive population genuinely exists.")
    termimage.show(
        figure_to_rgb(visualization.marker_histogram(marker_result.frame,
                                                     marker_result.threshold)),
        cols=60,
        caption="   the two populations, split at the learned threshold:",
    )
    take(f"threshold {marker_result.threshold:.0f}/255, a distinct positive "
         "population confirmed.")

    print("\n" + RULE)
    print(" Same image in, same numbers out — every run. Nothing here is staged.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
