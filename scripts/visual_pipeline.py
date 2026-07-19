"""The pipeline walkthrough, with the actual images and charts shown in the terminal.

    python scripts/visual_pipeline.py

Renders the micrograph, the segmentation overlay, the two-channel marker call,
and the marker-intensity histogram inline, so you can narrate what the numbers
mean while the pictures are on screen. Uses real terminal graphics on iTerm2 or
Warp, and colour block-art everywhere else.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import termimage
from src import measurements, positivity, preprocessing, segmentation, visualization
from src.config import DEFAULTS, SAMPLE_DIR

RULE = "=" * 70


def segment(plane, **over):
    params = dict(
        threshold_method="otsu", min_size=DEFAULTS["min_size"],
        smoothing_sigma=DEFAULTS["smoothing_sigma"], cleanup_radius=DEFAULTS["cleanup_radius"],
        separate_touching=True, peak_min_distance=DEFAULTS["peak_min_distance"],
        seeding=DEFAULTS["seeding"], seed_depth=DEFAULTS["seed_depth"],
    )
    params.update(over)
    return segmentation.segment(plane, **params)


def figure_to_rgb(figure) -> np.ndarray:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=90, bbox_inches="tight",
                   facecolor=figure.get_facecolor())
    buffer.seek(0)
    return iio.imread(buffer)


def main() -> int:
    print(RULE)
    print(" MICROSCOPY PIPELINE — the pictures behind the numbers")
    print(RULE)

    # 1. Segmentation, shown as original vs annotated.
    prepared = preprocessing.prepare(SAMPLE_DIR / "public_human_mitosis.png")
    result = segment(prepared.analysis)
    frame = measurements.measure(result.labels, prepared.intensity)

    print("\n1. SEGMENTATION — real fluorescence image, nuclei found automatically")
    print(f"   {result.n_objects} nuclei detected. Left: what the microscope saw. "
          "Right: what the tool found (yellow outlines).\n")
    termimage.show_side_by_side(
        visualization.to_display_rgb(prepared.analysis),
        visualization.annotate(prepared.analysis, result.labels, show_ids=False),
        labels=("original", f"annotated — {result.n_objects} cells"),
        cols=40,
    )

    # 2. The measurement histogram.
    print("\n2. MEASUREMENTS — every cell's size, as a distribution")
    termimage.show(figure_to_rgb(visualization.area_histogram(frame)), cols=64)

    # 3. Two-channel marker call.
    print("\n" + RULE)
    print(" MARKER-POSITIVE — the part built for a marker-based lab")
    print(RULE)
    nuclear = preprocessing.prepare(SAMPLE_DIR / "synthetic_marker_pair.png", channel="blue")
    marker = preprocessing.prepare(SAMPLE_DIR / "synthetic_marker_pair.png", channel="green")
    seg = segment(nuclear.analysis)
    marker_frame = measurements.measure(seg.labels, marker.intensity)
    call = positivity.call_by_mixture(marker_frame["mean_intensity"].to_numpy())
    marker_frame["marker_positive"] = call.positive

    print(f"\n3. POSITIVITY — {call.n_positive} of {call.n_total} cells positive "
          f"= {call.percent_positive:.1f}%")
    print("   Left: nuclei (blue channel). Right: the call — amber positive, "
          "slate negative.\n")
    termimage.show_side_by_side(
        visualization.to_display_rgb(nuclear.analysis),
        visualization.annotate_marker(marker.analysis, seg.labels, marker_frame),
        labels=("nuclei", f"{call.percent_positive:.0f}% positive"),
        cols=40,
    )

    print("\n4. THE THRESHOLD IT LEARNED — two populations, split at "
          f"{call.threshold:.0f}/255")
    termimage.show(
        figure_to_rgb(visualization.marker_histogram(marker_frame, call.threshold)),
        cols=64,
    )

    print("\n" + RULE)
    print(" Same images, same numbers, every run. Nothing here is staged.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
