"""Render the example figure used in the README.

    python scripts/make_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import measurements, preprocessing, segmentation, visualization
from src.config import DEFAULTS, ROOT, SAMPLE_DIR

ASSETS = ROOT / "assets"
SAMPLE = "public_human_mitosis.png"
ZOOM = (300, 430, 300, 430)  # a dense region, to show the splitting up close


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)

    prepared = preprocessing.prepare(SAMPLE_DIR / SAMPLE)
    result = segmentation.segment(
        prepared.analysis,
        min_size=DEFAULTS["min_size"],
        smoothing_sigma=DEFAULTS["smoothing_sigma"],
        peak_min_distance=DEFAULTS["peak_min_distance"],
        separate_touching=True,
    )
    frame = measurements.measure(result.labels, prepared.intensity)

    display = visualization.to_display_rgb(prepared.analysis)
    annotated = visualization.annotate(prepared.analysis, result.labels, show_ids=False)
    y0, y1, x0, x1 = ZOOM

    figure, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=130)

    axes[0, 0].imshow(display)
    axes[0, 0].set_title("Original (public fluorescence image)", fontsize=11)

    axes[0, 1].imshow(annotated)
    axes[0, 1].set_title(
        f"Detected objects: {result.n_objects}", fontsize=11
    )
    axes[0, 1].add_patch(
        plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                      edgecolor="#e53935", linewidth=1.6)
    )

    axes[1, 0].imshow(annotated[y0:y1, x0:x1], interpolation="nearest")
    axes[1, 0].set_title("Detail: touching nuclei separated by watershed", fontsize=11)

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks([])
        axis.set_yticks([])

    chart = axes[1, 1]
    chart.hist(frame["area_pixels"], bins=32, color="#4c8dae",
               edgecolor="white", linewidth=0.6)
    chart.axvline(frame["area_pixels"].median(), color="#d32f2f", linestyle="--",
                  linewidth=1.5,
                  label=f"median {frame['area_pixels'].median():.0f} px")
    chart.set_title("Object size distribution", fontsize=11)
    chart.set_xlabel("Area (pixels)")
    chart.set_ylabel("Object count")
    chart.legend(frameon=False, fontsize=9)
    chart.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Biomedical Microscopy Quantification Assistant - classical segmentation pipeline",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    output = ASSETS / "example_result.png"
    figure.savefig(output, bbox_inches="tight")
    print(f"wrote {output}  ({result.n_objects} objects, {len(frame)} measured rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
