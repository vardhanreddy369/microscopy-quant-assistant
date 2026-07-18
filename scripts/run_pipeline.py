"""Command-line entry point for the analysis pipeline.

This is the headless equivalent of the Streamlit app and the check that the
pipeline works independently of any UI:

    python scripts/run_pipeline.py sample_data/public_human_mitosis.png

It writes a CSV, a mask, and an annotated image, and prints the object count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import export, measurements, preprocessing, segmentation, visualization
from src.config import DEFAULTS, OUTPUT_DIR
from src.preprocessing import CHANNELS, BACKGROUNDS
from src.segmentation import THRESHOLD_METHODS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="path to the image to analyse")
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--channel", choices=CHANNELS, default=DEFAULTS["channel"])
    parser.add_argument("--background", choices=BACKGROUNDS, default=DEFAULTS["background"])
    parser.add_argument(
        "--threshold-method", choices=THRESHOLD_METHODS, default=DEFAULTS["threshold_method"]
    )
    parser.add_argument("--manual-threshold", type=float, default=None)
    parser.add_argument("--min-size", type=int, default=DEFAULTS["min_size"])
    parser.add_argument("--smoothing", type=float, default=DEFAULTS["smoothing_sigma"])
    parser.add_argument("--cleanup-radius", type=int, default=DEFAULTS["cleanup_radius"])
    parser.add_argument("--peak-distance", type=int, default=DEFAULTS["peak_min_distance"])
    parser.add_argument(
        "--no-separate", action="store_true", help="skip the watershed splitting step"
    )
    parser.add_argument(
        "--pixel-size", type=float, default=None,
        help="micrometres per pixel; omit to report pixels only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.image.exists():
        print(f"error: {args.image} does not exist", file=sys.stderr)
        return 2

    prepared = preprocessing.prepare(
        args.image, channel=args.channel, background=args.background
    )

    result = segmentation.segment(
        prepared.analysis,
        threshold_method=args.threshold_method,
        manual_threshold=args.manual_threshold,
        min_size=args.min_size,
        smoothing_sigma=args.smoothing,
        cleanup_radius=args.cleanup_radius,
        separate_touching=not args.no_separate,
        peak_min_distance=args.peak_distance,
    )

    frame = measurements.measure(result.labels, prepared.intensity, args.pixel_size)
    summary = measurements.summarize(frame)

    annotated = visualization.annotate(prepared.analysis, result.labels)
    mask_image = visualization.mask_to_image(result.mask)

    paths = export.save_outputs(
        args.outdir, args.image.stem, frame, annotated, mask_image
    )

    print(f"image            {args.image}")
    print(f"threshold        {result.threshold:.4f}  ({result.method})")
    print(f"objects detected {summary['count']}")
    if summary["count"]:
        print(f"  touching border  {summary['border_objects']}")
        print(f"  mean area        {summary['mean_area']:.1f} px")
        print(f"  median area      {summary['median_area']:.1f} px")
        print(f"  mean intensity   {summary['mean_intensity']:.1f} (0-255)")
        print(f"  mean circularity {summary['mean_circularity']:.3f}")
    else:
        print("  no objects found - try lowering the threshold or minimum size")

    print(f"rows in CSV      {len(frame)}")
    for name, path in paths.items():
        print(f"wrote {name:10s} {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
