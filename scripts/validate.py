"""Score the segmentation pipeline against BBBC039 ground truth.

    python scripts/validate.py                 # held-out test split
    python scripts/validate.py --split all     # all 200 images
    python scripts/validate.py --json out.json

Reports object-level accuracy, not just counting accuracy: a method can get the
count right while splitting one nucleus and merging two others.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocessing, segmentation, validation
from src.config import DEFAULTS
from src.validation import DatasetScore, decode_colored_mask, score_image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation_data" / "bbbc039"
SPLITS = ("training", "validation", "test", "all")


def load_split(name: str) -> list[str]:
    """Return the image stems belonging to a split."""
    if name == "all":
        return sorted(p.stem for p in (DATA / "images").glob("*.tif"))
    listing = DATA / "metadata" / f"{name}.txt"
    if not listing.exists():
        raise FileNotFoundError(f"missing {listing}; run scripts/fetch_validation_data.py")
    return [
        Path(line.strip()).stem
        for line in listing.read_text().splitlines()
        if line.strip()
    ]


def evaluate(stems: list[str], params: dict, thresholds=validation.DEFAULT_THRESHOLDS,
             progress: bool = True) -> DatasetScore:
    scores = []
    for index, stem in enumerate(stems, start=1):
        image_path = DATA / "images" / f"{stem}.tif"
        mask_path = DATA / "masks" / f"{stem}.png"
        if not image_path.exists() or not mask_path.exists():
            continue

        prepared = preprocessing.prepare(image_path, channel="grayscale",
                                         background="dark")
        result = segmentation.segment(
            prepared.analysis,
            threshold_method=params["threshold_method"],
            min_size=params["min_size"],
            smoothing_sigma=params["smoothing_sigma"],
            cleanup_radius=params["cleanup_radius"],
            separate_touching=params["separate_touching"],
            peak_min_distance=params["peak_min_distance"],
        )
        truth = decode_colored_mask(iio.imread(mask_path))
        scores.append(score_image(result.labels, truth, thresholds, name=stem))

        if progress and index % 25 == 0:
            print(f"  ...{index}/{len(stems)}", flush=True)

    return DatasetScore(images=scores, thresholds=tuple(thresholds))


def report(score: DatasetScore, title: str) -> None:
    summary = score.summary()
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")
    print(f"images                {summary['images']}")
    print(f"true objects          {summary['true_objects']:,}")
    print(f"predicted objects     {summary['predicted_objects']:,}")
    print()
    print(f"F1 @ IoU 0.50         {summary['f1_at_50']:.3f}")
    print(f"  precision           {summary['precision_at_50']:.3f}")
    print(f"  recall              {summary['recall_at_50']:.3f}")
    print(f"F1 @ IoU 0.75         {summary['f1_at_75']:.3f}")
    print(f"average precision     {summary['average_precision']:.3f}   "
          f"(mean over IoU 0.50-0.90)")
    print(f"mean IoU of matches   {summary['mean_matched_iou']:.3f}")
    print()
    print(f"split errors          {summary['split_errors']:,}   "
          f"(one nucleus cut into pieces)")
    print(f"merge errors          {summary['merge_errors']:,}   "
          f"(several nuclei fused into one)")
    print(f"count error (MAPE)    {summary['count_mape']:.1%}")

    print("\nF1 by IoU threshold:")
    for threshold in score.thresholds:
        bar = "#" * int(round(score.f1_at(threshold) * 40))
        print(f"  {threshold:.2f}  {score.f1_at(threshold):.3f}  {bar}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--min-size", type=int, default=DEFAULTS["min_size"])
    parser.add_argument("--smoothing", type=float, default=DEFAULTS["smoothing_sigma"])
    parser.add_argument("--peak-distance", type=int,
                        default=DEFAULTS["peak_min_distance"])
    parser.add_argument("--cleanup-radius", type=int,
                        default=DEFAULTS["cleanup_radius"])
    parser.add_argument("--threshold-method", default=DEFAULTS["threshold_method"])
    parser.add_argument("--no-separate", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not (DATA / "images").exists():
        print("validation data not found; run scripts/fetch_validation_data.py",
              file=sys.stderr)
        return 1

    params = {
        "threshold_method": args.threshold_method,
        "min_size": args.min_size,
        "smoothing_sigma": args.smoothing,
        "cleanup_radius": args.cleanup_radius,
        "peak_min_distance": args.peak_distance,
        "separate_touching": not args.no_separate,
    }

    stems = load_split(args.split)
    print(f"Scoring {len(stems)} images from the '{args.split}' split")
    print(f"parameters: {params}")

    score = evaluate(stems, params)
    report(score, f"BBBC039 - {args.split} split")

    if args.json:
        args.json.write_text(json.dumps(
            {"split": args.split, "parameters": params, "summary": score.summary()},
            indent=2,
        ) + "\n")
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
