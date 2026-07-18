"""Grid-search segmentation parameters on the BBBC039 *training* split.

    python scripts/tune_on_bbbc039.py

The test split is never touched here. Tuning on the data you then report on
inflates the score, so the search runs on the 100 training images only and the
chosen setting is scored separately by scripts/validate.py --split test.

Optimises average precision over IoU 0.50-0.90, which rewards getting object
boundaries right rather than merely getting the count right.
"""

from __future__ import annotations

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate import evaluate, load_split
from src.config import DEFAULTS

GRID = {
    "smoothing_sigma": [0.6, 1.0, 1.4, 1.8],
    "peak_min_distance": [3, 4, 5, 6, 7, 9],
    "min_size": [20, 40, 60, 100],
    "cleanup_radius": [0, 1],
}

FIXED = {"threshold_method": "otsu", "separate_touching": True}


def score_combo(params: dict) -> tuple[float, float, int, int, dict]:
    """Evaluate one parameter set on the training split."""
    stems = load_split("training")
    result = evaluate(stems, params, progress=False)
    summary = result.summary()
    return (
        summary["average_precision"],
        summary["f1_at_50"],
        summary["split_errors"],
        summary["merge_errors"],
        params,
    )


def main() -> int:
    combos = [
        {**FIXED, **dict(zip(GRID.keys(), values))}
        for values in itertools.product(*GRID.values())
    ]
    print(f"Evaluating {len(combos)} parameter sets on the training split "
          f"(100 images), test split untouched\n")

    results = []
    with ProcessPoolExecutor() as pool:
        for index, outcome in enumerate(pool.map(score_combo, combos), start=1):
            results.append(outcome)
            if index % 20 == 0:
                print(f"  ...{index}/{len(combos)}", flush=True)

    results.sort(key=lambda row: -row[0])

    print("\nTop 10 by average precision:")
    print(f"  {'AP':>6} {'F1@50':>6} {'split':>6} {'merge':>6}  parameters")
    for ap, f1, splits, merges, params in results[:10]:
        tidy = {k: params[k] for k in GRID}
        print(f"  {ap:.4f} {f1:.4f} {splits:6d} {merges:6d}  {tidy}")

    current = {**FIXED, **{k: DEFAULTS[k] for k in GRID}}
    current_row = next((r for r in results if r[4] == current), None)
    print("\nCurrent defaults:")
    if current_row:
        print(f"  AP={current_row[0]:.4f}  F1@50={current_row[1]:.4f}  "
              f"splits={current_row[2]}  merges={current_row[3]}")
    else:
        print("  (not in the search grid)")

    best_ap, best_f1, _, _, best = results[0]
    print(f"\nBest: {{k: best[k] for k in GRID}}".replace("{k: best[k] for k in GRID}",
                                                          str({k: best[k] for k in GRID})))
    print(f"  training AP={best_ap:.4f}  F1@50={best_f1:.4f}")
    print("\nNow confirm on the held-out test split, e.g.:")
    print(f"  python scripts/validate.py --split test "
          f"--smoothing {best['smoothing_sigma']} "
          f"--peak-distance {best['peak_min_distance']} "
          f"--min-size {best['min_size']} "
          f"--cleanup-radius {best['cleanup_radius']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
