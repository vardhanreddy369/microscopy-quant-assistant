"""Grid-search the segmentation defaults against known object counts.

The synthetic samples record how many nuclei were actually drawn, so counting
accuracy can be measured instead of judged by eye. This script picks the
parameter set with the lowest error on the two cases the pipeline claims to
handle (easy and touching) and reports the difficult case separately, since
tuning to a known failure would just hide it.

    python scripts/tune_defaults.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocessing, segmentation
from src.config import DEFAULTS, SAMPLE_DIR

SCORED = ("synthetic_easy.png", "synthetic_moderate.png")
REPORTED_ONLY = ("synthetic_difficult.png",)

# Superseded for choosing the shipped defaults: those now come from
# scripts/tune_on_bbbc039.py, which scores against real expert annotations
# rather than known counts. This script is kept because it is the only check
# that a parameter change does not break exact counting on the synthetic
# samples, which the annotation metrics do not measure.
GRID = {
    "smoothing_sigma": [0.6, 1.0, 1.4, 1.6, 2.0, 2.4],
    "peak_min_distance": [5, 7, 9, 11, 13],
    "min_size": [40, 60, 80],
}


def count_objects(path: Path, **params) -> int:
    prepared = preprocessing.prepare(path, channel="grayscale", background="dark")
    result = segmentation.segment(
        prepared.analysis,
        threshold_method="otsu",
        min_size=params["min_size"],
        smoothing_sigma=params["smoothing_sigma"],
        peak_min_distance=params["peak_min_distance"],
        separate_touching=True,
    )
    return result.n_objects


def main() -> int:
    truth = json.loads((SAMPLE_DIR / "ground_truth.json").read_text())

    combos = [
        dict(zip(GRID.keys(), values))
        for values in itertools.product(*GRID.values())
    ]
    print(f"Evaluating {len(combos)} parameter combinations on {len(SCORED)} scored images\n")

    scored_results = []
    for params in combos:
        errors = []
        counts = {}
        for name in SCORED:
            detected = count_objects(SAMPLE_DIR / name, **params)
            counts[name] = detected
            expected = truth[name]
            errors.append(abs(detected - expected) / expected)
        scored_results.append((sum(errors) / len(errors), params, counts))

    scored_results.sort(key=lambda row: row[0])

    print("Top 5 parameter sets (mean relative count error):")
    for error, params, counts in scored_results[:5]:
        detail = "  ".join(
            f"{name.replace('synthetic_', '').replace('.png', '')}={counts[name]}/{truth[name]}"
            for name in SCORED
        )
        print(f"  error={error:6.3f}  {params}  {detail}")

    best_error, best_params, best_counts = scored_results[0]

    print(f"\nBest: {best_params}  (mean relative error {best_error:.3f})")
    print("\nCurrent defaults in src/config.py:")
    current = {key: DEFAULTS[key] for key in GRID}
    print(f"  {current}")

    current_error = next(
        (row[0] for row in scored_results if row[1] == current), None
    )
    if current_error is None:
        print("  (not in the search grid)")
    else:
        print(f"  mean relative error {current_error:.3f}")
        if current_error <= best_error + 1e-9:
            print("  -> current defaults are already optimal on this grid")
        else:
            print(f"  -> consider switching to {best_params}")

    print("\nKnown failure case, reported not tuned:")
    for name in REPORTED_ONLY:
        detected = count_objects(SAMPLE_DIR / name, **best_params)
        expected = truth[name]
        print(f"  {name}: detected {detected} of {expected} true objects "
              f"({detected / expected:.0%} recall)")

    print("\nReal image (no ground truth available, checked for stability):")
    for sigma in (1.4, 1.6, 2.0):
        params = {**best_params, "smoothing_sigma": sigma}
        detected = count_objects(SAMPLE_DIR / "public_human_mitosis.png", **params)
        print(f"  public_human_mitosis.png: smoothing={sigma} -> {detected} objects")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
