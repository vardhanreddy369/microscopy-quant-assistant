"""Validate the marker-positivity method on accuracy AND reproducibility.

Accuracy is the usual axis: does the measured positive fraction match the known
one. Reproducibility is the axis the literature says is missing — a systematic
review of ~9,000 immunofluorescence papers found fewer than 10% report their
thresholding, and manual thresholds carry documented bias.

    python scripts/validate_positivity.py

Three things are measured:

  1. accuracy    -- recovered fraction against the known fraction, 0% to 70%
  2. determinism -- the same image analysed twice gives an identical number
  3. bias        -- how far a plausible human threshold can move the result,
                    which is the reproducibility problem quantified
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import measurements, positivity, preprocessing, segmentation
from src.config import DEFAULTS, SAMPLE_DIR

SERIES = [
    "synthetic_marker_00pct.png",
    "synthetic_marker_10pct.png",
    "synthetic_marker_30pct.png",
    "synthetic_marker_pair.png",
    "synthetic_marker_50pct.png",
    "synthetic_marker_70pct.png",
]


def marker_intensities(filename: str) -> np.ndarray:
    nuclear = preprocessing.prepare(SAMPLE_DIR / filename, channel="blue")
    marker = preprocessing.prepare(SAMPLE_DIR / filename, channel="green")
    result = segmentation.segment(
        nuclear.analysis,
        threshold_method="otsu",
        min_size=DEFAULTS["min_size"],
        smoothing_sigma=DEFAULTS["smoothing_sigma"],
        cleanup_radius=DEFAULTS["cleanup_radius"],
        separate_touching=True,
        peak_min_distance=DEFAULTS["peak_min_distance"],
        seeding=DEFAULTS["seeding"],
        seed_depth=DEFAULTS["seed_depth"],
    )
    frame = measurements.measure(result.labels, marker.intensity)
    return frame["mean_intensity"].to_numpy()


def main() -> int:
    truth_path = SAMPLE_DIR / "marker_ground_truth.json"
    if not truth_path.exists():
        print("run scripts/make_sample_data.py first", file=sys.stderr)
        return 1
    truth = json.loads(truth_path.read_text())

    print("1. ACCURACY — mixture method against known fractions")
    print(f"   {'sample':30s} {'true%':>6} {'measured':>9} {'error':>7} {'ΔBIC':>8}")
    errors = []
    for name in SERIES:
        values = marker_intensities(name)
        result = positivity.call_by_mixture(values)
        true = truth[name]["percent_positive"]
        measured = result.percent_positive if result.bimodal else 0.0
        error = measured - true
        if true > 0:
            errors.append(abs(error))
        shown = f"{measured:.1f}%" if result.bimodal else "none"
        print(f"   {name:30s} {true:6.1f} {shown:>9} {error:+7.1f} {result.delta_bic:8.1f}")
    print(f"   mean absolute error over the positive series: {np.mean(errors):.2f} points")
    print(f"   all-negative image correctly reported as no population: "
          f"{not positivity.call_by_mixture(marker_intensities(SERIES[0])).bimodal}")

    print("\n2. DETERMINISM — the same image, analysed ten times")
    values = marker_intensities("synthetic_marker_pair.png")
    thresholds = {positivity.call_by_mixture(values).threshold for _ in range(10)}
    print(f"   distinct thresholds across 10 runs: {len(thresholds)} "
          f"({'identical every time' if len(thresholds) == 1 else 'NOT reproducible'})")

    print("\n3. BIAS — how far a human threshold can move the answer")
    print("   The reproducibility problem, quantified. Two analysts pick different")
    print("   but individually reasonable cut-offs on the same image.")
    values = marker_intensities("synthetic_marker_30pct.png")
    true = truth["synthetic_marker_30pct.png"]["percent_positive"]
    automatic = positivity.call_by_mixture(values).percent_positive
    # A band of defensible manual thresholds around the data's midrange.
    lo, hi = np.percentile(values, 40), np.percentile(values, 60)
    manual_low = positivity.call_by_manual(values, lo).percent_positive
    manual_high = positivity.call_by_manual(values, hi).percent_positive
    print(f"   true fraction                         : {true:.1f}%")
    print(f"   mixture (objective, reproducible)     : {automatic:.1f}%")
    print(f"   manual threshold at the 40th pct      : {manual_low:.1f}%")
    print(f"   manual threshold at the 60th pct      : {manual_high:.1f}%")
    print(f"   spread between two 'reasonable' manual calls: "
          f"{abs(manual_high - manual_low):.1f} points")
    print("   The mixture removes that spread: one image gives one number.")

    print("\n4. EXAMPLE REPRODUCIBILITY REPORT")
    result = positivity.call_by_mixture(marker_intensities("synthetic_marker_pair.png"))
    report = positivity.reproducibility_report(result, {
        "image": "synthetic_marker_pair.png",
        "nuclear channel": "blue",
        "marker channel": "green",
        "segmentation": "classical watershed, h-maxima seeding",
    })
    print("\n".join("   " + line for line in report.splitlines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
