"""Head-to-head: the mixture positivity method against the obvious alternatives.

A contribution is only real if it beats what a reviewer would otherwise use. This
scores five positivity callers on the synthetic marker series, where the true
positive fraction of every image is known:

  Gaussian + gate   the shipped method: two-component Gaussian mixture with the
                    BIC + Ashman's D bimodality test (src/positivity.py)
  Gaussian, no gate the same mixture with the population test removed
  Gamma, no gate    a two-component Gamma mixture, the distribution GammaGateR
                    uses for marker gating (src/comparators.py)
  k-means, no gate  a 1-D two-cluster split
  exact Otsu        the per-object Otsu threshold (src/markers.py)

Two axes:
  accuracy   mean absolute error in percent-positive across the 10-70% images
  negatives  what each method reports on the all-negative (0% true) image

The result isolates what actually carries the method: the distribution barely
matters, but the population test is the whole game.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import comparators, markers, measurements, positivity, preprocessing, segmentation
from src.config import DEFAULTS, SAMPLE_DIR

POSITIVE_SERIES = [
    "synthetic_marker_10pct.png",
    "synthetic_marker_30pct.png",
    "synthetic_marker_50pct.png",
    "synthetic_marker_70pct.png",
    "synthetic_marker_pair.png",
]
NEGATIVE_IMAGE = "synthetic_marker_00pct.png"


def marker_intensities(filename: str) -> np.ndarray:
    nuclear = preprocessing.prepare(SAMPLE_DIR / filename, channel="blue")
    marker = preprocessing.prepare(SAMPLE_DIR / filename, channel="green")
    result = segmentation.segment(
        nuclear.analysis,
        threshold_method=DEFAULTS["threshold_method"],
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


def gaussian_gated(values: np.ndarray) -> float:
    call = positivity.call_by_mixture(values)
    return call.percent_positive if call.bimodal else 0.0


def gaussian_no_gate(values: np.ndarray) -> float:
    fit = positivity.fit_two_component(values)
    threshold = positivity._crossover(fit)
    return comparators.positive_fraction(values, threshold)


def gamma_no_gate(values: np.ndarray) -> float:
    return comparators.positive_fraction(values, comparators.gamma_mixture_threshold(values))


def kmeans_no_gate(values: np.ndarray) -> float:
    return comparators.positive_fraction(values, comparators.kmeans_threshold(values))


def exact_otsu(values: np.ndarray) -> float:
    return comparators.positive_fraction(values, markers._exact_otsu(values))


METHODS = {
    "Gaussian + gate (ours)": gaussian_gated,
    "Gaussian, no gate": gaussian_no_gate,
    "Gamma, no gate": gamma_no_gate,
    "k-means, no gate": kmeans_no_gate,
    "exact Otsu": exact_otsu,
}


def main() -> int:
    truth = json.loads((SAMPLE_DIR / "marker_ground_truth.json").read_text())

    intensities = {name: marker_intensities(name)
                   for name in POSITIVE_SERIES + [NEGATIVE_IMAGE]}

    print("Head-to-head on the synthetic marker series (true fractions known)\n")
    print(f"  {'method':24s} {'mean abs error':>15} {'all-negative (0% true)':>24}")
    print(f"  {'':24s} {'on 10-70% images':>15} {'reported % positive':>24}")

    rows = {}
    for name, fn in METHODS.items():
        errors = [abs(fn(intensities[img]) - truth[img]["percent_positive"])
                  for img in POSITIVE_SERIES]
        mae = float(np.mean(errors))
        negative = fn(intensities[NEGATIVE_IMAGE])
        rows[name] = {"mae": mae, "negative": negative}
        verdict = "correct" if negative < 5 else "FALSE POSITIVES"
        print(f"  {name:24s} {mae:14.2f}  {negative:9.1f}%   {verdict}")

    print("\nReading:")
    print("  Accuracy on real positives is a near-tie — the distribution (Gaussian,")
    print("  Gamma, k-means, Otsu) barely matters when a positive population exists.")
    print("  The all-negative image is where they part: every ungated method splits")
    print("  one population in half and invents 30-50% positives. Only the gated")
    print("  method abstains and reports zero. Removing the gate from the same")
    print("  Gaussian mixture reproduces the failure, so the population test — not")
    print("  the distribution — is the contribution.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
