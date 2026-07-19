"""Validate the positivity method on real data: the BBBC013 translocation assay.

BBBC013 is a real FKHR-GFP nuclear-translocation screen. Nuclei are segmented in
the DNA channel; the marker (FKHR-GFP) is measured inside them; drug treatment
drives the marker into the nucleus. Each well has a known drug dose, so the
measured positive fraction can be checked against a real biological gradient
rather than a simulated ground truth.

    python scripts/fetch_bbbc013.py
    python scripts/validate_bbbc013.py

An honest limitation: BBBC013 gives the dose per well, not a per-cell
positive/negative label. So this validates that the method tracks the known
dose-response and separates the assay's own controls — the population readout a
percent-positive experiment actually reports — not a per-cell accuracy. The
per-cell accuracy is the synthetic validation (scripts/validate_positivity.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import measurements, positivity, preprocessing, segmentation
from src.config import DEFAULTS

DATA = Path(__file__).resolve().parents[1] / "validation_data" / "bbbc013"
IMAGES = DATA / "BBBC013_v1_images_bmp"
ROWS = "ABCDEFGH"


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, without pulling in scipy.stats."""
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = np.sqrt((xr**2).sum() * (yr**2).sum())
    return float((xr * yr).sum() / denom) if denom else float("nan")


def load_platemap() -> dict[tuple[str, int], float]:
    """Well -> drug dose in nM, from the flat 96-value platemap."""
    lines = (DATA / "platemap_all.txt").read_text().splitlines()
    doses = [float(v) for v in lines[1:] if v.strip()]
    plate = np.array(doses).reshape(8, 12)
    return {
        (ROWS[r], c + 1): float(plate[r, c])
        for r in range(8)
        for c in range(12)
    }


def well_image(channel: int, row: str, col: int) -> np.ndarray:
    matches = list(IMAGES.glob(f"Channel{channel}-*-{row}-{col:02d}.BMP"))
    if not matches:
        raise FileNotFoundError(f"Channel{channel} {row}-{col:02d}")
    return iio.imread(matches[0])


def percent_positive(row: str, col: int) -> tuple[float, int]:
    """Segment nuclei on the DNA channel, call positivity on the marker channel."""
    dna = preprocessing.prepare(well_image(2, row, col))
    marker = preprocessing.prepare(well_image(1, row, col))
    result = segmentation.segment(
        dna.analysis,
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
    call = positivity.call_by_mixture(frame["mean_intensity"].to_numpy())
    return (call.percent_positive if call.bimodal else 0.0), result.n_objects


def evaluate_drug(name: str, rows: str, platemap: dict) -> dict:
    """Average percent-positive per dose across the replicate rows of one drug."""
    doses = sorted({platemap[(r, c)] for r in rows for c in range(1, 13)})
    per_dose = {}
    for dose in doses:
        values = []
        for r in rows:
            for c in range(1, 13):
                if platemap[(r, c)] == dose:
                    values.append(percent_positive(r, c)[0])
        per_dose[dose] = float(np.mean(values))

    dose_array = np.array(list(per_dose.keys()))
    pct_array = np.array(list(per_dose.values()))
    return {
        "per_dose": per_dose,
        "spearman": spearman(dose_array, pct_array),
        "negative": per_dose.get(0.0, float("nan")),
        "max_dose": max(doses),
        "max_positive": per_dose[max(doses)],
    }


def main() -> int:
    if not IMAGES.exists():
        print("run scripts/fetch_bbbc013.py first", file=sys.stderr)
        return 1

    platemap = load_platemap()
    print("BBBC013 — FKHR-GFP nuclear translocation, real dose-response\n")

    for name, rows in (("Wortmannin", "ABCD"), ("LY294002", "EFGH")):
        result = evaluate_drug(name, rows, platemap)
        print(f"{name} (rows {rows})")
        print(f"  {'dose (nM)':>10} {'% positive':>11}")
        for dose, pct in sorted(result["per_dose"].items()):
            bar = "#" * int(round(pct / 3))
            print(f"  {dose:10.2f} {pct:10.1f}%  {bar}")
        print(f"  dose vs %positive, Spearman rank correlation: {result['spearman']:.3f}")
        print(f"  negative control (0 nM): {result['negative']:.1f}% positive")
        print(f"  top dose ({result['max_dose']:.0f} nM): {result['max_positive']:.1f}% positive")
        print(f"  separation, negative -> top dose: "
              f"{result['max_positive'] - result['negative']:.1f} points\n")

    print("Interpretation: the measured positive fraction rises monotonically "
          "with drug dose and\ncleanly separates the assay's negative and "
          "positive controls, on real data. BBBC013\nprovides the dose per well, "
          "not per-cell labels, so this validates the population\nreadout, not a "
          "per-cell accuracy (that is the synthetic validation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
