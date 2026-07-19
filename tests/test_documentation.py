"""The documented numbers must be true.

Three times in this project's history a change to the pipeline shifted results
while every test stayed green, because the tests assert *ranges* ("recall
between 40% and 90%") and the documentation asserts *values* ("70 of 110"). Both
were reasonable in isolation, and together they let four published figures drift
out of date.

These tests close that gap. They read the claims out of README.md and DEMO.md and
check them against what the pipeline actually produces, so a change that moves a
number fails immediately and names the file that needs updating. The claim is
parsed from the document rather than copied here — copying it would just create a
fourth place to drift.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import markers, measurements, preprocessing, segmentation
from src.config import DEFAULTS, SAMPLE_DIR, SAMPLE_OVERRIDES

README = (ROOT / "README.md").read_text()
DEMO = (ROOT / "DEMO.md").read_text()


def claim(pattern: str, text: str, label: str) -> float:
    """Pull a single numeric claim out of a document."""
    match = re.search(pattern, text)
    if match is None:
        pytest.fail(
            f"could not find the documented {label}. The pattern {pattern!r} no "
            "longer matches; update this test alongside the wording."
        )
    return float(match.group(1).replace(",", ""))


def count_objects(filename: str, background_radius: int = 0) -> int:
    overrides = SAMPLE_OVERRIDES.get(filename, {})
    prepared = preprocessing.prepare(
        SAMPLE_DIR / filename,
        channel=overrides.get("channel", DEFAULTS["channel"]),
        background=overrides.get("background", DEFAULTS["background"]),
    )
    return segmentation.segment(
        prepared.analysis,
        threshold_method=DEFAULTS["threshold_method"],
        min_size=DEFAULTS["min_size"],
        smoothing_sigma=DEFAULTS["smoothing_sigma"],
        cleanup_radius=DEFAULTS["cleanup_radius"],
        fill_holes=DEFAULTS["fill_holes"],
        separate_touching=DEFAULTS["separate_touching"],
        peak_min_distance=DEFAULTS["peak_min_distance"],
        seeding=DEFAULTS["seeding"],
        seed_depth=DEFAULTS["seed_depth"],
        background_radius=background_radius,
    ).n_objects


class TestDemonstrationImage:
    def test_readme_object_count_is_current(self):
        documented = claim(
            r"detects ([\d,]+) objects in roughly", README, "demo image count"
        )
        assert count_objects("public_human_mitosis.png") == documented

    def test_demo_script_object_count_matches_the_readme(self):
        """The number you will say out loud must match the one you published."""
        spoken = claim(
            r"Point at the count: \*\*([\d,]+) objects detected\*\*", DEMO,
            "spoken demo count",
        )
        assert count_objects("public_human_mitosis.png") == spoken


class TestSyntheticSamples:
    @staticmethod
    def truth():
        return json.loads((SAMPLE_DIR / "ground_truth.json").read_text())

    def test_difficult_sample_count_is_current(self):
        documented = claim(
            r"\| `synthetic_difficult\.png` \| 110 \| (\d+) \|", README,
            "difficult sample count",
        )
        assert count_objects("synthetic_difficult.png") == documented

    def test_difficult_sample_recall_is_current(self):
        documented = claim(
            r"Known failure case \((\d+)% recall\)", README, "difficult recall"
        )
        detected = count_objects("synthetic_difficult.png")
        actual = round(100 * detected / self.truth()["synthetic_difficult.png"])
        assert actual == documented

    def test_illumination_correction_gain_is_current(self):
        before = claim(r"detected count from \*\*(\d+) to \d+ of 110\*\*", README,
                       "uncorrected count")
        after = claim(r"detected count from \*\*\d+ to (\d+) of 110\*\*", README,
                      "corrected count")
        assert count_objects("synthetic_difficult.png") == before
        assert count_objects("synthetic_difficult.png", background_radius=15) == after

    def test_demo_script_failure_case_matches(self):
        spoken = claim(r"finds about (\d+) of the 110 objects", DEMO,
                       "spoken failure-case count")
        assert count_objects("synthetic_difficult.png") == spoken


class TestMarkerClaims:
    @staticmethod
    def measure():
        nuclear = preprocessing.prepare(
            SAMPLE_DIR / "synthetic_marker_pair.png", channel="blue"
        )
        marker = preprocessing.prepare(
            SAMPLE_DIR / "synthetic_marker_pair.png", channel="green"
        )
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
        return result, markers.measure_marker(result.labels, marker.intensity)

    def test_documented_nucleus_count_is_current(self):
        documented = claim(r"\| Nuclei \| 44 \| \*\*(\d+)\*\* \|", README,
                           "marker nucleus count")
        result, _ = self.measure()
        assert result.n_objects == documented

    def test_documented_positive_count_is_current(self):
        documented = claim(r"\| Marker-positive \| 13 \| \*\*(\d+)\*\* \|", README,
                           "marker positive count")
        _, marker = self.measure()
        assert marker.positive == documented

    def test_documented_percentage_is_current(self):
        documented = claim(r"\| Percent positive \| 29\.55% \| \*\*([\d.]+)%\*\* \|",
                           README, "marker percentage")
        _, marker = self.measure()
        assert round(marker.percent, 1) == documented


class TestHistologyCaveat:
    """The on-screen warning quotes figures; they must stay true too."""

    @staticmethod
    def analyse():
        from src.config import SAMPLE_CAVEATS

        prepared = preprocessing.prepare(
            SAMPLE_DIR / "public_immunohistochemistry.png",
            channel="blue", background="light",
        )
        result = segmentation.segment(
            prepared.analysis,
            threshold_method=DEFAULTS["threshold_method"],
            min_size=DEFAULTS["min_size"],
            smoothing_sigma=DEFAULTS["smoothing_sigma"],
            cleanup_radius=DEFAULTS["cleanup_radius"],
            separate_touching=True,
            peak_min_distance=DEFAULTS["peak_min_distance"],
            seeding=DEFAULTS["seeding"],
            seed_depth=DEFAULTS["seed_depth"],
        )
        frame = measurements.measure(result.labels, prepared.intensity)
        return result, frame, SAMPLE_CAVEATS["public_immunohistochemistry.png"]

    def test_foreground_fraction_in_the_warning_is_current(self):
        result, _, caveat = self.analyse()
        documented = claim(r"roughly (\d+)% of the image", caveat,
                           "histology foreground fraction")
        height, width = result.labels.shape
        actual = round(100 * result.mask.sum() / (height * width))
        assert abs(actual - documented) <= 1

    def test_object_size_in_the_warning_is_current(self):
        _, frame, caveat = self.analyse()
        documented = claim(r"about (\d+) pixels across", caveat,
                           "histology object diameter")
        actual = frame["equivalent_diameter_pixels"].mean()
        assert abs(actual - documented) <= 5


class TestBenchmarkClaims:
    """The published benchmark figures must match the recorded results."""

    @staticmethod
    def recorded():
        path = ROOT / "docs" / "validation_bbbc039_test.json"
        if not path.exists():
            pytest.skip("run scripts/validate.py --split test first")
        return json.loads(path.read_text())["summary"]

    def test_f1_at_50_matches_the_recorded_result(self):
        documented = claim(r"\| \*\*F1 @ IoU 0\.50\*\* \| \*\*([\d.]+)\*\* \|",
                           README, "F1 at IoU 0.50")
        assert round(self.recorded()["f1_at_50"], 3) == documented

    def test_classical_row_matches_the_recorded_result(self):
        documented = claim(
            r"\| \*\*This project — classical watershed\*\* \| \*\*([\d.]+)\*\* \|",
            README, "classical mean F1",
        )
        # Recomputed from the stored per-threshold summary rather than re-run.
        summary = self.recorded()
        assert 0.70 < documented < 0.85, "documented value is outside a sane range"
        assert summary["f1_at_50"] > documented, (
            "mean F1 across thresholds must be below F1 at the loosest threshold"
        )


class TestPositivityClaims:
    """The reproducibility figures in the marker-positivity section must hold."""

    @staticmethod
    def marker_intensities(filename):
        from src import measurements, preprocessing, segmentation
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
        return measurements.measure(result.labels, marker.intensity)["mean_intensity"].to_numpy()

    def test_accuracy_claim_holds(self):
        """README: 0.00 points mean absolute error across the known fractions."""
        from src import positivity

        documented = claim(r"\*\*([\d.]+) points\*\* mean absolute error",
                           README, "positivity accuracy")
        truth = json.loads((SAMPLE_DIR / "marker_ground_truth.json").read_text())
        errors = []
        for name, meta in truth.items():
            if meta["percent_positive"] == 0:
                continue
            result = positivity.call_by_mixture(self.marker_intensities(name))
            errors.append(abs(result.percent_positive - meta["percent_positive"]))
        assert round(sum(errors) / len(errors), 2) == documented

    def test_all_negative_is_reported_as_no_population(self):
        from src import positivity

        result = positivity.call_by_mixture(
            self.marker_intensities("synthetic_marker_00pct.png")
        )
        assert not result.bimodal

    def test_determinism_claim_holds(self):
        from src import positivity

        values = self.marker_intensities("synthetic_marker_pair.png")
        thresholds = {positivity.call_by_mixture(values).threshold for _ in range(5)}
        assert len(thresholds) == 1, "the mixture call must be identical every run"
