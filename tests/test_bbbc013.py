"""Real-data validation on BBBC013, and tests for its harness.

The BBBC013 images are large and gitignored, so the data-dependent tests skip
when they are absent. The pure helpers — the rank correlation and the platemap
parser — are tested against hand-computable inputs and always run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_bbbc013 as harness

DATA_PRESENT = (ROOT / "validation_data" / "bbbc013" / "BBBC013_v1_images_bmp").exists()
needs_data = pytest.mark.skipif(
    not DATA_PRESENT, reason="run scripts/fetch_bbbc013.py first"
)


class TestSpearman:
    """The rank correlation is computed without scipy, so it is checked here."""

    def test_perfect_increasing_is_one(self):
        assert harness.spearman(np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40])) == pytest.approx(1.0)

    def test_perfect_decreasing_is_minus_one(self):
        assert harness.spearman(np.array([1, 2, 3, 4]), np.array([40, 30, 20, 10])) == pytest.approx(-1.0)

    def test_monotonic_but_nonlinear_is_still_one(self):
        # Spearman is rank-based, so a curved-but-monotonic relation scores 1.
        assert harness.spearman(np.array([1, 2, 3, 4]), np.array([1, 4, 9, 16])) == pytest.approx(1.0)

    def test_no_relation_is_near_zero(self):
        x = np.array([1, 2, 3, 4, 5, 6])
        y = np.array([3, 1, 4, 1, 5, 2])
        assert abs(harness.spearman(x, y)) < 0.6


class TestPlatemap:
    @needs_data
    def test_parses_ninety_six_wells(self):
        platemap = harness.load_platemap()
        assert len(platemap) == 96

    @needs_data
    def test_dose_zero_negative_controls_exist(self):
        platemap = harness.load_platemap()
        assert any(dose == 0.0 for dose in platemap.values())

    @needs_data
    def test_positive_control_dose_is_present(self):
        # 150 nM Wortmannin is the assay's designated positive control.
        platemap = harness.load_platemap()
        assert 150.0 in platemap.values()


@needs_data
class TestRealDoseResponse:
    """On real data, the measured positive fraction must track the known dose."""

    @staticmethod
    @pytest.fixture(scope="class")
    def wortmannin():
        return harness.evaluate_drug("Wortmannin", "ABCD", harness.load_platemap())

    @staticmethod
    @pytest.fixture(scope="class")
    def ly294002():
        return harness.evaluate_drug("LY294002", "EFGH", harness.load_platemap())

    def test_dose_response_is_strongly_positive_wortmannin(self, wortmannin):
        assert wortmannin["mixture_spearman"] > 0.5

    def test_dose_response_is_strongly_positive_ly294002(self, ly294002):
        assert ly294002["mixture_spearman"] > 0.5

    def test_controls_separate_wortmannin(self, wortmannin):
        # The top dose must read far more positive than the negative control.
        assert wortmannin["mixture_max"] - wortmannin["mixture_negative"] > 25

    def test_controls_separate_ly294002(self, ly294002):
        assert ly294002["mixture_max"] - ly294002["mixture_negative"] > 25

    def test_top_dose_is_majority_positive(self, wortmannin, ly294002):
        assert wortmannin["mixture_max"] > 50
        assert ly294002["mixture_max"] > 50

    def test_control_anchoring_improves_the_dose_response(self, wortmannin, ly294002):
        """Normalising to the plate's own negative controls must help, not hurt."""
        assert wortmannin["anchored_spearman"] > wortmannin["mixture_spearman"]
        assert ly294002["anchored_spearman"] > ly294002["mixture_spearman"]

    def test_control_anchoring_pins_negatives_near_zero(self, wortmannin, ly294002):
        assert wortmannin["anchored_negative"] < 3.0
        assert ly294002["anchored_negative"] < 3.0
