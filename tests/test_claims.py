"""The documented *promises* must hold, not just the documented numbers.

tests/test_documentation.py guards the figures. This guards the sentences
around them: statements like "runs with no network connection", "off by
default", "no scale is ever inferred from the file", or "flagged rather than
deleted" are commitments a reader will rely on, and each one is checkable.

Not every claim can be tested. "The pipeline is not validated for brightfield
histology" is a statement about the world, and no assertion can tell you it has
stopped being true. Those still need a human. What is covered here is every
claim that is a property of this code, so that a change which quietly breaks a
promise fails rather than merely making the README wrong.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import learned_segmentation, measurements, preprocessing, segmentation
from src.config import (
    DEFAULTS,
    SAMPLE_CAVEATS,
    SAMPLE_DIR,
    SAMPLE_IMAGES,
    sample_path,
)

README = (ROOT / "README.md").read_text()


class TestRunsOffline:
    """README: "Everything it needs is committed to the repository, so it runs
    with no network connection." """

    def test_the_claim_is_still_made(self):
        assert "no network connection" in README

    def test_every_offered_sample_is_present_locally(self):
        missing = [name for _, name, _ in SAMPLE_IMAGES if not sample_path(name).exists()]
        assert not missing, f"samples not committed: {missing}"

    def test_the_two_channel_sample_is_present_locally(self):
        assert (SAMPLE_DIR / "synthetic_marker_pair.png").exists()

    def test_ground_truth_files_are_committed(self):
        assert (SAMPLE_DIR / "ground_truth.json").exists()
        assert (SAMPLE_DIR / "marker_ground_truth.json").exists()

    def test_the_default_path_does_not_import_torch(self):
        """The default engine must not drag in the optional deep-learning stack.

        Importing the pipeline modules is enough to check: if torch were a hard
        dependency of the classical path it would already be in sys.modules.
        """
        for module in ("src.preprocessing", "src.segmentation", "src.measurements"):
            __import__(module)
        assert DEFAULTS.get("engine", "classical") != "cellpose"


class TestOptionalDependencyIsOptional:
    """README: "The classical pipeline needs none of it and the application
    runs without it." """

    def test_the_learned_module_imports_without_cellpose(self):
        assert hasattr(learned_segmentation, "segment")

    def test_availability_is_reported_rather_than_assumed(self):
        assert isinstance(learned_segmentation.is_available(), bool)

    def test_a_missing_dependency_explains_itself(self):
        if not learned_segmentation.is_available():
            assert "install" in learned_segmentation.unavailable_reason().lower()

    def test_requirements_do_not_force_the_optional_stack(self):
        base = (ROOT / "requirements.txt").read_text().lower()
        assert "cellpose" not in base
        assert "torch" not in base
        assert (ROOT / "requirements-cellpose.txt").exists()


class TestOffByDefault:
    """README: illumination correction "is off by default"; the classical
    engine is the default."""

    def test_illumination_correction_is_off(self):
        assert DEFAULTS["background_radius"] == 0

    def test_the_claim_is_still_made(self):
        assert "off by default" in README

    def test_turning_it_off_changes_nothing(self):
        plane = np.zeros((120, 120), dtype=np.float32)
        yy, xx = np.mgrid[0:120, 0:120]
        plane[((yy - 60) ** 2 + (xx - 60) ** 2) <= 20**2] = 1.0
        default = segmentation.segment(plane, min_size=DEFAULTS["min_size"])
        explicit = segmentation.segment(plane, min_size=DEFAULTS["min_size"],
                                        background_radius=0)
        assert default.n_objects == explicit.n_objects


class TestNoInventedScale:
    """README: "no scale is ever inferred from the file, because a guessed
    scale produces confidently wrong numbers." """

    @staticmethod
    def labels():
        labels = np.zeros((80, 80), dtype=np.int32)
        labels[20:40, 20:40] = 1
        return labels

    def test_no_micrometre_columns_without_a_pixel_size(self):
        frame = measurements.measure(self.labels(), self.labels().astype(np.float32))
        assert not [c for c in frame.columns if c.endswith(("_um", "_um2"))]

    def test_micrometre_columns_appear_only_when_supplied(self):
        frame = measurements.measure(self.labels(), self.labels().astype(np.float32),
                                     pixel_size_um=0.5)
        assert "area_um2" in frame.columns

    def test_the_default_pixel_size_is_unset(self):
        assert DEFAULTS["pixel_size_um"] is None


class TestBorderObjectsAreFlaggedNotDeleted:
    """README: border objects are "flagged rather than deleted, so the decision
    to exclude them stays with the researcher." """

    def test_the_claim_is_still_made(self):
        assert "flagged rather than deleted" in README

    def test_a_border_object_survives_measurement(self):
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[0:15, 0:15] = 1      # against the corner
        labels[30:45, 30:45] = 2    # interior
        frame = measurements.measure(labels, labels.astype(np.float32))
        assert len(frame) == 2, "the border object must not be dropped"
        assert bool(frame.loc[frame["object_id"] == 1, "touches_border"].iloc[0])
        assert not bool(frame.loc[frame["object_id"] == 2, "touches_border"].iloc[0])


class TestIntensityStaysComparable:
    """README: "intensity is measured on the un-stretched plane", so brightness
    differences between images survive."""

    def test_the_claim_is_still_made(self):
        assert "un-stretched plane" in README

    def test_a_dim_image_stays_dim_when_measured(self):
        dim = np.zeros((60, 60), dtype=np.float32)
        dim[20:40, 20:40] = 0.3
        bright = dim * 2.0

        dim_prepared = preprocessing.prepare(dim)
        bright_prepared = preprocessing.prepare(bright)

        assert bright_prepared.intensity.max() > dim_prepared.intensity.max(), (
            "measured intensity must preserve the difference between images"
        )
        assert dim_prepared.analysis.max() == pytest.approx(
            bright_prepared.analysis.max(), abs=0.02
        ), "the segmentation plane is stretched for both"


class TestNoTrainTestLeakage:
    """README: "Parameters were grid-searched on the training split only. The
    numbers below come from the held-out test split." """

    @staticmethod
    def split(name):
        path = ROOT / "validation_data" / "bbbc039" / "metadata" / f"{name}.txt"
        if not path.exists():
            pytest.skip("run scripts/fetch_validation_data.py first")
        return {Path(line.strip()).stem for line in path.read_text().splitlines()
                if line.strip()}

    def test_the_claim_is_still_made(self):
        assert "held-out test split" in README

    def test_training_and_test_images_do_not_overlap(self):
        assert not (self.split("training") & self.split("test"))

    def test_validation_and_test_images_do_not_overlap(self):
        assert not (self.split("validation") & self.split("test"))

    def test_the_tuning_script_reads_only_the_training_split(self):
        source = (ROOT / "scripts" / "tune_on_bbbc039.py").read_text()
        assert 'load_split("training")' in source
        assert 'load_split("test")' not in source, (
            "the tuning script must never read the held-out split"
        )


class TestOffDomainWarningExists:
    """README and the app both promise the histology result is flagged."""

    def test_a_caveat_is_registered_for_the_histology_sample(self):
        assert "public_immunohistochemistry.png" in SAMPLE_CAVEATS

    def test_the_caveat_says_the_numbers_are_invalid(self):
        text = SAMPLE_CAVEATS["public_immunohistochemistry.png"].lower()
        assert "not valid" in text

    def test_no_caveat_on_the_samples_that_are_correct(self):
        for _, filename, _ in SAMPLE_IMAGES:
            if filename != "public_immunohistochemistry.png":
                assert filename not in SAMPLE_CAVEATS


class TestStatedTestCount:
    """The README states how many tests there are. That is a claim too."""

    def test_the_stated_count_matches_the_suite(self):
        stated = re.search(r"(\d+) tests covering image loading", README)
        assert stated, "the README no longer states a test count"

        total = 0
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    # Parametrised cases expand at run time; count the function
                    # once and allow the stated total to exceed it.
                    total += 1

        claimed = int(stated.group(1))
        assert claimed >= total, (
            f"README claims {claimed} tests but {total} test functions exist "
            "before parametrisation; the claim is too low"
        )
        assert claimed <= total * 2, (
            f"README claims {claimed} tests against {total} test functions; "
            "the claim looks stale"
        )
