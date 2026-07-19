"""End-to-end tests driving the Streamlit app headlessly.

These cover the failure modes that would actually break a live demonstration:
an empty result, an unusual image, and the batch path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = str(ROOT / "app.py")
TIMEOUT = 300


def find(collection, label):
    """Locate a widget by its visible label."""
    for widget in collection:
        if widget.label == label:
            return widget
    raise KeyError(f"no widget labelled {label!r} in {[w.label for w in collection]}")


def find_starting(collection, prefix):
    """Locate a widget whose label starts with ``prefix``.

    Metric labels carry their unit, e.g. "Average area (px)" or "(µm²)", so
    tests match the stable part rather than a unit that depends on settings.
    """
    for widget in collection:
        if widget.label.startswith(prefix):
            return widget
    raise KeyError(
        f"no widget labelled {prefix!r}* in {[w.label for w in collection]}"
    )


@pytest.fixture(scope="module")
def default_run():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    return app


class TestDefaultLaunch:
    def test_app_starts_without_exception(self, default_run):
        assert not default_run.exception

    def test_it_shows_a_result_immediately(self, default_run):
        """The app must open on a finished analysis, not an empty screen."""
        objects = find(default_run.metric, "Objects detected")
        assert int(objects.value.replace(",", "")) > 0

    def test_it_shows_the_validation_disclaimer(self, default_run):
        assert any("validation" in w.value.lower() for w in default_run.warning)

    def test_no_error_banner_on_the_default_sample(self, default_run):
        assert not default_run.error


class TestEmptyResult:
    """A threshold above every pixel must guide the user, not crash or lie."""

    @staticmethod
    @pytest.fixture(scope="class")
    def impossible_threshold():
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        find(app.sidebar.radio, "Threshold").set_value("Manual").run()
        find(app.sidebar.slider, "Manual threshold").set_value(1.0).run()
        return app

    def test_does_not_crash(self, impossible_threshold):
        assert not impossible_threshold.exception

    def test_reports_zero_objects(self, impossible_threshold):
        assert find(impossible_threshold.metric, "Objects detected").value == "0"

    def test_explains_how_to_recover(self, impossible_threshold):
        message = " ".join(e.value for e in impossible_threshold.error).lower()
        assert "no objects" in message
        assert "threshold" in message

    def test_area_metrics_degrade_gracefully(self, impossible_threshold):
        assert find_starting(impossible_threshold.metric, "Average area").value == "n/a"


class TestMinimumSizeTooLarge:
    def test_oversized_minimum_removes_everything_without_crashing(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        find(app.sidebar.slider, "Minimum object size (pixels)").set_value(2000).run()
        assert not app.exception
        assert find(app.metric, "Objects detected").value == "0"


class TestSampleSwitching:
    def test_rgb_light_background_sample_applies_its_overrides(self):
        """Switching to the brightfield RGB image must not analyse it with
        fluorescence settings."""
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        selector = find(app.sidebar.selectbox, "or use a demonstration image")
        selector.set_value("Immunohistochemistry (real, brightfield)").run()

        assert not app.exception
        assert app.session_state["background"] == "light"
        assert app.session_state["channel"] == "blue"

    def test_difficult_sample_still_produces_a_result(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        selector = find(app.sidebar.selectbox, "or use a demonstration image")
        selector.set_value("Synthetic - difficult (known failure case)").run()

        assert not app.exception
        assert int(find(app.metric, "Objects detected").value.replace(",", "")) > 0

    def test_touching_sample_counts_exactly(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        selector = find(app.sidebar.selectbox, "or use a demonstration image")
        selector.set_value("Synthetic - touching").run()

        assert not app.exception
        assert find(app.metric, "Objects detected").value == "34"


class TestWatershedToggle:
    def test_disabling_separation_lowers_the_count(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        find(app.sidebar.selectbox, "or use a demonstration image").set_value(
            "Synthetic - touching"
        ).run()
        with_watershed = int(find(app.metric, "Objects detected").value)

        find(app.sidebar.checkbox, "Separate touching objects (watershed)").set_value(
            False
        ).run()
        without = int(find(app.metric, "Objects detected").value)

        assert without < with_watershed


class TestBatchMode:
    def test_batch_over_the_bundled_samples(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        find(app.sidebar.radio, "Mode").set_value("Batch (multiple images)").run()
        assert not app.exception

        # The batch is deliberately gated behind a button, since it is the
        # expensive path.
        find(app.button, "Run batch analysis").click().run()
        assert not app.exception

        total = find(app.metric, "Total objects across all images")
        assert int(total.value.replace(",", "")) > 0

    def test_batch_results_carry_a_source_image_column(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        find(app.sidebar.radio, "Mode").set_value("Batch (multiple images)").run()
        find(app.button, "Run batch analysis").click().run()

        frames = [element.value for element in app.dataframe]
        assert frames, "batch mode produced no tables"
        assert any("source_image" in frame.columns for frame in frames)


class TestBatchSizeGuard:
    """The batch path must apply the same size limit as the single-image path.

    It previously applied none, so one oversized upload could stall an entire
    run. The cap is lowered here rather than generating a real 40-megapixel
    image, which would make the suite slow for no extra coverage.
    """

    @staticmethod
    def run_batch_with_cap(max_pixels: int):
        from src import config

        original = config.MAX_PIXELS
        config.MAX_PIXELS = max_pixels
        try:
            app = AppTest.from_file(APP, default_timeout=TIMEOUT)
            app.run()
            find(app.sidebar.radio, "Mode").set_value("Batch (multiple images)").run()
            find(app.button, "Run batch analysis").click().run()
            return app
        finally:
            config.MAX_PIXELS = original

    def test_oversized_images_are_skipped_rather_than_analysed(self):
        app = self.run_batch_with_cap(1_000)  # every bundled sample exceeds this
        assert not app.exception

        summaries = [f for f in (e.value for e in app.dataframe) if "error" in f.columns]
        assert summaries, "no per-image summary table was produced"
        errors = summaries[0]["error"].astype(str)
        assert errors.str.contains("skipped").all(), (
            "batch mode analysed images that exceed the size limit"
        )

    def test_skipping_does_not_abort_the_whole_batch(self):
        app = self.run_batch_with_cap(1_000)
        summaries = [f for f in (e.value for e in app.dataframe) if "error" in f.columns]
        # Every image still gets a row, rather than the run dying on the first.
        assert len(summaries[0]) >= 5

    def test_normal_sizes_still_analyse(self):
        app = self.run_batch_with_cap(40_000_000)
        assert not app.exception
        total = find(app.metric, "Total objects across all images")
        assert int(total.value.replace(",", "")) > 0


class TestOffDomainSampleCaveat:
    """A result that is wrong must say so on screen, not just in a caption.

    The brightfield histology sample produces a tidy object count and a full
    measurement table while segmenting whole tissue regions rather than nuclei.
    A number that looks valid and is not is the failure mode this project is
    supposed to avoid.
    """

    def test_the_histology_sample_warns_that_it_is_invalid(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        find(app.sidebar.selectbox, "or use a demonstration image").set_value(
            "Immunohistochemistry (real, brightfield)"
        ).run()

        assert not app.exception
        errors = " ".join(e.value for e in app.error).lower()
        assert "not valid" in errors
        assert "nucleus" in errors or "nuclei" in errors

    def test_the_fluorescence_samples_do_not_warn(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        # The default sample is a good result; it must not be flagged.
        assert not [e for e in app.error if "not valid" in e.value.lower()]

    def test_switching_back_clears_the_warning(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        selector = find(app.sidebar.selectbox, "or use a demonstration image")
        selector.set_value("Immunohistochemistry (real, brightfield)").run()
        assert any("not valid" in e.value.lower() for e in app.error)

        find(app.sidebar.selectbox, "or use a demonstration image").set_value(
            "Synthetic - easy"
        ).run()
        assert not [e for e in app.error if "not valid" in e.value.lower()]


class TestSegmentationEngineSelector:
    """The engine choice must be offered, and must degrade safely."""

    def test_engine_control_exists(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        engine = find(app.sidebar.radio, "Engine")
        assert any("Classical" in option for option in engine.options)
        assert any("Cellpose" in option for option in engine.options)

    def test_classical_is_the_default(self):
        """The default must need no network, no GPU and no 1 GB download."""
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        assert find(app.sidebar.radio, "Engine").value.startswith("Classical")

    def test_selecting_cellpose_never_crashes(self):
        """Whether or not Cellpose is installed, choosing it must be safe.

        When it is missing the app must explain and fall back rather than
        raising, since the option is visible in every install.
        """
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        find(app.sidebar.radio, "Engine").set_value(
            "Cellpose (pretrained model)"
        ).run()
        assert not app.exception
        assert int(find(app.metric, "Objects detected").value.replace(",", "")) >= 0


class TestMarkerMode:
    """Percent-marker-positive is the shape of a real fluorescence endpoint."""

    @staticmethod
    @pytest.fixture(scope="class")
    def marker_run():
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        find(app.sidebar.radio, "Mode").set_value(
            "Marker positive (two-channel)"
        ).run()
        return app

    def test_it_runs_without_error(self, marker_run):
        assert not marker_run.exception
        assert not marker_run.error

    def test_it_reports_a_percentage(self, marker_run):
        value = find(marker_run.metric, "Percent positive").value
        assert value.endswith("%")

    def test_the_percentage_matches_the_known_truth(self, marker_run):
        """The bundled sample has a known positive fraction of 29.55%."""
        value = float(find(marker_run.metric, "Percent positive").value.rstrip("%"))
        assert value == pytest.approx(29.55, abs=1.5)

    def test_the_denominator_is_every_nucleus(self, marker_run):
        total = int(find(marker_run.metric, "Total nuclei").value.replace(",", ""))
        positive = int(find(marker_run.metric, "Positive").value.replace(",", ""))
        assert total == 44
        assert positive < total, "the denominator must include negative cells"

    def test_channel_controls_are_offered(self, marker_run):
        nuclear = find(marker_run.sidebar.selectbox, "Nuclear channel")
        marker = find(marker_run.sidebar.selectbox, "Marker channel")
        assert nuclear.value == "blue"
        assert marker.value == "green"

    def test_manual_threshold_changes_the_result(self):
        app = AppTest.from_file(APP, default_timeout=TIMEOUT)
        app.run()
        find(app.sidebar.radio, "Mode").set_value(
            "Marker positive (two-channel)"
        ).run()
        automatic = int(find(app.metric, "Positive").value)

        find(app.sidebar.radio, "Positivity threshold").set_value("Manual").run()
        find(app.sidebar.slider, "Marker threshold (0-255)").set_value(250.0).run()
        assert not app.exception
        # Almost nothing reaches 250, so the count must fall.
        assert int(find(app.metric, "Positive").value) < automatic
