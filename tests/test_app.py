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
        assert find(impossible_threshold.metric, "Average area").value == "n/a"


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
