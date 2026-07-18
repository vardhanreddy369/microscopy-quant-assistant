"""Tests for the shared interactive size policy.

The policy lives in config rather than in the UI because the single-image and
batch paths must agree. They previously did not: the batch loop had no size
guard at all, so one oversized file could stall a whole run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.config import (
    MAX_PIXELS,
    SIZE_OK,
    SIZE_SLOW,
    SIZE_TOO_LARGE,
    SLOW_PIXELS,
    size_verdict,
)


class TestSizeVerdict:
    def test_ordinary_image_is_ok(self):
        verdict, message = size_verdict(520, 696)
        assert verdict == SIZE_OK
        assert message == ""

    def test_large_image_is_flagged_slow_but_allowed(self):
        verdict, message = size_verdict(3000, 3000)  # 9 MP
        assert verdict == SIZE_SLOW
        assert "3000x3000" in message

    def test_enormous_image_is_rejected(self):
        verdict, message = size_verdict(10_000, 10_000)  # 100 MP
        assert verdict == SIZE_TOO_LARGE
        assert "exceeds" in message

    def test_rejection_message_tells_the_user_what_to_do(self):
        _, message = size_verdict(10_000, 10_000)
        assert "crop" in message.lower() or "downscale" in message.lower()

    def test_message_reports_width_by_height_not_height_by_width(self):
        _, message = size_verdict(100, 90_000)  # tall*wide, 9 MP
        assert "90000x100" in message

    @pytest.mark.parametrize(
        "pixels,expected",
        [
            (SLOW_PIXELS, SIZE_OK),
            (SLOW_PIXELS + 1, SIZE_SLOW),
            (MAX_PIXELS, SIZE_SLOW),
            (MAX_PIXELS + 1, SIZE_TOO_LARGE),
        ],
    )
    def test_boundaries_are_exclusive(self, pixels, expected):
        assert size_verdict(1, pixels)[0] == expected

    def test_thresholds_are_ordered(self):
        assert 0 < SLOW_PIXELS < MAX_PIXELS


class TestPolicyIsShared:
    def test_lowering_the_limit_affects_the_verdict(self):
        """size_verdict reads the module global, so tests can lower the cap.

        This is what makes the batch guard testable without generating a real
        40-megapixel image.
        """
        original = config.MAX_PIXELS
        try:
            config.MAX_PIXELS = 100
            assert size_verdict(50, 50)[0] == SIZE_TOO_LARGE
        finally:
            config.MAX_PIXELS = original
        assert size_verdict(50, 50)[0] != SIZE_TOO_LARGE
