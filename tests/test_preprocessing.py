"""Image loading and preparation tests."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocessing
from src.preprocessing import ImageLoadError


def png_bytes(array: np.ndarray) -> bytes:
    return iio.imwrite("<bytes>", array, extension=".png")


class TestLoading:
    def test_loads_grayscale_png(self):
        original = np.arange(256, dtype=np.uint8).reshape(16, 16)
        loaded = preprocessing.load_image(io.BytesIO(png_bytes(original)))
        assert loaded.shape == (16, 16)

    def test_loads_rgb_png(self):
        original = np.zeros((10, 12, 3), dtype=np.uint8)
        loaded = preprocessing.load_image(io.BytesIO(png_bytes(original)))
        assert loaded.shape == (10, 12, 3)

    def test_accepts_a_numpy_array_directly(self):
        array = np.zeros((8, 8), dtype=np.uint8)
        assert preprocessing.load_image(array).shape == (8, 8)

    def test_stack_is_max_projected_to_two_dimensions(self):
        # (Z, H, W): a z-stack, not a colour image.
        stack = np.zeros((5, 20, 20), dtype=np.uint8)
        stack[2, 10, 10] = 200
        projected = preprocessing.load_image(stack)
        assert projected.shape == (20, 20)
        assert projected[10, 10] == 200

    def test_rgba_keeps_three_usable_channels(self):
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        assert preprocessing.select_channel(
            preprocessing.load_image(rgba), "red"
        ).shape == (10, 10)

    def test_garbage_bytes_raise_a_clear_error(self):
        with pytest.raises(ImageLoadError):
            preprocessing.load_image(io.BytesIO(b"this is not an image"))


class TestChannelSelection:
    @staticmethod
    def rgb_sample():
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[..., 0] = 200  # red
        image[..., 1] = 100  # green
        image[..., 2] = 50   # blue
        return image

    @pytest.mark.parametrize(
        "channel,expected", [("red", 200), ("green", 100), ("blue", 50)]
    )
    def test_selects_the_requested_channel(self, channel, expected):
        plane = preprocessing.select_channel(self.rgb_sample(), channel)
        assert plane.max() == pytest.approx(expected / 255.0, abs=0.01)

    def test_grayscale_of_rgb_is_two_dimensional(self):
        assert preprocessing.select_channel(self.rgb_sample(), "grayscale").ndim == 2

    def test_grayscale_image_ignores_channel_choice(self):
        gray = np.full((5, 5), 128, dtype=np.uint8)
        assert preprocessing.select_channel(gray, "red").shape == (5, 5)

    def test_invalid_channel_is_rejected(self):
        with pytest.raises(ValueError):
            preprocessing.select_channel(self.rgb_sample(), "magenta")


class TestBitDepth:
    def test_integer_images_convert_against_the_dtype_range(self):
        # Absolute, not relative: a 12-bit value stored in uint16 converts to
        # 4095/65535, so two images of different brightness stay different.
        image = np.zeros((10, 10), dtype=np.uint16)
        image[5, 5] = 4095
        plane = preprocessing.select_channel(image, "grayscale")
        assert plane.max() == pytest.approx(4095 / 65535, abs=1e-4)

    def test_uint8_converts_to_the_familiar_255_scale(self):
        image = np.full((4, 4), 128, dtype=np.uint8)
        assert preprocessing.select_channel(image, "grayscale").max() == pytest.approx(
            128 / 255, abs=1e-4
        )

    def test_a_dim_integer_image_is_still_segmentable(self):
        # The absolute conversion leaves a 12-bit image dark, so the analysis
        # plane must do the stretching or nothing would ever be detected.
        image = np.zeros((40, 40), dtype=np.uint16)
        image[15:25, 15:25] = 4095
        prepared = preprocessing.prepare(image)
        assert prepared.intensity.max() == pytest.approx(4095 / 65535, abs=1e-4)
        assert prepared.analysis.max() == pytest.approx(1.0, abs=0.01)

    def test_float_image_already_in_range_is_untouched(self):
        image = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        plane = preprocessing.select_channel(image, "grayscale")
        assert plane.min() == pytest.approx(0.0)
        assert plane.max() == pytest.approx(1.0)


class TestPrepare:
    @staticmethod
    def dim_image():
        """Bright spot at 40% intensity on a dark field."""
        image = np.zeros((40, 40), dtype=np.float32)
        image[15:25, 15:25] = 0.4
        return image

    def test_returns_three_planes_of_matching_shape(self):
        prepared = preprocessing.prepare(self.dim_image())
        assert prepared.analysis.shape == prepared.intensity.shape == (40, 40)

    def test_analysis_plane_is_contrast_stretched(self):
        prepared = preprocessing.prepare(self.dim_image())
        assert prepared.analysis.max() == pytest.approx(1.0, abs=0.01)

    def test_intensity_plane_is_not_contrast_stretched(self):
        # This is what keeps mean intensity comparable across a batch: a dim
        # image must stay dim in the measured plane.
        prepared = preprocessing.prepare(self.dim_image())
        assert prepared.intensity.max() == pytest.approx(0.4, abs=0.01)

    def test_two_images_of_different_brightness_stay_distinguishable(self):
        bright = self.dim_image() * 2.0
        dim_prepared = preprocessing.prepare(self.dim_image())
        bright_prepared = preprocessing.prepare(bright)
        assert bright_prepared.intensity.max() > dim_prepared.intensity.max()
        # ...while both are stretched to full range for segmentation.
        assert bright_prepared.analysis.max() == pytest.approx(
            dim_prepared.analysis.max(), abs=0.02
        )

    def test_light_background_is_inverted(self):
        image = np.ones((20, 20), dtype=np.float32)
        image[8:12, 8:12] = 0.0  # dark object on a light field
        prepared = preprocessing.prepare(image, background="light")
        assert prepared.analysis[10, 10] > prepared.analysis[0, 0]

    def test_invalid_background_is_rejected(self):
        with pytest.raises(ValueError):
            preprocessing.prepare(self.dim_image(), background="sideways")


class TestSmoothing:
    def test_zero_sigma_is_a_no_op(self):
        image = np.random.default_rng(0).random((20, 20)).astype(np.float32)
        assert np.allclose(preprocessing.smooth(image, 0), image)

    def test_smoothing_reduces_variance(self):
        image = np.random.default_rng(0).random((40, 40)).astype(np.float32)
        assert preprocessing.smooth(image, 2.0).var() < image.var()
