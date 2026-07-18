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

    def test_stack_array_is_max_projected_to_two_dimensions(self):
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


class TestMultiFrameFiles:
    """Reading *encoded* multi-page files, not arrays passed in directly.

    These go through the real decoder because that is where the bug was:
    imageio returns only page 0 of a multi-page file, so a z-stack was silently
    analysed as its first slice. A test that hands load_image a numpy array
    skips the decoder entirely and passes while the upload path is broken.
    """

    @staticmethod
    def encode(array, extension=".tiff") -> io.BytesIO:
        return io.BytesIO(iio.imwrite("<bytes>", array, extension=extension))

    def test_signal_on_a_later_slice_survives(self):
        stack = np.zeros((5, 32, 32), dtype=np.uint8)
        stack[3, 10:20, 10:20] = 200  # nothing at all on page 0
        loaded = preprocessing.load_image(self.encode(stack))
        assert loaded.shape == (32, 32)
        assert loaded.max() == 200, "z-stack collapsed to its first slice"

    def test_projection_takes_the_maximum_across_slices(self):
        # 5 slices, not 3 or 4: a leading axis of exactly 3 or 4 is ambiguous
        # with colour, and TIFF writers store it as component planes rather
        # than as pages.
        stack = np.zeros((5, 16, 16), dtype=np.uint8)
        stack[0, 5, 5] = 10
        stack[3, 5, 5] = 250
        assert preprocessing.load_image(self.encode(stack))[5, 5] == 250

    def test_a_four_plane_tiff_is_read_as_colour_not_as_pages(self):
        """Documents a real TIFF ambiguity rather than pretending it away.

        A uint8 array with a leading axis of 3 or 4 is written by TIFF as
        colour component planes, so it reads back as one colour image. That is
        the file's own declaration and the loader should honour it rather than
        guess that the user meant a z-stack.
        """
        planes = np.zeros((4, 16, 16), dtype=np.uint8)
        planes[2, 5, 5] = 250
        loaded = preprocessing.load_image(self.encode(planes))
        assert loaded.shape == (16, 16, 4)
        # Still usable: channel selection reaches the data.
        assert preprocessing.select_channel(loaded, "blue").max() > 0

    def test_multipage_colour_tiff_keeps_its_channels(self):
        stack = np.zeros((3, 16, 24, 3), dtype=np.uint8)
        stack[1, ..., 1] = 180
        loaded = preprocessing.load_image(self.encode(stack))
        assert loaded.shape == (16, 24, 3)
        assert loaded[..., 1].max() == 180

    @pytest.mark.parametrize(
        "array,extension,expected",
        [
            (np.full((16, 24), 7, np.uint8), ".png", (16, 24)),
            (np.full((16, 24, 3), 7, np.uint8), ".png", (16, 24, 3)),
            (np.full((16, 24, 4), 7, np.uint8), ".png", (16, 24, 4)),
            (np.full((16, 24), 7, np.uint8), ".tiff", (16, 24)),
            (np.full((16, 24), 777, np.uint16), ".tiff", (16, 24)),
        ],
    )
    def test_single_frame_formats_keep_their_shape(self, array, extension, expected):
        """Requesting all frames must not add a spurious axis to normal images."""
        assert preprocessing.load_image(self.encode(array, extension)).shape == expected

    def test_narrow_image_is_not_mistaken_for_a_colour_image(self):
        # A 3-pixel-wide grayscale image has a trailing dimension of 3, which a
        # shape-based guess would read as RGB.
        narrow = np.full((32, 3), 7, dtype=np.uint8)
        assert preprocessing.load_image(self.encode(narrow, ".png")).shape == (32, 3)

    def test_stack_survives_the_full_prepare_pipeline(self):
        stack = np.zeros((4, 40, 40), dtype=np.uint8)
        stack[2, 12:28, 12:28] = 220
        prepared = preprocessing.prepare(self.encode(stack))
        assert prepared.analysis.shape == (40, 40)
        assert prepared.intensity.max() > 0


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


class TestIlluminationCorrection:
    """Flattening a slowly varying background before thresholding."""

    @staticmethod
    def uneven_field(size=200, radius=9):
        """Identical bright spots on a background that fades across the frame."""
        image = np.zeros((size, size), dtype=np.float32)
        yy, xx = np.mgrid[0:size, 0:size]
        image += 0.55 * (xx / size)  # gradient: dark left, bright right
        for cx in (30, 100, 170):
            image[((yy - size // 2) ** 2 + (xx - cx) ** 2) <= radius**2] += 0.35
        return np.clip(image, 0, 1)

    def test_zero_radius_is_a_no_op(self):
        image = self.uneven_field()
        assert np.allclose(preprocessing.correct_illumination(image, 0), image)

    def test_flattens_a_background_gradient(self):
        image = self.uneven_field()
        corrected = preprocessing.correct_illumination(image, 25)
        # Compare background-only columns on the dark and bright sides.
        row = 10  # away from the objects
        assert abs(float(corrected[row, 20] - corrected[row, 180])) < abs(
            float(image[row, 20] - image[row, 180])
        )

    def test_objects_survive_correction(self):
        image = self.uneven_field()
        corrected = preprocessing.correct_illumination(image, 25)
        centre = 100
        for cx in (30, 100, 170):
            assert corrected[centre, cx] > corrected[10, cx], (
                "object at x=%d was removed by the correction" % cx
            )

    def test_radius_smaller_than_objects_erases_them(self):
        """Documents the failure mode: the radius must exceed the objects."""
        image = self.uneven_field(radius=20)
        corrected = preprocessing.correct_illumination(image, 3)
        assert corrected[100, 100] < 0.5

    def test_it_is_off_by_default(self):
        # The published BBBC039 scores assume no correction. If this default
        # changed, those numbers would no longer describe the shipped pipeline.
        from src.config import DEFAULTS

        assert DEFAULTS["background_radius"] == 0
