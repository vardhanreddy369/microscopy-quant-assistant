"""Generate the bundled sample images.

Run once; the resulting PNGs are committed so the app never needs a network
connection during a demo.

Three synthetic cases are produced on purpose:
  easy      - well separated nuclei, high signal-to-noise
  moderate  - nuclei that touch at their boundaries, the case watershed exists for
  difficult - dense, heavily overlapping, noisy, unevenly illuminated; the
              pipeline is expected to do poorly here, and showing that is the point

Because the generator places every nucleus itself, it knows the true count. Those
counts are written to ground_truth.json and used by scripts/tune_defaults.py to
measure accuracy instead of eyeballing it.

A real public microscopy image is saved alongside them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"


@dataclass
class Nucleus:
    y: float
    x: float
    radius: float
    brightness: float
    elongation: float = 1.0
    angle: float = 0.0


def _overlaps(placed: list[Nucleus], y: float, x: float, radius: float,
              min_factor: float) -> bool:
    """True if a nucleus here would sit closer than ``min_factor`` allows.

    ``min_factor`` is a multiple of the summed radii: above 1.0 leaves a visible
    gap, near 0.85 makes objects touch with a neck between them, and well below
    that produces the deep overlap that no distance-transform watershed can undo.
    """
    for other in placed:
        limit = (radius + other.radius) * min_factor
        if np.hypot(y - other.y, x - other.x) < limit:
            return True
    return False


def _draw(canvas: np.ndarray, nucleus: Nucleus) -> None:
    """Add one elliptical nucleus with a flat-ish centre and a defined edge."""
    height, width = canvas.shape
    reach = int(nucleus.radius * max(nucleus.elongation, 1.0) * 2.5) + 3
    y0, y1 = max(0, int(nucleus.y) - reach), min(height, int(nucleus.y) + reach)
    x0, x1 = max(0, int(nucleus.x) - reach), min(width, int(nucleus.x) + reach)
    if y0 >= y1 or x0 >= x1:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dy = yy - nucleus.y
    dx = xx - nucleus.x

    cos_a, sin_a = np.cos(nucleus.angle), np.sin(nucleus.angle)
    ry = (dy * cos_a + dx * sin_a) / (nucleus.radius * nucleus.elongation)
    rx = (-dy * sin_a + dx * cos_a) / nucleus.radius

    # A super-Gaussian gives a plateau then a fast roll-off, closer to a stained
    # nucleus than a plain Gaussian blob.
    profile = nucleus.brightness * np.exp(-((ry**2 + rx**2) ** 1.7))
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], profile)


def _render(placed: list[Nucleus], size: int, rng: np.random.Generator,
            noise: float, background: float, illumination: float = 0.0) -> np.ndarray:
    canvas = np.zeros((size, size), dtype=np.float32)
    for nucleus in placed:
        _draw(canvas, nucleus)

    image = canvas + background

    if illumination > 0:
        yy, xx = np.mgrid[0:size, 0:size]
        gradient = (0.6 * (xx / size) + 0.4 * (1.0 - yy / size)).astype(np.float32)
        image = image * (1.0 - illumination) + image * illumination * 2.0 * gradient
        image = image + illumination * 0.25 * gradient

    if noise > 0:
        # Signal-dependent noise, which is how a real detector behaves.
        image = image + rng.normal(0.0, noise, image.shape).astype(np.float32)
        image = image + rng.normal(0.0, noise * 0.6, image.shape) * np.sqrt(
            np.clip(image, 0, None)
        )

    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def make_easy(size: int = 512, seed: int = 7) -> tuple[np.ndarray, int]:
    """Well separated nuclei on a clean dark background."""
    rng = np.random.default_rng(seed)
    placed: list[Nucleus] = []

    attempts = 0
    while len(placed) < 32 and attempts < 6000:
        attempts += 1
        radius = rng.uniform(11.0, 17.0)
        y = rng.uniform(radius + 5, size - radius - 5)
        x = rng.uniform(radius + 5, size - radius - 5)
        if _overlaps(placed, y, x, radius, min_factor=1.45):
            continue
        placed.append(
            Nucleus(y, x, radius, rng.uniform(0.72, 0.95),
                    rng.uniform(1.0, 1.18), rng.uniform(0, np.pi))
        )

    return _render(placed, size, rng, noise=0.012, background=0.035), len(placed)


def make_moderate(size: int = 512, seed: int = 11) -> tuple[np.ndarray, int]:
    """Nuclei that touch at their boundaries: the case that needs watershed.

    Cluster members are placed just under the sum of their radii apart, so they
    share a boundary and merge into one connected component under a threshold,
    but still have two distinct distance-transform peaks.
    """
    rng = np.random.default_rng(seed)
    placed: list[Nucleus] = []

    def new_nucleus(y: float, x: float, radius: float) -> Nucleus:
        return Nucleus(y, x, radius, rng.uniform(0.72, 0.94),
                       rng.uniform(1.0, 1.15), rng.uniform(0, np.pi))

    for _ in range(9):
        seed_radius = rng.uniform(12.0, 16.0)
        seed_y = rng.uniform(80, size - 80)
        seed_x = rng.uniform(80, size - 80)
        if _overlaps(placed, seed_y, seed_x, seed_radius, min_factor=1.5):
            continue
        placed.append(new_nucleus(seed_y, seed_x, seed_radius))

        anchor_y, anchor_x, anchor_r = seed_y, seed_x, seed_radius
        for _ in range(int(rng.integers(1, 4))):
            radius = rng.uniform(12.0, 16.0)
            for _ in range(60):
                angle = rng.uniform(0, 2 * np.pi)
                # 0.82-0.95 of the summed radii: a clear touch, not a merge.
                distance = (anchor_r + radius) * rng.uniform(0.82, 0.95)
                y = anchor_y + distance * np.sin(angle)
                x = anchor_x + distance * np.cos(angle)
                if not (radius + 4 < y < size - radius - 4):
                    continue
                if not (radius + 4 < x < size - radius - 4):
                    continue
                # Must touch the anchor but not deeply overlap anything else.
                if _overlaps([n for n in placed if n is not placed[-1]],
                             y, x, radius, min_factor=0.80):
                    continue
                if _overlaps([placed[-1]], y, x, radius, min_factor=0.80):
                    continue
                placed.append(new_nucleus(y, x, radius))
                anchor_y, anchor_x, anchor_r = y, x, radius
                break

    attempts = 0
    while len(placed) < 34 and attempts < 4000:
        attempts += 1
        radius = rng.uniform(11.0, 16.0)
        y = rng.uniform(radius + 5, size - radius - 5)
        x = rng.uniform(radius + 5, size - radius - 5)
        if _overlaps(placed, y, x, radius, min_factor=1.35):
            continue
        placed.append(new_nucleus(y, x, radius))

    return _render(placed, size, rng, noise=0.02, background=0.05), len(placed)


def make_difficult(size: int = 512, seed: int = 23) -> tuple[np.ndarray, int]:
    """Dense, deeply overlapping, noisy, unevenly lit.

    This is the honest failure case. Nuclei are allowed to overlap far past the
    point where a distance transform can recover the individuals, brightness
    varies enough that no single global threshold suits every object, and small
    debris mimics real objects.
    """
    rng = np.random.default_rng(seed)
    placed: list[Nucleus] = []

    attempts = 0
    while len(placed) < 110 and attempts < 8000:
        attempts += 1
        radius = rng.uniform(7.0, 20.0)
        y = rng.uniform(radius, size - radius)
        x = rng.uniform(radius, size - radius)
        if _overlaps(placed, y, x, radius, min_factor=0.45):
            continue
        placed.append(
            Nucleus(y, x, radius, rng.uniform(0.28, 0.95),
                    rng.uniform(1.0, 1.9), rng.uniform(0, np.pi))
        )

    true_count = len(placed)

    # Debris: looks like small objects but is not, so it is excluded from the
    # ground-truth count on purpose.
    debris = [
        Nucleus(rng.uniform(0, size), rng.uniform(0, size),
                rng.uniform(2.0, 4.5), rng.uniform(0.3, 0.7))
        for _ in range(40)
    ]

    image = _render(placed + debris, size, rng, noise=0.075,
                    background=0.11, illumination=0.55)
    return image, true_count


def save_real_references() -> list[str]:
    """Save real public microscopy images from the scikit-image collection.

    They are written to disk here so the app reads local files at demo time and
    never depends on a network connection or a warm cache.
    """
    from skimage import data

    saved: list[str] = []
    for filename, loader in (
        ("public_human_mitosis.png", "human_mitosis"),
        ("public_immunohistochemistry.png", "immunohistochemistry"),
    ):
        try:
            image = np.asarray(getattr(data, loader)())
        except Exception as exc:  # noqa: BLE001 - optional, network may be absent
            print(f"  {loader} unavailable ({exc})")
            continue
        iio.imwrite(SAMPLE_DIR / filename, image)
        print(f"  wrote {filename}  shape={image.shape} dtype={image.dtype}")
        saved.append(filename)
    return saved


SOURCE_NOTES = """Sample data
===========

public_human_mitosis.png
    Real fluorescence microscopy image of human cells undergoing mitosis,
    nuclei stained and imaged on a dark background.
    Source: the scikit-image sample data collection (`skimage.data.human_mitosis`),
    which distributes it from https://gitlab.com/scikit-image/data.
    scikit-image is BSD-3-Clause licensed and provides this image for public
    demonstration and testing use.
    This is the primary demonstration image: real data, bright nuclei on a dark
    background, with both separated and touching nuclei present.

public_immunohistochemistry.png
    Real brightfield immunohistochemistry image (H-DAB stained tissue), RGB.
    Source: the scikit-image sample data collection
    (`skimage.data.immunohistochemistry`), BSD-3-Clause.
    Included because it is a light-background colour image, which exercises the
    channel-selection and background-mode controls. The classical nucleus
    pipeline is not tuned for brightfield histology and should not be read as
    validated for it.

Both public images are demonstration data. Neither is from any UCF laboratory,
and neither contains patient-identifying information.

synthetic_easy.png
synthetic_moderate.png
synthetic_difficult.png
    Generated programmatically by scripts/make_sample_data.py using fixed random
    seeds, so they are exactly reproducible. They are simulations, not real
    biological images, and exist to exercise three defined conditions:

      easy       well separated nuclei, high signal-to-noise
      moderate   nuclei touching at their boundaries; the case watershed addresses
      difficult  dense, deeply overlapping, noisy, unevenly illuminated

    Because the generator places each nucleus, the true object count is known
    and recorded in ground_truth.json. This is what makes it possible to report
    counting accuracy rather than assert it.

    The difficult image is included deliberately as a failure case. The
    classical pipeline does not handle it well, and that limitation is measured
    and reported rather than hidden. Its debris objects are excluded from the
    ground-truth count, so over-counting there is a real error.

No image in this directory came from a publication, a clinical source, or an
unpublished dataset.
"""


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating sample data...")

    ground_truth: dict[str, int] = {}
    for name, builder in (
        ("synthetic_easy.png", make_easy),
        ("synthetic_moderate.png", make_moderate),
        ("synthetic_difficult.png", make_difficult),
    ):
        image, count = builder()
        iio.imwrite(SAMPLE_DIR / name, image)
        ground_truth[name] = count
        print(f"  wrote {name}  shape={image.shape}  true objects={count}")

    save_real_references()

    (SAMPLE_DIR / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2) + "\n"
    )
    (SAMPLE_DIR / "source_information.txt").write_text(SOURCE_NOTES)
    print("  wrote ground_truth.json")
    print("  wrote source_information.txt")
    print("Done.")


if __name__ == "__main__":
    main()
