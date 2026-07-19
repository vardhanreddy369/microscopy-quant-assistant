"""Search for a better classical pipeline, scored on the BBBC039 training split.

The published classical baselines on this benchmark are CellProfiler at 0.790
(basic) and 0.811 (advanced), measured as mean F1 across IoU 0.50-0.95. The
shipped watershed scores 0.770. This script tests whether that gap can be closed
from two directions the error profile points at:

  merges      258 against 87 splits -> the seeding is too conservative
  boundaries  mean matched IoU 0.886 against Cellpose's 0.930 -> a single global
              threshold puts every boundary in roughly, but not exactly, the
              right place, and mean F1 punishes that hard at IoU 0.90 and 0.95

Everything here is measured on the TRAINING split. The test split is not touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import h_maxima, remove_small_objects
from skimage.segmentation import relabel_sequential, watershed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocessing
from src.validation import DatasetScore, decode_colored_mask, score_image

DATA = Path(__file__).resolve().parents[1] / "validation_data" / "bbbc039"
MIN_SIZE = 60


def foreground(plane, sigma=0.6):
    blurred = ndi.gaussian_filter(plane, sigma) if sigma else plane
    mask = blurred > threshold_otsu(blurred)
    mask = ndi.binary_fill_holes(mask)
    return remove_small_objects(mask, max_size=MIN_SIZE - 1), blurred


# --- seeding strategies -------------------------------------------------

def seeds_peak(distance, mask, min_distance=9):
    """The shipped approach: one global exclusion radius."""
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask,
                            exclude_border=False)
    markers = np.zeros(distance.shape, dtype=bool)
    if coords.size:
        markers[tuple(coords.T)] = True
    return ndi.label(markers)[0]


def seeds_hmaxima(distance, mask, depth=2.0):
    """Keep only maxima that stand at least ``depth`` above their surroundings.

    Shallow bumps on a ragged distance map become seeds under plain peak
    finding; requiring a minimum prominence removes them without needing a
    global spacing assumption.
    """
    peaks = h_maxima(distance, depth) * mask
    return ndi.label(peaks)[0]


def seeds_adaptive(distance, mask, factor=0.62, floor=3):
    """Suppression radius proportional to each candidate's own object radius.

    A single global ``min_distance`` cannot fit an image whose nuclei vary in
    size: set it for the large nuclei and small touching pairs merge; set it for
    the small ones and large nuclei shatter. The distance transform already
    estimates each object's radius at its own peak, so each candidate can
    suppress its neighbours over a radius scaled to itself.

    Candidates are taken brightest-first, so the largest object in a cluster
    claims its territory before smaller ones compete for it.
    """
    coords = peak_local_max(distance, min_distance=floor, labels=mask,
                            exclude_border=False)
    if coords.size == 0:
        return np.zeros(distance.shape, dtype=np.int32)

    values = distance[tuple(coords.T)]
    order = np.argsort(-values)
    coords, values = coords[order], values[order]

    kept = []
    kept_xy = np.empty((0, 2))
    for point, radius in zip(coords, values):
        if kept_xy.shape[0]:
            gap = np.hypot(*(kept_xy - point).T)
            if np.any(gap < max(floor, factor * radius)):
                continue
        kept.append(point)
        kept_xy = np.vstack([kept_xy, point])

    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(np.array(kept).T)] = True
    return ndi.label(markers)[0]


# --- boundary refinement ------------------------------------------------

def refine_boundaries(labels, plane, band=3, weight=0.5):
    """Re-cut each object's edge against a threshold local to that object.

    One global threshold sits between the image's bright and dim objects, so
    every boundary lands slightly inside the bright nuclei and slightly outside
    the dim ones. Mean F1 averages over IoU up to 0.95, where a systematic
    one-pixel bias is expensive.

    For each object a local cut-off is taken between its own interior intensity
    and the intensity just outside it, and only pixels in a thin band around the
    existing boundary are reconsidered, so the object cannot move or merge.
    """
    if labels.max() == 0:
        return labels

    refined = labels.copy()
    objects = ndi.find_objects(labels)
    for index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        pad = tuple(
            slice(max(0, s.start - band - 1), min(d, s.stop + band + 1))
            for s, d in zip(slices, labels.shape)
        )
        window = labels[pad]
        values = plane[pad]
        inside = window == index
        if not inside.any():
            continue

        dilated = ndi.binary_dilation(inside, iterations=band)
        halo = dilated & (window == 0)
        if not halo.any():
            continue

        interior = float(np.median(values[inside]))
        outside = float(np.median(values[halo]))
        if interior <= outside:
            continue
        cut = outside + weight * (interior - outside)

        # Only the band around the current edge is reconsidered.
        eroded = ndi.binary_erosion(inside, iterations=band)
        candidate = (dilated & (values >= cut)) | eroded
        candidate = candidate & (dilated & ((window == 0) | inside))
        candidate = ndi.binary_fill_holes(candidate)

        keep = ndi.label(candidate)[0]
        if keep.max() > 1:
            centre = keep[eroded] if eroded.any() else keep[inside]
            centre = centre[centre > 0]
            if centre.size == 0:
                continue
            candidate = keep == np.bincount(centre).argmax()

        region = refined[pad]
        region[inside] = 0
        region[candidate & (region == 0)] = index
        refined[pad] = region
    return refined


# --- assembly -----------------------------------------------------------

def run(plane, seeding="peak", refine=False, **kwargs):
    mask, blurred = foreground(plane)
    if not mask.any():
        return np.zeros(plane.shape, dtype=np.int32)

    distance = ndi.distance_filter = ndi.distance_transform_edt(mask)
    distance = ndi.gaussian_filter(distance, 1.0)

    if seeding == "peak":
        markers = seeds_peak(distance, mask, kwargs.get("min_distance", 9))
    elif seeding == "hmaxima":
        markers = seeds_hmaxima(distance, mask, kwargs.get("depth", 2.0))
    else:
        markers = seeds_adaptive(distance, mask, kwargs.get("factor", 0.62))

    labels = watershed(-distance, markers, mask=mask)
    labels = remove_small_objects(labels, max_size=MIN_SIZE - 1)

    if refine:
        labels = refine_boundaries(labels, blurred, kwargs.get("band", 3),
                                   kwargs.get("weight", 0.5))
        labels = remove_small_objects(labels, max_size=MIN_SIZE - 1)

    return relabel_sequential(labels)[0].astype(np.int32)


def evaluate(stems, **kwargs):
    scores = []
    for stem in stems:
        prepared = preprocessing.prepare(DATA / "images" / f"{stem}.tif")
        labels = run(prepared.analysis, **kwargs)
        truth = decode_colored_mask(iio.imread(DATA / "masks" / f"{stem}.png"))
        scores.append(score_image(labels, truth))
    result = DatasetScore(images=scores)
    mean_f1 = float(np.mean([result.f1_at(t) for t in result.thresholds]))
    summary = result.summary()
    return mean_f1, summary


def main() -> int:
    stems = [
        Path(line.strip()).stem
        for line in (DATA / "metadata" / "training.txt").read_text().splitlines()
        if line.strip()
    ]
    print(f"Training split: {len(stems)} images. Test split untouched.\n")
    print(f"  {'variant':44s} {'meanF1':>7} {'F1@50':>7} {'IoU':>6} {'split':>6} {'merge':>6}")

    trials = [
        ("shipped: peak d=9", dict(seeding="peak", min_distance=9)),
        ("peak d=7", dict(seeding="peak", min_distance=7)),
        ("h-maxima depth=1.5", dict(seeding="hmaxima", depth=1.5)),
        ("h-maxima depth=2.5", dict(seeding="hmaxima", depth=2.5)),
        ("adaptive factor=0.55", dict(seeding="adaptive", factor=0.55)),
        ("adaptive factor=0.70", dict(seeding="adaptive", factor=0.70)),
        ("shipped + refine", dict(seeding="peak", min_distance=9, refine=True)),
        ("adaptive 0.62 + refine", dict(seeding="adaptive", factor=0.62, refine=True)),
    ]

    best = None
    for name, kwargs in trials:
        mean_f1, summary = evaluate(stems, **kwargs)
        print(f"  {name:44s} {mean_f1:7.4f} {summary['f1_at_50']:7.4f} "
              f"{summary['mean_matched_iou']:6.3f} {summary['split_errors']:6d} "
              f"{summary['merge_errors']:6d}")
        if best is None or mean_f1 > best[0]:
            best = (mean_f1, name, kwargs)

    print(f"\n  best on training: {best[1]}  meanF1={best[0]:.4f}")
    print("  published classical baselines: CellProfiler 0.790 / 0.811")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
