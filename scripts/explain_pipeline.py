"""A narrated, terminal walkthrough of what actually runs during an analysis.

Run this live to show — and honestly label — the machinery under the hood:

    python scripts/explain_pipeline.py

It separates the three levels of "intelligence" in the tool, because they are
not the same thing and it matters that you can tell them apart:

  * classical segmentation  -> NOT AI (deterministic image processing)
  * Cellpose segmentation    -> real AI (a trained deep neural network)
  * the positivity model     -> statistical machine learning (not a neural net)

Cellpose is shown only if it is installed; the rest runs offline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import learned_segmentation, measurements, positivity, preprocessing, segmentation
from src.config import DEFAULTS, SAMPLE_DIR

RULE = "=" * 66
DEMO = "public_human_mitosis.png"
MARKER = "synthetic_marker_pair.png"


def head(title: str) -> None:
    print("\n" + RULE)
    print(" " + title)
    print(RULE)


def step(label: str, tag: str) -> None:
    print(f"\n[{label}]  {tag}")


def main() -> int:
    head("UNDER THE HOOD — what actually runs when you analyse an image")
    print(" Nothing here is hidden. Each stage is labelled with what it really is.")

    # --- 1. LOAD ---------------------------------------------------------
    step("1/5  LOAD THE IMAGE", "plain file reading — no intelligence")
    raw = preprocessing.load_image(SAMPLE_DIR / DEMO)
    print(f"   file            : {DEMO}")
    print(f"   raw pixels      : {raw.shape[1]}x{raw.shape[0]} = {raw.size:,} numbers")
    print(f"   intensity range : {int(raw.min())} .. {int(raw.max())}  (just brightness values)")
    prepared = preprocessing.prepare(SAMPLE_DIR / DEMO)
    print("   -> This is only reading pixels off disk. No model, no decisions yet.")

    # --- 2. CLASSICAL SEGMENTATION --------------------------------------
    step("2/5  CLASSICAL SEGMENTATION", "image processing — NOT AI (deliberately)")
    print("   This is 1990s-era geometry. No training, no model. Watch the steps:")
    plane = prepared.analysis
    blurred = ndi.gaussian_filter(plane, DEFAULTS["smoothing_sigma"])
    from skimage.filters import threshold_otsu
    thr = threshold_otsu(blurred)
    mask = blurred > thr
    print(f"     a) Otsu threshold at {thr:.3f}  -> split bright from dark")
    distance = ndi.distance_transform_edt(mask)
    print(f"     b) distance transform -> furthest-from-edge = each cell's centre "
          f"(deepest point {distance.max():.0f}px)")
    t0 = time.perf_counter()
    result = segmentation.segment(
        plane, threshold_method="otsu", min_size=DEFAULTS["min_size"],
        smoothing_sigma=DEFAULTS["smoothing_sigma"], cleanup_radius=DEFAULTS["cleanup_radius"],
        separate_touching=True, peak_min_distance=DEFAULTS["peak_min_distance"],
        seeding=DEFAULTS["seeding"], seed_depth=DEFAULTS["seed_depth"],
    )
    dt = (time.perf_counter() - t0) * 1000
    print(f"     c) watershed flood from each seed -> {result.n_objects} objects  ({dt:.0f} ms)")
    print("   -> Pure math on geometry. Be honest: this is NOT artificial intelligence.")

    # --- 3. CELLPOSE (the real AI) --------------------------------------
    step("3/5  CELLPOSE SEGMENTATION", "*** THIS is the AI — a deep neural network ***")
    if learned_segmentation.is_available():
        print("   Loading the trained network (~1.15 GB of learned weights)...")
        t0 = time.perf_counter()
        learned = learned_segmentation.segment(plane)
        dt = time.perf_counter() - t0
        print(f"   Ran inference -> {learned.n_objects} objects  ({dt:.1f} s)")
        print("   -> A neural network that LEARNED what a cell looks like from tens of")
        print("      thousands of labelled examples. No thresholds, no geometry rules —")
        print("      it predicts the boundaries. This is the genuine 'AI' in the tool.")
    else:
        print("   (Cellpose not installed here — skipping. When installed it loads a")
        print("    ~1.15 GB trained neural network and predicts cell boundaries directly.)")

    # --- 4. MEASURE ------------------------------------------------------
    step("4/5  MEASURE EACH CELL", "plain arithmetic — no intelligence")
    frame = measurements.measure(result.labels, prepared.intensity)
    print(f"   Computed area, perimeter, circularity, intensity for {len(frame)} cells.")
    row = frame.iloc[0]
    print(f"   e.g. cell #1: area={row['area_pixels']:.0f}px  "
          f"circularity={row['circularity']:.2f}  intensity={row['mean_intensity']:.0f}")
    print("   -> Deterministic maths on the shapes. A spreadsheet, essentially.")

    # --- 5. POSITIVITY MODEL (statistical ML) ---------------------------
    step("5/5  THE POSITIVITY MODEL", "statistical machine learning (not a neural net)")
    print("   The novel part. It LEARNS the positive/negative threshold from the data")
    print("   instead of you eyeballing it. Watch the algorithm actually converge:\n")
    nuc = preprocessing.prepare(SAMPLE_DIR / MARKER, channel="blue")
    mk = preprocessing.prepare(SAMPLE_DIR / MARKER, channel="green")
    seg = segmentation.segment(
        nuc.analysis, threshold_method="otsu", min_size=DEFAULTS["min_size"],
        smoothing_sigma=DEFAULTS["smoothing_sigma"], cleanup_radius=DEFAULTS["cleanup_radius"],
        separate_touching=True, peak_min_distance=DEFAULTS["peak_min_distance"],
        seeding=DEFAULTS["seeding"], seed_depth=DEFAULTS["seed_depth"],
    )
    marker_frame = measurements.measure(seg.labels, mk.intensity)
    values = marker_frame["mean_intensity"].to_numpy()

    # Mirror the EM from positivity.fit_two_component, printing each iteration.
    v = np.sort(values.astype(np.float64))
    split = np.median(v)
    low, high = v[v <= split], v[v > split]
    means = np.array([low.mean(), high.mean()])
    sigmas = np.array([max(low.std(), 1e-3), max(high.std(), 1e-3)])
    weights = np.array([low.size / v.size, high.size / v.size])
    prev = -np.inf
    for it in range(1, 200):
        logc = np.stack([
            np.log(weights[k]) - 0.5 * ((v - means[k]) / sigmas[k]) ** 2
            - np.log(sigmas[k]) for k in range(2)
        ])
        lognorm = np.logaddexp(logc[0], logc[1])
        ll = float(lognorm.sum())
        resp = np.exp(logc - lognorm)
        tot = resp.sum(axis=1)
        weights = tot / v.size
        means = (resp * v).sum(axis=1) / tot
        sigmas = np.sqrt(np.maximum((resp * (v - means[:, None]) ** 2).sum(axis=1) / tot, 1e-6))
        if it in (1, 2, 3, 5, 10) or abs(ll - prev) < 1e-6:
            print(f"     EM iter {it:2d}:  negative pop mean={means[0]:5.1f}  "
                  f"positive pop mean={means[1]:6.1f}  fit={ll:,.0f}")
        if abs(ll - prev) < 1e-6:
            print("     ...converged.")
            break
        prev = ll

    call = positivity.call_by_mixture(values)
    print("\n   Two populations found. Now the TEST — is the split real?")
    print(f"     BIC (2 groups vs 1) : {call.delta_bic:+.0f}   (>10 = strong evidence)")
    print(f"     threshold placed at : {call.threshold:.1f} on the 0-255 scale")
    print(f"     RESULT              : {call.n_positive} of {call.n_total} positive "
          f"= {call.percent_positive:.1f}%")
    print("   -> It learned the threshold and checked a positive population exists.")
    print("      That is machine learning, but statistics — not a neural network.")

    # --- HONEST SUMMARY --------------------------------------------------
    head("SO WHAT IS 'AI' HERE — HONESTLY")
    print("   Cellpose engine     : real AI      (deep neural network, trained on data)")
    print("   Positivity model    : statistical ML (learns the threshold; not a neural net)")
    print("   Classical watershed : NOT AI       (deterministic image processing)")
    print()
    print("   And note: the software agents that helped BUILD and map this project are")
    print("   development tooling. They are NOT part of the running product — the app is")
    print("   a fixed, deterministic pipeline. Same image in, same numbers out, every time.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
