# Biomedical Microscopy Quantification Assistant

A proof-of-concept research software application that transforms microscopy
images into object-level quantitative measurements.

![Example result](assets/example_result.png)

## Current capabilities

- Fluorescence-channel selection (grayscale, red, green, blue)
- Cell/nucleus segmentation by classical image processing
- Separation of touching objects using a distance-transform watershed
- Optional correction of uneven illumination before thresholding
- Cell counting
- Area, shape, and intensity measurements per object
- Multi-page TIFF and z-stack input, flattened by maximum intensity projection
- Batch analysis of many images into a single CSV
- CSV and annotated-image export

## Important limitation

This application is a technical demonstration and has not been validated for
research, diagnostic, or clinical use. It performs **classical image
processing**, not machine learning, and it makes no biological or clinical
interpretation of what it measures.

It *has* been scored against a public expert-annotated dataset (see
[Validation](#validation)): F1 = 0.899 at IoU 0.50 on 5,720 hand-annotated
nuclei it was not tuned on. That establishes the method works on one public
benchmark of Hoechst-stained nuclei. It establishes nothing about any
particular laboratory's images, stains, magnifications, or cell types. Using it
for real work would still need representative images from the lab in question,
annotations from someone who knows those images, an agreed definition of the
measurements that matter, and comparison against whatever process that lab uses
today.

## Quick start

```bash
./run_demo.sh
```

That sets up the environment on first run and opens the app in your browser.
Equivalent manual steps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

The app opens on a bundled public sample and shows a completed analysis
immediately. Everything it needs is committed to the repository, so it runs
with no network connection.

To run the pipeline without the interface:

```bash
python scripts/run_pipeline.py sample_data/public_human_mitosis.png
```

That writes a CSV, a segmentation mask, and an annotated image to `outputs/`.

A three-minute demonstration script, including the failure case to show
and the questions to expect, is in [DEMO.md](DEMO.md).

## Method

```text
Load image
   -> select channel (or convert to grayscale)
   -> invert if the background is light
   -> percentile contrast normalisation          [segmentation copy only]
   -> optional illumination correction           [segmentation copy only, off by default]
   -> Gaussian smoothing
   -> threshold (Otsu, Li, Yen, Triangle, or manual)
   -> morphological opening and closing
   -> fill holes, remove objects below the size floor
   -> distance transform
   -> local maxima become one marker per object
   -> watershed splits touching objects
   -> measure every labelled region
```

Two intensity planes are kept deliberately. Segmentation runs on a
contrast-stretched copy so that thresholding behaves consistently, while
**intensity is measured on the un-stretched plane**. Because the stretch is
computed per image, measuring on it would rescale every image onto the same
range and erase exactly the brightness differences a batch comparison is
looking for.

Integer images are converted against their dtype range rather than their own
observed maximum, for the same reason: a dim image must stay dim.

Multi-page files are read in full and flattened by maximum intensity
projection. This matters more than it sounds: image readers return only the
first page of a multi-page file by default, so a confocal z-stack would
otherwise be analysed as its first slice alone — often the most out-of-focus
one — and report a plausible count with no warning.

### Measurements

Per detected object:

| Column | Meaning |
| --- | --- |
| `object_id` | Label, matching the annotated image |
| `centroid_x`, `centroid_y` | Centre of mass, in pixels |
| `area_pixels` | Area |
| `perimeter_pixels` | Boundary length |
| `equivalent_diameter_pixels` | Diameter of a circle of equal area |
| `circularity` | `4*pi*area / perimeter^2`, clipped to 1 |
| `eccentricity`, `solidity` | Shape descriptors |
| `major_axis_pixels`, `minor_axis_pixels` | Fitted ellipse axes |
| `mean_intensity`, `maximum_intensity`, `minimum_intensity` | Signal, 0–255 scale |
| `touches_border` | Object is clipped by the field of view |

Circularity uses the Crofton perimeter estimate, which is less biased on
digitised boundaries. Values above 1 are a discretisation artefact on very
small objects and are clipped rather than reported.

Objects touching the image border have truncated area and shape. They are
**flagged rather than deleted**, so the decision to exclude them stays with the
researcher.

Measurements are reported in pixels. Micrometre columns appear only if you
supply a pixel size; no scale is ever inferred from the file, because a guessed
scale produces confidently wrong numbers.

## Validation

### Against expert annotations (BBBC039)

Scored against [BBBC039](https://bbbc.broadinstitute.org/BBBC039): 200 real
fluorescence fields of Hoechst-stained U2OS nuclei with 23,617 manually
annotated objects, released CC0 by the Broad Institute.

Parameters were grid-searched on the **training** split only. The numbers below
come from the **held-out test split**, which was never used for tuning.

| Metric | Test split (50 images, 5,720 nuclei) | All 200 images (23,617 nuclei) |
| --- | ---: | ---: |
| **F1 @ IoU 0.50** | **0.899** | 0.894 |
| Precision @ 0.50 | 0.951 | 0.949 |
| Recall @ 0.50 | 0.853 | 0.845 |
| F1 @ IoU 0.75 | 0.838 | 0.830 |
| Average precision (IoU 0.50–0.95) | 0.653 | 0.645 |
| Mean IoU of matched objects | 0.886 | 0.886 |
| Count error (MAPE) | 10.9% | 11.0% |

Reproduce with:

```bash
python scripts/fetch_validation_data.py     # 76 MB, CC0
python scripts/validate.py --split test
```

These are object-level metrics, not counting metrics. That distinction matters:
a method can produce the right count while splitting one nucleus and merging
two others. Matching uses an optimal assignment maximising IoU, so a single
predicted blob cannot claim several true nuclei.

**Where the errors are.** Precision (0.951) is much higher than recall (0.853):
what it reports is nearly always a real nucleus, but it misses some. On the test
split there are 258 merge errors against 87 split errors, so the dominant
failure is still fusing touching nuclei rather than fragmenting single ones.
That is the honest weak point of a distance-transform watershed, and it is where
a learned model such as Cellpose would be expected to help most.

**What this changed.** An earlier version of these defaults was tuned by eye on
a single crop of one image. The annotated data disagreed: lighter smoothing with
no morphological dilation raised mean matched IoU from 0.862 to 0.886, F1 from
0.889 to 0.899, and average precision from 0.605 to 0.653. The synthetic samples
score identically under both, so only real annotations could tell them apart.
That is the argument for validating against annotated data rather than against
intuition.

Average precision is averaged over the full ten-threshold sweep, IoU 0.50 to
0.95 in steps of 0.05, the same range used by the Data Science Bowl and COCO, so
the figure is comparable to published numbers rather than to a private variant.

### Against known counts (synthetic samples)

The synthetic samples are generated by placing each nucleus programmatically,
so the true object count is known and recorded in
`sample_data/ground_truth.json`. This makes counting accuracy measurable rather
than asserted.

| Sample | True objects | Detected | Notes |
| --- | ---: | ---: | --- |
| `synthetic_easy.png` | 32 | **32** | Well separated nuclei |
| `synthetic_moderate.png` | 34 | **34** | Touching nuclei; the case watershed addresses |
| `synthetic_difficult.png` | 110 | 72 | **Known failure case (65% recall)** |

The difficult sample is dense, deeply overlapping, noisy, and unevenly
illuminated. It is included on purpose, and taking it apart showed that its
shortfall is **two separate problems**, only one of which is a watershed
limitation. An earlier version of this README claimed no amount of parameter
tuning could help. That was wrong, and the diagnosis is worth stating.

Because the generator places every nucleus, each true nucleus can be checked
against the output individually:

- **98 of 110 nuclei are found** — their centre falls inside a detected object.
  Only 12 are missed outright.
- All 12 of those lie **outside the foreground mask**, and none were merged into
  a neighbour. They are dim (brightness 0.29–0.48) and clustered in the dark
  corner: 37% are missed where `x < 128`, and none at all where `x > 256`.

That is not overlap. That is a single global threshold failing on an unevenly
lit field, and it is fixable. Turning on illumination correction takes the
detected count from **72 to 90 of 110**.

What remains after that is the genuine limit: 90 detected objects still contain
more than 90 nuclei, because nuclei overlapping past a certain point share one
distance-transform peak and cannot be separated. A test asserts the uncorrected
result stays in its documented range so this claim cannot silently drift.

On the demonstration image (`public_human_mitosis.png`, 512×512) the pipeline
detects 260 objects in roughly 0.1 seconds. There is no ground truth for that
image, so the number is a result, not an accuracy claim.

## Sample data

All bundled images are public demonstration data. None came from a
publication, a clinical source, or an unpublished dataset, and none contains
patient-identifying information. Full attribution is in
`sample_data/source_information.txt`.

The two real images come from the scikit-image sample collection
(BSD-3-Clause): a fluorescence image of human cells in mitosis, and an
H-DAB stained immunohistochemistry image. The three synthetic images are
generated by `scripts/make_sample_data.py` with fixed seeds and are exactly
reproducible.

## Known limitations

- Dense or heavily overlapping objects are under-counted (measured above).
- The watershed assumes roughly convex objects. Highly irregular cells will be
  split incorrectly.
- A single global threshold does not suit unevenly illuminated images. The
  optional illumination correction addresses this, but it is off by default: it
  costs roughly ten times the runtime and is worth at most +0.004 F1 on the
  evenly lit BBBC039 benchmark, which is inside noise. Set its radius larger
  than your objects — a smaller radius hollows them out and under-measures area
  while leaving the object count looking correct.
- Brightfield histology is supported only in the sense that the controls accept
  it. The pipeline is tuned for bright objects on a dark background and is not
  validated for stained tissue.
- Validation covers one public benchmark of Hoechst-stained nuclei. Nothing
  here is validated for any other stain, cell type, magnification, or
  laboratory.

## Possible next steps

Which of these is worth building depends entirely on what an actual lab
workflow needs:

- Count marker-positive versus marker-negative cells and report the percentage
- Compare treatment and control groups with aggregated statistics
- Quantify stained area for histology (fibrosis, damage)
- Measure fluorescence intensity inside cells for uptake or colocalisation work
- Add a pretrained generalist segmentation model (for example Cellpose) as an
  alternative mode, evaluated against manual annotations before being trusted

## Project structure

```text
app.py                      Streamlit interface
requirements.txt
src/
  config.py                 Shared defaults and sample registry
  preprocessing.py          Loading, channel selection, normalisation
  segmentation.py           Thresholding, cleanup, watershed
  measurements.py           Per-object shape and intensity measurements
  visualization.py          Overlays, masks, charts
  export.py                 CSV and PNG serialisation
  validation.py             IoU matching and segmentation metrics
scripts/
  run_pipeline.py           Command-line pipeline
  make_sample_data.py       Regenerate sample images and ground truth
  tune_defaults.py          Grid-search defaults against known counts
  fetch_validation_data.py  Download BBBC039 (76 MB, CC0)
  validate.py               Score the pipeline against expert annotations
  tune_on_bbbc039.py        Grid-search on the BBBC039 training split
  make_figure.py            Render the README figure
docs/                       Validation data notes and recorded results
sample_data/                Public and synthetic images, attribution, ground truth
tests/                      140 tests
outputs/                    Generated results
```

## Tests

```bash
pytest tests/ -q
```

140 tests covering image loading, multi-page and bit-depth handling, channel
selection,
thresholding, watershed separation, measurement correctness, counting accuracy
against ground truth, the scoring metrics themselves, and the interface driven
headlessly including the empty-result and batch paths.

The scoring code is tested against cases with hand-computable answers, because
a bug there would produce authoritative-looking but meaningless accuracy
numbers.
