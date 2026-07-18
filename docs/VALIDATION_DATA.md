# Validation data

The biggest gap in this project used to be that nothing was validated against
expert annotation. Closing it needed real microscopy images with per-object
ground-truth masks.

That is now done — see [Results](#results-done) — and this document records
which datasets were considered, which was chosen and why, and what the scoring
turned up. It is kept as the working record behind the numbers in the README.

## Conclusion first

Use **BBBC039** for accuracy, and **BBBC005** for a robustness curve. Both are
CC0 public domain, download directly with no account, and are the original
sources that several of the Kaggle re-uploads are copied from.

| Dataset | Size | What it gives us | Licence |
| --- | ---: | --- | --- |
| [BBBC039](https://bbbc.broadinstitute.org/BBBC039) | 76 MB | 200 real fluorescence fields, ~23,000 hand-annotated nuclei, per-object masks | CC0 |
| [BBBC005](https://bbbc.broadinstitute.org/BBBC005) | 11 MB (masks) / 1.8 GB (all images) | Known cell count per image (1–100) plus 48 focus-blur levels | CC0 |

Verified download URLs (all return HTTP 200):

```text
https://data.broadinstitute.org/bbbc/BBBC039/images.zip      74 MB
https://data.broadinstitute.org/bbbc/BBBC039/masks.zip        2 MB
https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip    <1 MB
https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip  11 MB
```

## Why BBBC039 is the right accuracy set

It matches what this pipeline is actually built for. The images are Hoechst-
stained nuclei (DNA channel) in fluorescence: bright objects on a dark
background, 16-bit TIFF, 520×696. That is the same modality as the demo image,
so a score on it is a fair test rather than an off-distribution one.

Critically, the masks are **per object**, with touching nuclei distinguished.
That is what lets us move past counting and report real segmentation quality:

- F1 / average precision at matched IoU thresholds
- Per-object IoU distribution
- Split and merge error rates, which is precisely where the watershed step
  either earns its place or does not

The cells are U2OS treated with ~200 bioactive compounds, so nuclei vary in
size, brightness, and density within one dataset. That variety is the point.

## Why BBBC005 is the right robustness set

Cell count is encoded directly in the filename
(`SIMCEPImages_well_Ccells_Fblur_ssamples_wstain.TIF`, where `C` is the count),
and the same fields are rendered at 48 levels of focus blur.

That gives something the current synthetic samples cannot: an **accuracy-versus-
defocus curve**. Instead of asserting "out-of-focus images degrade results", we
could publish the exact point where counting accuracy falls off. It extends the
methodology already used in `sample_data/ground_truth.json` to thousands of
images at no extra effort.

The full image archive is 1.8 GB. The 11 MB ground-truth archive plus a
filtered subset of images is enough; there is no need to pull all 19,200.

## On the Kaggle options

The request was to look at Kaggle, so for the record:

| Kaggle dataset | Verdict |
| --- | --- |
| [2018 Data Science Bowl](https://www.kaggle.com/competitions/data-science-bowl-2018/data) | 670 train images with per-nucleus masks, mixed modalities (brightfield *and* fluorescence). Genuinely good, but it is competition data, so use is governed by competition rules, and much of it derives from BBBC sets anyway. |
| [Synthetic Cell Images and Masks (BBBC005-v1)](https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1) | A re-upload of BBBC005. Prefer the original. |
| [Sartorius Cell Instance Segmentation](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation) | Neuronal cells in phase contrast. Different modality, much harder, and classical watershed will do badly. Useful later as an honest hard case, not for headline validation. |
| [Blood Cell Segmentation (BCCD with masks)](https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask) | Brightfield blood smears, semantic masks. Easy but off-modality. Low priority. |
| [Breast Cancer Cell Segmentation](https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation) | Histology. Off-modality for this pipeline. |

Everything on Kaggle requires an account and an API token
(`~/.kaggle/kaggle.json`) to download programmatically. Neither is configured on
this machine. The BBBC originals need neither, which is another reason to
prefer them.

## What is *not* available, and it matters

There is no public Kaggle dataset of **cardiac** fluorescence microscopy,
stem-cell, or exosome-uptake images with segmentation masks. Cardiac fibrosis
work in the literature uses Masson's trichrome or picrosirius red staining and
is generally published as figures, not as annotated datasets.

The honest consequence: public data can prove this pipeline counts nuclei
correctly, but it cannot demonstrate anything about Professor Singla's actual
assays. That gap is closed by representative images from the lab, not by more
downloading. Which is exactly the ask in `DEMO.md`.

## Results (done)

BBBC039 is now wired in. Reproduce with:

```bash
python scripts/fetch_validation_data.py
python scripts/validate.py --split test
```

Headline on the **held-out test split** (50 images, 5,720 nuclei), with
parameters grid-searched on the training split only:

| Metric | Value |
| --- | ---: |
| F1 @ IoU 0.50 | **0.899** |
| Precision / Recall @ 0.50 | 0.951 / 0.853 |
| F1 @ IoU 0.75 | 0.838 |
| Average precision (IoU 0.50–0.90) | 0.706 |
| Mean IoU of matched objects | 0.886 |
| Split / merge errors | 87 / 258 |

Recorded in `validation_bbbc039_test.json` and `validation_bbbc039_all.json`.

### The mask-format trap

The BBBC039 masks are **not** labelled by object id. They are 4-coloured so any
two touching nuclei carry different colours, and individual objects are
recovered by connected components computed *within each colour*.

Reading the colour channel as an object id is the obvious first guess and it is
wrong: it yields 531 objects across the dataset instead of 23,617. Every
downstream metric would have been silently meaningless. `decode_colored_mask`
in `src/validation.py` does it correctly, and the count is asserted against the
documented ~23,000.

### What it changed

The defaults had been tuned by eye on a single crop. The annotated data
disagreed, and lighter smoothing with no morphological dilation won:

| | Old (by eye) | New (from annotations) |
| --- | ---: | ---: |
| F1 @ 0.50 | 0.889 | **0.899** |
| Average precision | 0.660 | **0.706** |
| Mean matched IoU | 0.862 | **0.886** |
| Split errors | 112 | **87** |

The synthetic samples score identically (32/32 and 34/34) under both settings,
so no amount of work on the synthetic data could have found this. That is the
case for real annotations in one table.

Test-split scores came out marginally *above* training-split scores, which is
the expected signature of a method with very few free parameters and no
capacity to memorise.

## Still open

- **BBBC005** for an accuracy-versus-defocus curve. Not yet done.
- **Nothing here touches Professor Singla's assays.** Public data proves the
  method counts Hoechst-stained nuclei well. It says nothing about cardiac
  tissue, stem cells, or exosome uptake. That gap closes with lab images, not
  with more downloading.
