# Validation data

The single biggest gap in this project is the line in the README: *"Nothing here
is validated against expert annotation."* Closing it needs real microscopy
images with per-object ground-truth masks.

This document records which datasets are worth using and why.

## Conclusion first

Use **BBBC039** for accuracy, and **BBBC005** for a robustness curve. Both are
CC0 public domain, download directly with no account, and are the original
sources that several of the Kaggle re-uploads are copied from.

| Dataset | Size | What it gives us | Licence |
| --- | ---: | --- | --- |
| [BBBC039](https://bbbc.broadinstitute.org/BBBC039) | 76 MB | 200 real fluorescence fields, ~23,000 hand-annotated nuclei, per-object masks | CC0 |
| [BBBC005](https://bbbc.broadinstitute.org/BBBC005) | 11 MB (masks) / 1.8 GB (all images) | Known cell count per image (1–100) plus 48 focus-blur levels | CC0 |

Verified download URLs (all return HTTP 200):

```
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

## Suggested next step

1. Fetch BBBC039 (76 MB) into a gitignored `validation_data/` directory.
2. Decode the coloured masks into labelled matrices.
3. Score the existing pipeline: F1 at IoU 0.5–0.9, plus split/merge counts.
4. Put the real number in the README, whatever it turns out to be.

Step 4 is the point. A measured F1 against 23,000 hand-annotated nuclei is a far
stronger thing to show than any amount of describing the method.
