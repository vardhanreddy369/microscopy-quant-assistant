"""Biomedical Microscopy Quantification Assistant.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src import export, measurements, preprocessing, segmentation, visualization
from src.config import DEFAULTS, SAMPLE_IMAGES, SAMPLE_OVERRIDES, sample_path
from src.preprocessing import BACKGROUNDS, CHANNELS, ImageLoadError
from src.segmentation import THRESHOLD_METHODS

# Guards against a multi-gigapixel upload stalling the app during a live demo.
MAX_PIXELS = 40_000_000
SLOW_PIXELS = 6_000_000

AUTO_METHODS = ("otsu", "li", "yen", "triangle")

st.set_page_config(
    page_title="Microscopy Quantification Assistant",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
      [data-testid="stMetricValue"] { font-size: 2.0rem; }
      [data-testid="stMetricLabel"] { font-size: 0.95rem; }
      .block-container { padding-top: 2.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=24)
def analyze(
    image_bytes: bytes,
    channel: str,
    background: str,
    threshold_method: str,
    manual_threshold: float | None,
    min_size: int,
    smoothing_sigma: float,
    cleanup_radius: int,
    fill_holes: bool,
    separate_touching: bool,
    peak_min_distance: int,
    pixel_size_um: float | None,
):
    """Run the full pipeline. Cached on the image bytes and every parameter."""
    prepared = preprocessing.prepare(
        io.BytesIO(image_bytes), channel=channel, background=background
    )
    result = segmentation.segment(
        prepared.analysis,
        threshold_method=threshold_method,
        manual_threshold=manual_threshold,
        min_size=min_size,
        smoothing_sigma=smoothing_sigma,
        cleanup_radius=cleanup_radius,
        fill_holes=fill_holes,
        separate_touching=separate_touching,
        peak_min_distance=peak_min_distance,
    )
    frame = measurements.measure(result.labels, prepared.intensity, pixel_size_um)
    return prepared, result, frame


def read_source_bytes(uploaded, sample_filename: str) -> tuple[bytes, str] | None:
    """Return (bytes, display name) for whichever source is selected."""
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name
    path = sample_path(sample_filename)
    if not path.exists():
        return None
    return path.read_bytes(), path.name


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Analysis controls")

mode = st.sidebar.radio(
    "Mode",
    ("Single image", "Batch (multiple images)"),
    help="Batch mode analyses several images with identical settings and "
         "returns one combined CSV.",
)

st.sidebar.divider()

sample_labels = [label for label, _, _ in SAMPLE_IMAGES]
sample_files = {label: filename for label, filename, _ in SAMPLE_IMAGES}
sample_notes = {label: note for label, _, note in SAMPLE_IMAGES}

uploaded_single = None
uploaded_batch: list = []

if mode == "Single image":
    st.sidebar.subheader("Image source")
    uploaded_single = st.sidebar.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )
    sample_label = st.sidebar.selectbox(
        "or use a demonstration image",
        sample_labels,
        help="Bundled public and synthetic images. These work with no network "
             "connection.",
    )
    st.sidebar.caption(sample_notes[sample_label])
    if uploaded_single is not None:
        st.sidebar.info("Using the uploaded image. Remove it to return to the samples.")
else:
    st.sidebar.subheader("Image sources")
    uploaded_batch = st.sidebar.file_uploader(
        "Upload several images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )
    use_samples = st.sidebar.checkbox(
        "Include the bundled sample images", value=not uploaded_batch
    )
    sample_label = sample_labels[0]

selected_file = sample_files[sample_label]

# Applying a sample's overrides means switching to the histology image does not
# silently analyse it with fluorescence settings that make no sense for it.
if st.session_state.get("_active_sample") != selected_file:
    st.session_state["_active_sample"] = selected_file
    overrides = SAMPLE_OVERRIDES.get(selected_file, {})
    st.session_state["channel"] = overrides.get("channel", DEFAULTS["channel"])
    st.session_state["background"] = overrides.get("background", DEFAULTS["background"])

st.sidebar.divider()
st.sidebar.subheader("Preprocessing")

channel = st.sidebar.selectbox(
    "Analysis channel", CHANNELS, key="channel",
    help="For a fluorescence image, pick the channel the stain is in. "
         "Grayscale mixes all channels together.",
)
background = st.sidebar.selectbox(
    "Background", BACKGROUNDS, key="background",
    help="'dark' means bright objects on a dark background, the usual "
         "fluorescence case. 'light' inverts the image first.",
)
smoothing_sigma = st.sidebar.slider(
    "Noise reduction (Gaussian sigma)", 0.0, 6.0,
    float(DEFAULTS["smoothing_sigma"]), 0.2,
    help="Blur applied before thresholding. Higher removes more noise but "
         "merges nearby objects.",
)

st.sidebar.divider()
st.sidebar.subheader("Segmentation")

threshold_mode = st.sidebar.radio("Threshold", ("Automatic", "Manual"), horizontal=True)
if threshold_mode == "Automatic":
    threshold_method = st.sidebar.selectbox(
        "Automatic method", AUTO_METHODS,
        index=AUTO_METHODS.index(DEFAULTS["threshold_method"]),
        help="Otsu is the standard choice for a clearly bimodal image.",
    )
    manual_threshold = None
else:
    threshold_method = "manual"
    manual_threshold = st.sidebar.slider(
        "Manual threshold", 0.0, 1.0, float(DEFAULTS["manual_threshold"]), 0.01,
        help="Intensity cut-off on the contrast-normalised image.",
    )

min_size = st.sidebar.slider(
    "Minimum object size (pixels)", 0, 2000, int(DEFAULTS["min_size"]), 10,
    help="Objects smaller than this are discarded as noise or debris.",
)
separate_touching = st.sidebar.checkbox(
    "Separate touching objects (watershed)", value=bool(DEFAULTS["separate_touching"])
)
peak_min_distance = st.sidebar.slider(
    "Minimum distance between object centres", 1, 40,
    int(DEFAULTS["peak_min_distance"]), 1,
    disabled=not separate_touching,
    help="Roughly the smallest expected object radius. Too large merges "
         "touching objects; too small splits single objects in two.",
)

with st.sidebar.expander("Advanced"):
    cleanup_radius = st.slider(
        "Morphological cleanup radius", 0, 5, int(DEFAULTS["cleanup_radius"]), 1
    )
    fill_holes = st.checkbox("Fill holes inside objects", value=bool(DEFAULTS["fill_holes"]))
    show_ids_choice = st.selectbox(
        "Object ID labels", ("Automatic", "Always", "Never"),
        help="Automatic hides the numbers when there are too many objects for "
             "them to be readable.",
    )

st.sidebar.divider()
st.sidebar.subheader("Scale (optional)")
pixel_size_um = st.sidebar.number_input(
    "Pixel size (micrometres per pixel)", min_value=0.0, value=0.0, step=0.01,
    format="%.4f",
    help="Leave at 0 to report pixels only. Only set this if you know the "
         "true scale for the image; a guessed value produces confidently "
         "wrong micrometre measurements.",
)
pixel_size = pixel_size_um if pixel_size_um > 0 else None

SHOW_IDS = {"Automatic": "auto", "Always": True, "Never": False}[show_ids_choice]

PARAMS = dict(
    channel=channel,
    background=background,
    threshold_method=threshold_method,
    manual_threshold=manual_threshold,
    min_size=min_size,
    smoothing_sigma=smoothing_sigma,
    cleanup_radius=cleanup_radius,
    fill_holes=fill_holes,
    separate_touching=separate_touching,
    peak_min_distance=peak_min_distance,
    pixel_size_um=pixel_size,
)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Biomedical Microscopy Quantification Assistant")
st.markdown(
    "A proof-of-concept tool for converting microscopy images into "
    "reproducible quantitative measurements."
)
st.warning(
    "Demonstration software using public sample images. Segmentation is "
    "classical image processing, not a trained model or an AI diagnosis. "
    "Results require scientific validation before research or clinical use.",
    icon="⚠️",
)


def render_single(image_bytes: bytes, display_name: str) -> None:
    prepared, result, frame = analyze(image_bytes, **PARAMS)
    summary = measurements.summarize(frame)

    area_column = "area_um2" if pixel_size else "area_pixels"
    area_unit = "µm²" if pixel_size else "px"

    cards = st.columns(4)
    cards[0].metric("Objects detected", f"{summary['count']:,}")
    cards[1].metric(
        "Average area",
        "n/a" if not summary["count"] else f"{frame[area_column].mean():,.0f} {area_unit}",
    )
    cards[2].metric(
        "Median area",
        "n/a" if not summary["count"] else f"{frame[area_column].median():,.0f} {area_unit}",
    )
    cards[3].metric(
        "Average intensity",
        "n/a" if not summary["count"] else f"{summary['mean_intensity']:.1f}",
        help="Mean signal inside objects, on a 0-255 scale, measured before "
             "contrast normalisation.",
    )

    if result.empty:
        st.error(
            "No objects were detected. Try lowering the threshold, reducing the "
            "minimum object size, or checking that the background mode matches "
            "the image.",
            icon="🔍",
        )
    elif summary["border_objects"]:
        st.caption(
            f"{summary['border_objects']} object(s) touch the image border. Their "
            "area and shape are truncated by the field of view; they are flagged "
            "in the `touches_border` column rather than removed."
        )

    annotated = visualization.annotate(prepared.analysis, result.labels, show_ids=SHOW_IDS)

    if (
        SHOW_IDS == "auto"
        and result.n_objects > visualization.MAX_OBJECTS_FOR_IDS
    ):
        st.caption(
            f"ID numbers are hidden because {result.n_objects} objects are too "
            "dense to label readably. Every ID is still in the table below. "
            "Set 'Object ID labels' to 'Always' in Advanced to force them on."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(visualization.to_display_rgb(prepared.original), width="stretch")
    with right:
        st.subheader("Annotated result")
        st.image(annotated, width="stretch")

    with st.expander("Segmentation mask"):
        mask_cols = st.columns(2)
        mask_cols[0].image(
            visualization.mask_to_image(result.mask),
            caption="Binary foreground mask",
            width="stretch",
        )
        mask_cols[1].image(
            visualization.labels_to_color(result.labels),
            caption="Individual objects (one colour per object)",
            width="stretch",
        )
        st.caption(
            f"Threshold {result.threshold:.4f} using the "
            f"{'manual value' if result.method == 'manual' else result.method + ' method'}"
            f"{'; touching objects separated by watershed' if result.separated else ''}."
        )

    with st.expander("Distribution charts", expanded=False):
        chart_cols = st.columns(2)
        chart_cols[0].pyplot(visualization.area_histogram(frame), width="stretch")
        chart_cols[1].pyplot(
            visualization.intensity_histogram(frame), width="stretch"
        )

    with st.expander("Measurement table", expanded=False):
        st.dataframe(frame, width="stretch", height=340)
        st.caption(
            f"{len(frame)} rows, one per detected object. Circularity is "
            "4*pi*area/perimeter^2 clipped to 1."
        )

    st.subheader("Downloads")
    stem = Path(display_name).stem
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Measurements (CSV)",
        export.dataframe_to_csv_bytes(frame),
        file_name=f"{stem}_measurements.csv",
        mime="text/csv",
        width="stretch",
    )
    download_cols[1].download_button(
        "Annotated image (PNG)",
        export.image_to_png_bytes(annotated),
        file_name=f"{stem}_annotated.png",
        mime="image/png",
        width="stretch",
    )
    download_cols[2].download_button(
        "Segmentation mask (PNG)",
        export.image_to_png_bytes(visualization.mask_to_image(result.mask)),
        file_name=f"{stem}_mask.png",
        mime="image/png",
        width="stretch",
    )


def render_batch(sources: list[tuple[bytes, str]]) -> None:
    st.subheader("Batch analysis")
    st.caption(
        f"{len(sources)} image(s) will be analysed with identical settings. "
        "Intensity is measured before per-image contrast normalisation, so "
        "values stay comparable between images."
    )

    if not st.button("Run batch analysis", type="primary"):
        st.info("Press **Run batch analysis** to process every selected image.")
        return

    per_object: list[pd.DataFrame] = []
    rows: list[dict] = []
    progress = st.progress(0.0, text="Analysing...")

    for index, (image_bytes, name) in enumerate(sources, start=1):
        progress.progress(index / len(sources), text=f"Analysing {name}...")
        try:
            _, result, frame = analyze(image_bytes, **PARAMS)
        except (ImageLoadError, ValueError) as exc:
            rows.append({"source_image": name, "objects_detected": None, "error": str(exc)})
            continue

        summary = measurements.summarize(frame)
        rows.append(
            {
                "source_image": name,
                "objects_detected": summary["count"],
                "mean_area_pixels": summary["mean_area"],
                "median_area_pixels": summary["median_area"],
                "mean_intensity": summary["mean_intensity"],
                "mean_circularity": summary["mean_circularity"],
                "objects_touching_border": summary["border_objects"],
                "threshold": result.threshold,
                "error": "",
            }
        )
        if not frame.empty:
            tagged = frame.copy()
            tagged.insert(0, "source_image", name)
            per_object.append(tagged)

    progress.empty()

    summary_frame = pd.DataFrame(rows)
    st.markdown("**Per-image summary**")
    st.dataframe(summary_frame, width="stretch")

    combined = (
        pd.concat(per_object, ignore_index=True) if per_object else pd.DataFrame()
    )

    total = int(summary_frame["objects_detected"].fillna(0).sum())
    st.metric("Total objects across all images", f"{total:,}")

    if combined.empty:
        st.warning("No objects were detected in any image.", icon="🔍")
        return

    st.markdown("**Combined per-object measurements**")
    st.dataframe(combined, width="stretch", height=340)

    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Combined measurements (CSV)",
        export.dataframe_to_csv_bytes(combined),
        file_name="batch_measurements.csv",
        mime="text/csv",
        width="stretch",
    )
    download_cols[1].download_button(
        "Per-image summary (CSV)",
        export.dataframe_to_csv_bytes(summary_frame),
        file_name="batch_summary.csv",
        mime="text/csv",
        width="stretch",
    )


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

try:
    if mode == "Single image":
        source = read_source_bytes(uploaded_single, selected_file)
        if source is None:
            st.error(
                f"Sample image `{selected_file}` is missing. Regenerate it with "
                "`python scripts/make_sample_data.py`.",
                icon="📁",
            )
        else:
            image_bytes, display_name = source
            probe = preprocessing.load_image(io.BytesIO(image_bytes))
            pixels = probe.shape[0] * probe.shape[1]
            if pixels > MAX_PIXELS:
                st.error(
                    f"This image is {probe.shape[1]}x{probe.shape[0]} pixels, which "
                    "is too large to analyse interactively. Crop or downscale it "
                    "first.",
                    icon="🛑",
                )
            else:
                if pixels > SLOW_PIXELS:
                    st.info(
                        f"Large image ({probe.shape[1]}x{probe.shape[0]}). "
                        "Analysis may take a few seconds.",
                        icon="⏳",
                    )
                with st.spinner("Analysing..."):
                    render_single(image_bytes, display_name)
    else:
        sources: list[tuple[bytes, str]] = [
            (item.getvalue(), item.name) for item in uploaded_batch
        ]
        if use_samples:
            for _, filename, _ in SAMPLE_IMAGES:
                path = sample_path(filename)
                if path.exists():
                    sources.append((path.read_bytes(), path.name))

        if not sources:
            st.info(
                "Upload two or more images, or tick **Include the bundled sample "
                "images**, to run a batch.",
                icon="📤",
            )
        else:
            render_batch(sources)

except ImageLoadError as exc:
    st.error(f"That file could not be read as an image. {exc}", icon="🚫")
except MemoryError:
    st.error(
        "Ran out of memory analysing this image. Try a smaller or cropped image.",
        icon="🛑",
    )
except Exception as exc:  # noqa: BLE001 - a demo must not show a raw traceback
    st.error(f"Analysis failed: {exc}", icon="🚫")
    with st.expander("Technical detail"):
        st.exception(exc)

st.divider()
st.caption(
    "Method: channel selection, percentile contrast normalisation, Gaussian "
    "smoothing, thresholding, morphological cleanup, distance transform, and "
    "watershed separation of touching objects, followed by region measurement. "
    "This is classical image processing. It has not been validated against "
    "expert-annotated data and should not be used for research conclusions or "
    "clinical decisions."
)
