# Graph Report - .  (2026-07-19)

## Corpus Check
- 65 files · ~180,652 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 915 nodes · 1601 edges · 59 communities (38 shown, 21 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 92 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Marker Threshold Core
- Percent-Positive Workflow
- Rendering & CLI Output
- Preprocessing Tests
- Segmentation Tests
- Docs, Demo & Benchmark Claims
- Documentation Number Guard
- Positivity Tests
- BBBC039 Fetch & Tuning
- Synthetic Sample Generation
- BBBC013 Real-Data Tests
- IoU Scoring & Validation
- Cellpose Engine
- Image Preprocessing
- Classical Segmentation Core
- Config & Size Policy
- App
- Readme
- Validation
- Validation
- Experiment Classical
- Claims
- Learned Segmentation
- Validation
- App
- Measurements
- Synthetic Difficult
- Measurements
- App
- Misc
- Preprocessing
- Sample Data
- App
- Demo
- Measurements
- App
- Claims
- Claims
- App
- App
- Claims
- Claims
- Measurements
- App
- App
- Claims
- Claims
- Validation
- Measurements
- Measurements
- App
- Public Immunohistochemistry
- Readme
- Readme
- Run Demo
- Misc
- Readme
- Readme

## God Nodes (most connected - your core abstractions)
1. `DatasetScore` - 37 edges
2. `score_image (optimal-assignment IoU matching)` - 27 edges
3. `measure_marker (score marker channel per nucleus)` - 25 edges
4. `find()` - 25 edges
5. `segment (full classical pipeline)` - 23 edges
6. `segment_with_defaults()` - 22 edges
7. `boxes()` - 19 edges
8. `prepare (derive display/segmentation/intensity planes)` - 18 edges
9. `claim()` - 18 edges
10. `measure (regionprops table on intensity plane)` - 16 edges

## Surprising Connections (you probably didn't know these)
- `run_demo.sh Zero-Setup Launcher` --semantically_similar_to--> `Three-Minute Demonstration Script`  [INFERRED] [semantically similar]
  run_demo.sh → DEMO.md
- `Marker-Positive Two-Channel Mode Pitch` --semantically_similar_to--> `Reproducible Marker-Positivity Quantification`  [INFERRED] [semantically similar]
  DEMO.md → README.md
- `Positivity Four-Axis Validation (accuracy, determinism, bias, report)` --semantically_similar_to--> `Reproducible Marker-Positivity Quantification`  [INFERRED] [semantically similar]
  docs/positivity_validation.txt → README.md
- `BBBC013 Dose-Response Result Table` --semantically_similar_to--> `BBBC013 Real-Data Dose-Response Validation`  [INFERRED] [semantically similar]
  docs/bbbc013_validation.txt → README.md
- `Documentation number guard` --verifies--> `DEMO.md (guarded doc)`  [EXTRACTED]
  tests/test_documentation.py → DEMO.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Two-channel marker-positive workflow** — src_preprocessing_prepare, src_segmentation_segment, src_measurements_measure, src_markers_measure_marker, src_positivity_call_by_mixture [INFERRED 0.85]
- **Classical vs Cellpose engine dispatch (same result type)** — app_run_segmentation, src_segmentation_segment, src_learned_segmentation_segment, src_segmentation_segmentationresult [INFERRED 0.85]
- **GMM + BIC + Ashman's D bimodality decision** — src_positivity_fit_one_component, src_positivity_fit_two_component, src_positivity_ashman_d, src_positivity_call_by_mixture [INFERRED 0.80]
- **BBBC039 fetch + tune + validate flow** — scripts_fetch_validation_data, scripts_tune_on_bbbc039, scripts_validate [INFERRED 0.85]
- **BBBC013 fetch + validate flow** — scripts_fetch_bbbc013, scripts_validate_bbbc013, dataset_bbbc013 [INFERRED 0.85]
- **Synthetic generate + tune + positivity flow** — scripts_make_sample_data, scripts_tune_defaults, scripts_validate_positivity [INFERRED 0.80]
- **Documentation guard suite (numbers and promises)** — tests_test_documentation_number_guard, tests_test_claims_promises_guard, test_documentation_readme_doc [INFERRED 0.85]
- **Ground-truth accuracy tests** — tests_test_segmentation_module, tests_test_positivity_module, tests_test_bbbc013_module [INFERRED 0.80]
- **Optional learned-segmentation path guard** — tests_test_learned_segmentation_module, src_learned_segmentation, tests_test_claims_optional_dependency_optional [INFERRED 0.75]
- **Validation Evidence Suite** — readme_bbbc039_validation, readme_bbbc013_realdata_validation, readme_synthetic_count_validation [INFERRED 0.85]
- **Honesty and Self-Assessment Mechanisms** — readme_documentation_test, readme_reproducibility_report, readme_difficult_failure_case [INFERRED 0.80]
- **Two Segmentation Engines and Their Benchmarks** — readme_classical_ceiling_0772, readme_cellpose_score_0873, readme_caicedo_benchmark_comparison [INFERRED 0.80]
- **Marker-positive validation series** — sample_data_synthetic_marker_00pct, sample_data_synthetic_marker_10pct, sample_data_synthetic_marker_30pct, sample_data_synthetic_marker_pair, sample_data_synthetic_marker_50pct, sample_data_synthetic_marker_70pct [INFERRED 0.85]

## Communities (59 total, 21 thin omitted)

### Community 0 - "Marker Threshold Core"
Cohesion: 0.06
Nodes (37): Microscopy quantification pipeline.  Classical image-processing pipeline that tu, choose_threshold (otsu/manual), _exact_otsu (exhaustive 1-D split), MarkerResult dataclass, measure_marker (score marker channel per nucleus), _object_means(), DataFrame, ndarray (+29 more)

### Community 1 - "Percent-Positive Workflow"
Cohesion: 0.05
Nodes (56): Percent-marker-positive workflow., render_marker(), marker_ground_truth.json, BBBC013 dataset, Synthetic marker-positivity series, BBBC013 validates population dose-response, not per-cell accuracy, Positivity targets the unreported-thresholding reproducibility gap, download() (+48 more)

### Community 2 - "Rendering & CLI Output"
Cohesion: 0.07
Nodes (53): render_single(), Figure, Figure passes every shipped default explicitly to match the app, main(), Render the example figure used in the README.      python scripts/make_figure.py, build_parser(), main(), ArgumentParser (+45 more)

### Community 3 - "Preprocessing Tests"
Cohesion: 0.06
Nodes (14): BytesIO, png_bytes(), ndarray, Image loading and preparation tests., Requesting all frames must not add a spurious axis to normal images., Bright spot at 40% intensity on a dark field., Reading *encoded* multi-page files, not arrays passed in directly.      These go, Documents a real TIFF ambiguity rather than pretending it away.          A uint8 (+6 more)

### Community 4 - "Segmentation Tests"
Cohesion: 0.09
Nodes (16): disk_image(), Segmentation tests, including counting accuracy against known truth., Counting accuracy on samples whose true object count is known., The difficult sample is expected to under-count.          Asserted as a range so, Uneven illumination defeats a single global threshold.      This is a distinct f, Synthetic image with filled bright disks on a dark background., Three identical disks on a background that fades across the frame., The bright end of the gradient rises above the global threshold.          Measur (+8 more)

### Community 5 - "Docs, Demo & Benchmark Claims"
Cohesion: 0.06
Nodes (43): Marker-Positive Two-Channel Mode Pitch, Rationale: Honesty About Failure Builds Credibility, The Closing Ask (adaptable to a real lab need), Three-Minute Demonstration Script, Volunteer the Limitation (failure case first), F1 = 0.901 @ IoU 0.50 (Test Split), BBBC039 Expert-Annotation Validation, Caicedo et al. 2019 Benchmark Comparison (+35 more)

### Community 6 - "Documentation Number Guard"
Cohesion: 0.09
Nodes (17): claim(), count_objects(), The documented numbers must be true.  Three times in this project's history a ch, The on-screen warning quotes figures; they must stay true too., The published benchmark figures must match the recorded results., The reproducibility figures in the marker-positivity section must hold., README: 0.00 points mean absolute error across the known fractions., The real-data figures must match what the harness produces, when the     BBBC013 (+9 more)

### Community 7 - "Positivity Tests"
Cohesion: 0.08
Nodes (13): one_population(), Tests for the reproducible marker-positivity module.  The module makes a statist, The bundled series has known positive fractions from 0% to 70%., The important direction: 0% true positive must not be called positive., The same input must give an identical fit every time.          Reproducibility i, The failure this exists to prevent: calling a split on one population., TestAgainstKnownFractions, TestBimodalityDetection (+5 more)

### Community 8 - "BBBC039 Fetch & Tuning"
Cohesion: 0.10
Nodes (27): BBBC039 dataset, Tune on training split to avoid leakage, download(), main(), Path, Download the BBBC039 validation dataset.      python scripts/fetch_validation_da, main(), Grid-search segmentation parameters on the BBBC039 *training* split.      python (+19 more)

### Community 9 - "Synthetic Sample Generation"
Cohesion: 0.14
Nodes (27): ground_truth.json, Synthetic sample images, Difficult case is an honest, reported-not-tuned failure, Generator knows the true count, enabling measured accuracy, _draw(), main(), make_difficult(), make_easy() (+19 more)

### Community 10 - "BBBC013 Real-Data Tests"
Cohesion: 0.08
Nodes (7): Real-data validation on BBBC013, and tests for its harness.  The BBBC013 images, The rank correlation is computed without scipy, so it is checked here., On real data, the measured positive fraction must track the known dose., Normalising to the plate's own negative controls must help, not hurt., TestPlatemap, TestRealDoseResponse, TestSpearman

### Community 11 - "IoU Scoring & Validation"
Cohesion: 0.19
Nodes (9): Match predicted objects to true objects and tally errors.      Matching maximise, Rationale: IoU 0.50-0.95 sweep, arange endpoint care, score_image (optimal-assignment IoU matching), boxes(), Optimal assignment must not let a single blob claim both nuclei., Label image from (label, row_slice, col_slice) specs., TestAggregation, TestErrorAttribution (+1 more)

### Community 12 - "Cellpose Engine"
Cohesion: 0.12
Nodes (13): load_model(), ndarray, Optional segmentation with a pretrained generalist model (Cellpose-SAM).  The cl, Load and cache the pretrained model.      The first call downloads about 1.2 GB, Segment a prepared analysis plane with the pretrained model.      Returns the sa, segment (pretrained model inference), Outcome of one segmentation run., SegmentationResult dataclass (+5 more)

### Community 13 - "Image Preprocessing"
Cohesion: 0.14
Nodes (19): correct_illumination (white top-hat), normalize_contrast(), prepare (derive display/segmentation/intensity planes), PreparedImage dataclass (original/analysis/intensity), ndarray, Image loading, channel selection, and intensity preparation.  The analysis image, Reduce an image to a single 2-D analysis plane.      A single channel of a fluor, Percentile contrast stretch.      Percentiles rather than min/max so a handful o (+11 more)

### Community 14 - "Classical Segmentation Core"
Cohesion: 0.17
Nodes (19): Gaussian blur. sigma <= 0 is a no-op., smooth (Gaussian blur), build_mask (threshold + morphological cleanup), compute_threshold (otsu/li/yen/triangle/manual), _drop_small (min_size filter), ndarray, Classical segmentation of bright objects on a dark background.  This is delibera, Threshold and clean up a binary foreground mask. (+11 more)

### Community 15 - "Config & Size Policy"
Cohesion: 0.15
Nodes (9): Shared analysis defaults.  The Streamlit app and the command-line script both re, Classify an image by pixel count. Returns ``(verdict, message)``., size_verdict (pixel-count guard), test_config, Tests for the shared interactive size policy.  The policy lives in config rather, size_verdict reads the module global, so tests can lower the cap.          This, Rationale: batch and single paths must share size policy, TestPolicyIsShared (+1 more)

### Community 16 - "App"
Cohesion: 0.15
Nodes (16): analyze(), analyze_marker(), Biomedical Microscopy Quantification Assistant.  Run with:  streamlit run app.py, Run the full pipeline. Cached on the image bytes and every parameter., Segment with whichever engine is selected.      Both engines return the same res, Two-channel analysis: segment nuclei, score a marker channel inside them., Return (bytes, display name) for whichever source is selected., read_source_bytes() (+8 more)

### Community 17 - "Readme"
Cohesion: 0.14
Nodes (18): BBBC013 Dose-Response Result Table, Manual-Threshold Bias Spread of 20 Points, Positivity Four-Axis Validation (accuracy, determinism, bias, report), BBBC013 Real-Data Dose-Response Validation, Bimodality Test (BIC + Ashman's D), Control-Anchoring Improvement 0.65->0.80, BBBC013 FKHR-GFP Translocation Dataset (CC BY 3.0), Dose-Response Spearman Correlation (+10 more)

### Community 18 - "Validation"
Cohesion: 0.15
Nodes (11): ImageScore, intersection_matrix(), iou_matrix(), ndarray, Instance-segmentation scoring against ground-truth masks.  Counting accuracy alo, Pixel overlap between every predicted and every true object., Intersection-over-union between every predicted and every true object., Scores for a single image. (+3 more)

### Community 19 - "Validation"
Cohesion: 0.18
Nodes (4): DatasetScore, Aggregated scores over a set of images., Mean over thresholds of TP / (TP + FP + FN), the Data Science Bowl metric., Mean absolute percentage error of the per-image object count.

### Community 20 - "Experiment Classical"
Cohesion: 0.20
Nodes (14): Experiment probes the classical ceiling vs CellProfiler baseline, evaluate(), foreground(), main(), Search for a better classical pipeline, scored on the BBBC039 training split.  T, Re-cut each object's edge against a threshold local to that object.      One glo, The shipped approach: one global exclusion radius., Keep only maxima that stand at least ``depth`` above their surroundings.      Sh (+6 more)

### Community 21 - "Claims"
Cohesion: 0.15
Nodes (7): The documented *promises* must hold, not just the documented numbers.  tests/tes, README: border objects are "flagged rather than deleted, so the decision     to, README: "intensity is measured on the un-stretched plane", so brightness     dif, The README states how many tests there are. That is a claim too., TestBorderObjectsAreFlaggedNotDeleted, TestIntensityStaysComparable, TestStatedTestCount

### Community 22 - "Learned Segmentation"
Cohesion: 0.23
Nodes (4): disks(), A learned model produces no intensity threshold, so it must not         report o, The size control must mean the same thing in both modes., TestSegmentation

### Community 23 - "Validation"
Cohesion: 0.24
Nodes (6): app: Streamlit UI and orchestration, DEFAULTS (BBBC039-tuned parameters), decode_colored_mask (BBBC039 4-colour decode), Decode a BBBC039-style colour-coded mask into per-object labels.      The publis, Tests for the scoring code itself.  A bug here would produce authoritative-looki, TestColorMaskDecoding

### Community 24 - "App"
Cohesion: 0.24
Nodes (5): find(), Locate a widget by its visible label., Percent-marker-positive is the shape of a real fluorescence endpoint., The bundled sample has a known positive fraction of 29.55%., TestMarkerMode

### Community 25 - "Measurements"
Cohesion: 0.24
Nodes (4): labelled_disk(), Measurement correctness tests., TestIntensity, TestPixelScale

### Community 26 - "Synthetic Difficult"
Cohesion: 0.22
Nodes (10): Example Result Figure, Known Failure Case, Primary Demo Sample, Classical Segmentation Pipeline, Touching Objects Case, Well-Separated Object Case, Human Mitosis Fluorescence Sample, Synthetic Difficult Sample (+2 more)

### Community 27 - "Measurements"
Cohesion: 0.27
Nodes (9): _empty_frame(), measure (regionprops table on intensity plane), DataFrame, Per-object shape and intensity measurements.  All spatial measurements are in pi, Headline numbers for the summary cards., Measure every labelled object.      ``intensity_plane`` should be the un-normali, Rationale: no guessed pixel-to-micrometre scale, summarize() (+1 more)

### Community 28 - "App"
Cohesion: 0.20
Nodes (4): End-to-end tests driving the Streamlit app headlessly.  These cover the failure, TestBatchMode, TestMinimumSizeTooLarge, TestWatershedToggle

### Community 29 - "Misc"
Cohesion: 0.20
Nodes (10): Promise: border objects flagged not deleted, Promise: intensity measured un-stretched, test_claims, Promise: no scale inferred from file, Promise: no train/test leakage, Promise: illumination correction off by default, Promise: optional dependency stays optional, Rationale: code-property promises are checkable (+2 more)

### Community 30 - "Preprocessing"
Cohesion: 0.29
Nodes (4): Flattening a slowly varying background before thresholding., Identical bright spots on a background that fades across the frame., Documents the failure mode: the radius must exceed the objects., TestIlluminationCorrection

### Community 31 - "Sample Data"
Cohesion: 0.50
Nodes (9): Bimodality no-population detection, Marker-positive validation series, Percent-marker-positive / positivity module, Synthetic marker 0% positive (all-negative case), Synthetic marker 10% positive, Synthetic marker 30% positive, Synthetic marker 50% positive, Synthetic marker 70% positive (+1 more)

### Community 32 - "App"
Cohesion: 0.22
Nodes (4): find_starting(), Locate a widget whose label starts with ``prefix``.      Metric labels carry the, A threshold above every pixel must guide the user, not crash or lie., TestEmptyResult

### Community 33 - "Demo"
Cohesion: 0.29
Nodes (8): validate_bbbc013 harness, DEMO.md (guarded doc), README.md (guarded doc), test_bbbc013, Documented promises guard, test_documentation, Documentation number guard, Rationale: tests assert ranges, docs assert values (drift gap)

### Community 34 - "Measurements"
Cohesion: 0.29
Nodes (5): circularity_from (4*pi*area/perimeter^2 clipped), ndarray, 4*pi*area / perimeter**2, clipped to [0, 1].      A perfect circle scores 1. Val, Rationale: Crofton perimeter for circularity, TestCircularity

### Community 35 - "App"
Cohesion: 0.29
Nodes (4): The engine choice must be offered, and must degrade safely., The default must need no network, no GPU and no 1 GB download., Whether or not Cellpose is installed, choosing it must be safe.          When it, TestSegmentationEngineSelector

### Community 37 - "Claims"
Cohesion: 0.29
Nodes (3): README: "Everything it needs is committed to the repository, so it runs     with, The default engine must not drag in the optional deep-learning stack.          I, TestRunsOffline

## Knowledge Gaps
- **83 isolated node(s):** `run_demo.sh script`, `config: shared analysis defaults`, `Rationale: intensity measured on un-stretched plane`, `Rationale: scale ints by dtype range not observed peak`, `Rationale: read all frames, max-project z-stacks` (+78 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ImageLoadError` connect `App` to `Preprocessing Tests`, `Image Preprocessing`, `Classical Segmentation Core`, `Preprocessing`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `DatasetScore` connect `Validation` to `BBBC039 Fetch & Tuning`, `IoU Scoring & Validation`, `Validation`, `Validation`, `Experiment Classical`, `Validation`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Why does `SegmentationResult dataclass` connect `Cellpose Engine` to `Marker Threshold Core`, `IoU Scoring & Validation`, `Classical Segmentation Core`, `Learned Segmentation`, `Measurements`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `DatasetScore` (e.g. with `TestAggregation` and `TestColorMaskDecoding`) actually correct?**
  _`DatasetScore` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `measure_marker (score marker channel per nucleus)` (e.g. with `separation (population gap metric)` and `PreparedImage dataclass (original/analysis/intensity)`) actually correct?**
  _`measure_marker (score marker channel per nucleus)` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `segment (full classical pipeline)` (e.g. with `segment (pretrained model inference)` and `ValueError`) actually correct?**
  _`segment (full classical pipeline)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `run_demo.sh script`, `config: shared analysis defaults`, `Rationale: intensity measured on un-stretched plane` to the rest of the system?**
  _83 weakly-connected nodes found - possible documentation gaps or missing edges._