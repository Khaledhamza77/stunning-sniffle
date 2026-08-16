# Scene Calibration

`scripts/calibrate_scene.py` — derives the road geometry for the fixed traffic camera
from the object detector's own output, and writes it to `config/lanes_cctv052.json`.

Everything downstream (lane assignment, lane occupancy, speed estimation, density)
depends on this file. It is produced once, offline, and then reused.

---

## 1. What it produces, and the principle behind it

The calibration answers three questions about the scene:

| Question                                    | Answer produced                           |
| ------------------------------------------- | ----------------------------------------- |
| Where is each lane?                         | 5 polynomial centerlines,`x = f(y)`     |
| Which part of the frame is measurable road? | An ROI polygon                            |
| How many metres is a pixel?                 | A scale curve,`metres_per_pixel = f(y)` |

**The governing principle: only calibrate the region where the lanes are actually
separable in the data, and make that region the ROI (region of interest).**

A traffic camera looks down a road that recedes toward the horizon. Perspective
compresses the lanes together with distance until they are no longer distinguishable —
and beyond that point, any lane assignment is invented rather than measured. Rather
than extrapolating lanes into that region, this script finds where separability breaks
down and stops there. The calibrated region is deliberately smaller than the frame.

The trade-off is accepted knowingly: less coverage, but every number downstream comes
from a region where the geometry was actually observed.

### Coordinate convention

All coordinates are in **native frame pixels (320 × 240)**. `y` increases *downward*,
so **larger `y` = closer to the camera**. This matters for reading the code and the
config: the "top" of the frame is the far field.

---

## 2. Prerequisites

1. **Dataset present** — `data/highway-traffic-videos-dataset/`, via
   `scripts/download_dataset.py`.
2. **Patch ROI recovered** — `scripts/recover_patch_roi.py` must have been run first.
   It writes `outputs/calibration/patch_roi_recovery.json`, and this script raises a
   `RuntimeError` if that file is missing.
3. **GPU** — detection is hardcoded to CUDA device 0 (`device=0` in `detect_in_clip`).

## 3. Usage

```bash
# normal run: reuses the cached detections in outputs/calibration/detections.csv
uv run python scripts/calibrate_scene.py

# re-run YOLO detection from scratch (required after changing SEED or CLIP_SAMPLE)
uv run python scripts/calibrate_scene.py --no-cache
```

The detection cache stores no sampling parameters, so the script verifies that the
cached clips match the currently configured sample and raises rather than silently
calibrating against the wrong detections.

---

## 4. Algorithm

### Step 1 — Sample clips

Clips are sampled per congestion label with **different counts per label**, because the
labels serve different purposes:

- **Light traffic (20 clips)** — carries the lane geometry. Lane centers are only
  meaningful when vehicles are spread across lanes rather than packed bumper to bumper.
  Volume matters here: the near-field bands are sparsely populated in light traffic, and
  a band skipped for lack of detections would be excluded from the calibrated region for
  the wrong reason.
- **Medium and heavy (8 clips each)** — only feed the scale curve and the ROI edge
  shift. Both were measured to be flat past ~8 clips per label, so sampling more only
  costs detection time.

### Step 2 — Detect and bound

YOLOv8m runs over each clip at `IMGSZ=640` (the 320×240 source upscaled 2×, since small
distant vehicles are missed at native resolution), skipping frame 0, which is corrupted
in every clip in this dataset. Only the four vehicle classes are kept. Detection runs at
the permissive `DETECT_CONF`, and calibration then keeps only detections at or above
`CONF_THRESH`. They are further restricted to the **patch ROI box** recovered from the
dataset's own supplementary files, which excludes the opposite carriageway.

### Step 3 — Reject off-road detections

Within each horizontal band, detections are sorted by `x` and split wherever the gap
between neighbours exceeds `OUTLIER_GAP_PX`. Only the **largest** contiguous run is
kept. The real carriageway produces far more detections than any spurious cluster
(tree-line false positives, the far carriageway), so this isolates it without
hand-picked pixel bounds.

### Step 4 — Cluster each band into lanes

The frame is divided into horizontal bands of `BAND_HEIGHT_PX`. Each band is clustered
independently on `x` into `NUM_LANES` groups using **agglomerative clustering with
average linkage**, and the cluster means are recorded as that band's lane centers.

Average linkage was chosen over the alternatives after both failed on this data:

- **k-means** assumes equal-variance, equal-density clusters. Lane occupancy is uneven,
  so its centers drifted toward the busier lanes.
- **single linkage** merges on nearest-point distance, so one stray detection between
  two lanes chains them into a single cluster.

### Step 5 — Find the separable region (the key step)

Bands are not all trustworthy, and a band can be wrong while still *looking* internally
fine — its five clusters can be cleanly separated from each other yet not correspond to
the five lanes. Detecting that needs a check external to the clustering itself.

The check used is **perspective consistency of the lane span**, where span is the
distance from the leftmost to the rightmost lane center. Under perspective, a road must
appear *narrower* the further away it is. So span must increase monotonically with `y`
(toward the camera). A band that violates this has not measured the road:

- **Span jumps up** → the clustering absorbed something that is not a lane.
- **Span collapses** → part of the road was lost.

The script keeps the **longest run of consecutive bands** whose span is monotonic within
`SPAN_TOLERANCE_PX`. Bands must also be adjacent — a gap means an unmeasured band, and
the run does not span it.

This requires no threshold tuned to a particular camera, and on this scene it correctly
identified two failure regions arising from two unrelated physical causes (§7).

### Step 6 — Fit lanes

One degree-2 polynomial `x = f(y)` is fit per lane through the accepted band centers.
The run's `y`-extent becomes the **calibrated region**; lanes are neither defined nor
evaluated outside it. Maximum residuals are recorded per lane as a fit-quality check.

### Step 7 — Derive the ROI

The outer lane centerlines run down the *middle* of the outer lanes, not along the road
edge. They are shifted outward by enough to enclose the detections falling beyond them —
using the 99th percentile of the overshoot rather than the maximum, so a few strays do
not balloon the ROI — then clipped to the patch box. The ROI spans only the calibrated
`y`-range.

### Step 8 — Fit the scale model

The spacing between lane centerlines is compared against the standard lane width to
give lateral metres-per-pixel at each row, and the model

```
metres_per_pixel(y) = h / (y - y_horizon)
```

is fit across the calibrated region. That functional form is what a pinhole camera
viewing a flat plane produces, and it fits **3.8× better than a straight line**
(RMS 0.0039 vs 0.0148). The difference is not cosmetic: a straight line is visibly off at
both ends of the region and would bias every speed computed there.

**Why lane width rather than vehicle width.** Calibrating from car boxes was tried
first and is measurably biased. It put the implied lane width at 2.70 m, 26% under the
3.66 m standard, because the camera views vehicles obliquely — each box contains the
car's rear *plus* part of its side, so it spans about 2.42 m rather than the car's true
~1.8 m. Lane width avoids this: it is an engineering standard rather than a population
average, it is a purely lateral measurement, and it does not depend on how tightly the
detector draws boxes. The implied car-box width is now emitted as a *consistency check*
instead (§7).

### Step 8b — Lateral vs along-road scale

The fitted scale is **lateral**. Vehicles travel *along* the road, and under perspective
those two scales differ. For a pinhole camera viewing a flat plane, along-road position is

```
D(y) = f · metres_per_pixel(y)
```

for the camera's focal length `f` in pixels. `f` cannot be recovered from this dataset —
there is no camera metadata and no surveyed reference length — so along-road distance is
carried in units of `metres / f`, which `scripts/scene.py` calls *raw*. Two consequences:

- **Ratios are exact.** `f` cancels, so anything expressed relative to free-flow speed
  needs no anchor. This is what the congestion analysis uses, and it also means the
  *absolute* scale of `metres_per_pixel` is irrelevant there — only its shape matters.
- **Absolute speeds need an anchor**, and are therefore assumption-based rather than
  measured.

Using the lateral scale directly for along-road motion — the obvious shortcut — would be
wrong by a factor of `f / (y - y_horizon)`, which varies about fourfold across this
region, so `scene.py` deliberately does not offer it.

---

## 5. Parameters

All are module-level constants at the top of the script.

### Detection

| Parameter             | Value          | Meaning                                                                                               |
| --------------------- | -------------- | ----------------------------------------------------------------------------------------------------- |
| `MODEL_NAME`        | `yolov8m.pt` | Pretrained COCO weights. No training performed. Chosen by measurement, not size — see below. |
| `IMGSZ`             | `640`        | Inference resolution: the 320×240 source upscaled 2×. Small distant vehicles are missed at native resolution. Shared with `analyze_traffic.py`. |
| `DETECT_CONF`       | `0.15`       | Confidence detection actually runs at. Permissive, so the cache is a superset either stage can filter down from without re-running inference. |
| `CONF_THRESH`       | `0.25`       | Confidence **calibration** filters up to — see "two thresholds" below. |
| `VEHICLE_CLASS_IDS` | `{2,3,5,7}`  | COCO ids for car, motorcycle, bus, truck.                                                             |

#### Model size: bigger is worse here

Every model was run over all 254 clips and scored on held-out congestion accuracy.
The relationship is an **inverted U with the peak in the middle**:

| model (@640) | detections | speed coverage | tracks ≤2 frames | accuracy | runtime |
|---|---|---|---|---|---|
| `yolov8s` | 65,896 | 62% | 25% | 92.5% | — |
| **`yolov8m`** | **78,563** | **71%** | **18%** | **93.3%** | **~3 min** |
| `yolov8l` | 71,904 | 69% | 19% | 92.1% | ~3.5 min |
| `yolov8x` | fewest — ~20% below `yolov8s` | — | — | — | slowest |

Larger models are better calibrated and therefore **more conservative**. On blurry,
low-resolution, out-of-distribution footage they decline exactly the marginal
detections this task depends on. `yolov8m` wins on detections, fragmentation, speed
coverage, accuracy *and* runtime simultaneously.

Resolution behaves the same way. Running at `IMGSZ=1280` was measured over all 254
clips and was worse on every axis — 70,269 detections (vs 78,563), 25% fragmentation
(vs 18%), 92.5% accuracy (vs 93.3%) — at roughly 4× the runtime. Beyond 2× upscaling,
the detector is fed interpolated pixels rather than information.

*(A cautionary note on method: an early version of this comparison used 4 hand-picked
clips and reported that 1280 halved fragmentation. It did not — those clips were the
demo selection, i.e. the easy cases. Every figure above is over all 254 clips.)*

#### Two confidence thresholds, because the stages want opposite things

Detection runs once at `DETECT_CONF = 0.15`; calibration then filters up to
`CONF_THRESH = 0.25`. Placing a lane centerline needs **well-localised** boxes, since a
loose box shifts the centroid and blurs the per-band cluster. Measured max lane-fit
residual against this threshold:

| filter | detections used | max lane residual | calibrated region |
|---|---|---|---|
| 0.15 | 7321 | 15.2 px | y=[78,198] |
| **0.25** | 5241 | **6.9 px** | y=[78,198] |
| 0.35 | 3803 | 4.3 px | shrinks to [78,186] |
| 0.50 | 2034 | 1.7 px | collapses to [78,126] |

Stricter keeps improving the fit but starves the far bands and shortens the calibrated
region, so 0.25 is where the fit is clean and the region still full length. Analysis
keeps the permissive 0.15 instead, because there a missed vehicle is lost outright
while localisation noise averages out in the speed fit.

### Sampling

| Parameter                   | Value                             | Meaning                                                                      |
| --------------------------- | --------------------------------- | ---------------------------------------------------------------------------- |
| `SEED`                    | `42`                            | Seeds clip selection; makes the sample reproducible.                         |
| `LANE_SAMPLE_LIGHT_CLIPS` | `20`                            | Light clips sampled — these carry the lane geometry.                        |
| `CONTEXT_CLIPS_PER_CLASS` | `8`                             | Medium/heavy clips sampled — these only feed the scale curve and ROI shift. |
| `CLIP_SAMPLE`             | `{light:20, medium:8, heavy:8}` | The combined per-label sample spec.                                          |

### Geometry

| Parameter                   | Value    | Meaning                                                                                                       |
| --------------------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `NUM_LANES`               | `5`    | Lanes on this carriageway.                                                                                    |
| `BAND_HEIGHT_PX`          | `12`   | Height of each horizontal analysis band. Smaller gives finer resolution but fewer detections per band.        |
| `MIN_DETECTIONS_PER_BAND` | `40`   | Below this, a band's cluster centers are too noisy to trust and the band is not measured.                     |
| `SPAN_TOLERANCE_PX`       | `3.0`  | Slack allowed when testing span monotonicity, so measurement noise does not break a run.                      |
| `LANE_POLY_DEGREE`        | `2`    | Degree of the lane centerline polynomials. Degree 2 captures the road's curve without over-fitting 11 points. |
| `OUTLIER_GAP_PX`          | `25.0` | Gap between neighbouring`x` positions that splits a band into separate clusters for outlier rejection.      |

### Scale

| Parameter                             | Value   | Meaning                                                                                                           |
| ------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `LANE_WIDTH_M`                      | `3.66` | Standard US Interstate lane width (12 ft), used as the scale reference. Scales all lateral distances, but cancels out of speed and density — see §8. |
| `MIN_CAR_DETECTIONS_FOR_SCALE_BAND` | `10`  | Minimum car detections for a band to contribute to the implied-car-width consistency check.                       |

---

## 6. Function reference

### Dataset access

| Function                                         | Purpose                                                                                                                                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_info_table()`                            | Parses the dataset's`info.txt` into a DataFrame: filename, date, timestamp, direction, day/night, weather, start frame, frame count, congestion label.                                    |
| `sample_clips(info, per_label, seed)`          | Selects clips per congestion label using`per_label` counts. Raises `KeyError` on a label with no configured count rather than silently sampling zero.                                   |
| `detect_in_clip(model, filename, start_frame)` | Runs YOLO on one clip, skipping frames before`start_frame`. Returns one dict per vehicle detection with class, confidence, center, and box dimensions, mapped back to native coordinates. |
| `load_patch_roi_rect()`                        | Loads the recovered patch rectangle. Raises if`recover_patch_roi.py` has not been run.                                                                                                    |
| `rect_to_polygon(rect)`                        | Converts`(x0, y0, w, h)` to a 4-vertex polygon.                                                                                                                                           |
| `filter_to_rect(df, rect)`                     | Keeps only detections inside the rectangle.                                                                                                                                                 |

### Banding, outlier rejection, clustering

| Function                                           | Purpose                                                                                                                                                                                          |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `band_key(y)`                                    | Maps a`y` coordinate to its band index.                                                                                                                                                        |
| `band_center_y(band)`                            | Maps a band index back to its center`y`.                                                                                                                                                       |
| `filter_main_cluster(df, gap)`                   | Per band, keeps only the largest contiguous run of`x` positions — the off-road outlier filter of §3.                                                                                         |
| `hierarchical_1d_labels(values, k)`              | Average-linkage agglomerative clustering on 1-D data, with labels remapped so that**0 is leftmost and k−1 is rightmost**. This ordering is what makes lane index comparable across bands. |
| `measure_bands(df, num_lanes)`                   | Clusters every band that has enough detections and yields`num_lanes` distinct clusters. Returns one record per band: `band`, `y`, `n`, `centers`, `span`, `min_gap`.               |
| `longest_perspective_consistent_run(bands, tol)` | Selects the longest run of adjacent bands with monotonically increasing span — the separability test of §5. Returns the accepted subset.                                                       |

### Lane and ROI geometry

| Function                                                          | Purpose                                                                                                                                                                                         |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit_lane_polys(run, num_lanes, degree)`                        | Fits one polynomial per lane through the accepted band centers. Raises if there are too few bands for the requested degree.                                                                     |
| `lane_fit_residuals(run, polys)`                                | Maximum absolute residual per lane, in pixels — the fit-quality metric reported and stored.                                                                                                    |
| `derive_roi(df, polys, y_min, y_max, rect, enclose_percentile)` | Builds the ROI by shifting the outer lane curves outward to enclose detections, then clipping to the patch box. Returns sample`y` values, left and right curves, and the two shift distances. |
| `fit_scale_curve(df, y_min, y_max)`                             | Fits metres-per-pixel against`y` from median car box widths.                                                                                                                                  |

### Output

| Function            | Purpose                                                                                            |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| `visualize(...)`  | Writes the five diagnostic images of §7.                                                          |
| `main(use_cache)` | Orchestrates the pipeline, validates the detection cache, prints a summary, and writes the config. |

---

## 7. Output

### `config/lanes_cctv052.json`

| Key                               | Type               | Meaning                                                                                                                                                                                        |
| --------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_comment`                      | string             | Self-describing summary of how the file was produced.                                                                                                                                          |
| `frame_width`, `frame_height` | int                | Native frame size the coordinates refer to.                                                                                                                                                    |
| `calibrated_y_range`            | `[y_min, y_max]` | **The region lanes are valid in.** Downstream code must not use lane assignment outside this range.                                                                                      |
| `roi_polygon`                   | list of`[x, y]`  | The measurable road region, 120 vertices (60 samples per side).                                                                                                                                |
| `roi_polygon_note`              | string             | How the polygon was derived.                                                                                                                                                                   |
| `roi_edge_shift_px`             | `{left, right}`  | Outward shift applied to each outer lane curve.                                                                                                                                                |
| `patch_roi_box`                 | polygon            | The recovered patch rectangle used to bound detections.                                                                                                                                        |
| `patch_roi_box_note`            | string             | Provenance of that rectangle.                                                                                                                                                                  |
| `num_lanes`                     | int                | Lane count.                                                                                                                                                                                    |
| `lane_polynomials`              | 5 × 3 floats      | `numpy.polyfit` coefficients. Evaluate with `numpy.polyval(coeffs, y)` → `x`.                                                                                                           |
| `lane_polynomial_degree`        | int                | Degree of those polynomials.                                                                                                                                                                   |
| `lane_curve_note`               | string             | How to evaluate them, and their validity range.                                                                                                                                                |
| `lane_max_fit_residual_px`      | 5 floats           | Per-lane fit quality.                                                                                                                                                                          |
| `scale_model`                   | `{h, y_horizon, rms_residual_m_per_px}` | LATERAL scale: `metres_per_pixel(y) = h / (y - y_horizon)`.                                                                                                        |
| `scale_note`                    | string             | Derivation and the reference used.                                                                                                                                                             |
| `along_road_note`               | string             | States that along-road position is `f · metres_per_pixel(y)` and that `f` is unknown, so ratios are exact and absolutes need an anchor.                                                    |
| `implied_car_box_width_m`       | float              | Consistency check: what the median car box measures under this scale (~2.42 m vs a real car's ~1.8 m — the oblique-view inflation).                                                            |
| `implied_car_box_width_note`    | string             | Explains that this is an output, not an input.                                                                                                                                                 |
| `calibration_meta`              | object             | Full provenance: model, thresholds, seed, sample spec, the exact light clips used, all geometry parameters, band counts, excluded bands, detection totals, class breakdown, and UTC timestamp. |

**Evaluating a lane centerline:**

Prefer `scripts/scene.py`, which wraps all of this:

```python
import sys; sys.path.insert(0, "scripts")
from scene import Scene

s = Scene.from_config()
s.lane_x(2, 150.0)              # x of lane 2 at row y=150
s.metres_per_pixel(150.0)       # lateral scale there
s.lane_of(200.0, 150.0)         # lane index, or -1 outside the ROI
s.along_road_raw_delta(150, 160)  # along-road distance, in metres / f
```

Or read the config directly:

```python
import json, numpy as np
cfg = json.load(open("config/lanes_cctv052.json"))
y_min, y_max = cfg["calibrated_y_range"]
lane2_x_at_y150 = np.polyval(cfg["lane_polynomials"][2], 150.0)   # 150 is in range
sm = cfg["scale_model"]
metres_per_pixel = sm["h"] / (150.0 - sm["y_horizon"])
```

### Diagnostic images (`outputs/calibration/`)

| File                         | Shows                                                                                                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `roi_lanes_scatter.png`    | All pooled detections, the rejected off-road ones, the patch box, and the final ROI.                                                                              |
| `lane_clusters.png`        | Every band's lane centers — kept bands coloured by lane, excluded bands as grey crosses — with the fitted curves.                                               |
| `band_span_diagnostic.png` | Span against`y`. **The justification for where the calibrated region ends**: accepted bands form a monotonic line, excluded bands visibly depart from it. |
| `roi_lanes_on_frame.png`   | Lanes, band centers, and both boundaries drawn over a real frame, upscaled 3×.                                                                                   |
| `scale_curve.png`          | Metres-per-pixel against`y`.                                                                                                                                    |
| `detections.csv`           | The pooled detections, cached so fitting can be re-run without repeating inference.                                                                               |

### Result on camera cctv052

|                             |                                                        |
| --------------------------- | ------------------------------------------------------ |
| Clips sampled               | 36 (20 light, 8 medium, 8 heavy)                       |
| Detections pooled           | 22,433 detected → 16,811 at conf ≥ 0.25 → 16,285 inside the ROI |
| Bands measured              | 15                                                     |
| Bands accepted              | 11                                                     |
| **Calibrated region** | **y ∈ [78, 198]**                               |
| Max lane fit residual       | 2.5 – 6.6 px                                          |
| Lateral scale               | 0.238 m/px at y=78 (far) → 0.089 m/px at y=198 (near); `h`=17.48, `y_horizon`=2.6, RMS 0.0039 |
| Implied car-box width       | 2.42 m (consistency check; a real car is ~1.8 m)      |
| ROI edge shift              | 10.6 px left, 16.9 px right                           |

Four bands were excluded, and each boundary has an identified physical cause:

- **y = 54, 66 (far field)** — an on-ramp merges from the right. The clustering absorbs
  the merging traffic as a spurious fifth lane, and the span jumps from 64.0 px at the
  accepted band y=78 to 105.3 px at y=66 — wider despite being *further away*, which
  perspective forbids.
- **y = 210, 222 (near field)** — the carriageway runs off the **right edge of the
  frame** (the rightmost lane center reaches x=314.0 in a 320 px-wide frame), truncating
  the outer lane so the span collapses from 147.9 px at y=198 to 134.1 px at y=210.

Neither cause was supplied to the algorithm; both were isolated by the span test alone.

---

## 8. Assumptions and limitations

1. **Absolute speeds are anchored, not measured.** The along-road scale contains an
   unrecoverable focal length `f` (§4 step 8b). Reporting speeds in km/h therefore
   requires choosing `f` — in practice by asserting that free-flowing traffic moves at
   about the posted limit — which makes the reference case true by construction.
   **Absolute speeds should be read as indicative, not survey-grade.** Speeds expressed
   as a *fraction of free-flow* carry no such caveat: `f` cancels exactly.
2. **The lateral scale inherits the lane-width assumption.** Lanes are taken to be
   3.66 m. If this stretch is built to a different width, every lateral distance scales
   with it. Speed and density are unaffected, since any constant factor in
   `metres_per_pixel` is absorbed by the `f` anchor.
3. **One scale per row is itself an approximation.** A single `metres_per_pixel(y)`
   assumes the road is flat and locally straight. This road curves, and the two
   independent references disagree in shape by ~18% across the region (lane spacing vs
   car widths), which bounds how well that assumption holds. A full homography would be
   the principled fix and would need a surveyed reference length.
4. **Coverage is partial by design.** Roughly the middle 60% of the patch box is
   calibrated. Vehicles outside it are still detectable but are not lane-assigned.
5. **Lane geometry comes from light traffic only.** Cluster centers in dense traffic
   reflect where vehicles are queued rather than where lanes lie.
6. **The camera must be static.** All geometry is fixed to this viewpoint; any pan,
   zoom, or repositioning invalidates the calibration.
7. **Congestion sampling is deliberately unrepresentative.** 20/8/8 does not match the
   dataset's true 165/45/44 split. It has no effect on the lane-derived scale, and only
   a marginal one on the implied-car-width check.
8. **Lane count is fixed** at `NUM_LANES = 5`. The script does not infer it.

## 9. Applying it to another camera

The method carries over to any fixed camera viewing a straight-ish multi-lane
carriageway, since nothing in the separability test is specific to this scene. Expect
to revisit:

- `NUM_LANES` — must match the new road.
- The patch ROI box — `recover_patch_roi.py` is specific to this dataset's supplementary
  files. Another camera needs a bounding region supplied another way.
- `BAND_HEIGHT_PX` and `MIN_DETECTIONS_PER_BAND` — both scale with frame resolution and
  traffic volume.
- `OUTLIER_GAP_PX` — depends on apparent vehicle spacing, so also resolution-dependent.

`SPAN_TOLERANCE_PX` and the perspective test itself should transfer unchanged; they
encode a property of perspective projection rather than of this road.
