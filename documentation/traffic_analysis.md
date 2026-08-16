# Traffic Analysis Pipeline

How vehicles are detected, tracked, measured, and classified — and how accurate the
result is.

This is stage 2 of the project. Stage 1 works out the road geometry and is documented
separately in [`scene_calibration.md`](scene_calibration.md); everything here depends on
its output.

| Script | Role |
|---|---|
| `scripts/scene.py` | Shared module. Applies the calibration: ROI membership, lane assignment, px→m conversion. |
| `scripts/analyze_traffic.py` | Detects and tracks vehicles across all clips, computes traffic parameters, writes three tables. |
| `scripts/evaluate_congestion.py` | Classifies each clip's congestion level and scores it against ground truth. |

---

## 1. Running it

```bash
# stage 1 (once) - see scene_calibration.md
uv run python scripts/recover_patch_roi.py
uv run python scripts/calibrate_scene.py

# stage 2
uv run python scripts/analyze_traffic.py            # ~254 clips, GPU, several minutes
uv run python scripts/analyze_traffic.py --limit 6  # quick smoke test
uv run python scripts/evaluate_congestion.py        # seconds; reads the CSVs
```

`analyze_traffic.py` is the expensive stage and writes CSVs; `evaluate_congestion.py`
is cheap and reads them. That split is deliberate — the classification can be
re-run and iterated on without repeating inference over 13,000 frames.

---

## 2. `scene.py` — applying the calibration

Not runnable. It exists so that every consumer of the calibration shares one
implementation: if the metrics pipeline and the demo renderer each rolled their own
lane assignment, the annotated video could disagree with the reported numbers without
anything visibly breaking.

### The distance problem it solves

This is the subtlest part of the project and the reason the module exists at all.

The calibration provides **lateral** metres-per-pixel, `mpp(y) = h / (y - y_horizon)`.
Vehicles, however, travel **along** the road, and under perspective those two scales are
not the same. For a pinhole camera viewing a flat plane, along-road position is

```
D(y) = f · mpp(y)
```

where `f` is the camera's focal length in pixels. `f` cannot be recovered from this
dataset — there is no camera metadata and no surveyed reference length — so along-road
positions are carried in units of `metres / f`, which the module calls **raw**.

Two consequences shape the whole pipeline:

- **Ratios are exact.** `f` cancels in any ratio, so a speed expressed as a fraction of
  free-flow needs no anchor whatsoever. The congestion classifier uses exactly this, and
  is therefore free of any absolute-calibration assumption.
- **Absolute speeds need an anchor**, supplied by `anchor_focal_from_speed()`, and are
  assumption-based rather than measured.

Using the lateral scale directly for along-road motion — the obvious shortcut — would be
wrong by a factor of `f / (y - y_horizon)`, which varies about **fourfold** across the
calibrated region. The error would therefore be not just large but *different at every
row*. The module deliberately offers no method that does this.

### `Scene` API

Construct with `Scene.from_config()`, which reads `config/lanes_cctv052.json`.

| Method | Returns |
|---|---|
| `lane_x(lane, y)` | x of a lane centerline at row y. |
| `roi_edges(y)` | Left and right x bounds of measurable road at row y. |
| `in_calibrated_rows(y)` | Whether y is inside the calibrated band. |
| `in_roi(x, y)` | Whether a point is on measurable road. |
| `lane_of(x, y)` | Nearest lane index, or **-1** outside the ROI. |
| `metres_per_pixel(y)` | Lateral scale. Correct for widths; **not** for along-road motion. |
| `along_road_raw(y)` | Along-road position, in metres / f. |
| `along_road_raw_delta(y1, y2)` | Along-road distance between two rows, in metres / f. |
| `anchor_focal_from_speed(raw, kmh)` | Pins `f` from an assumed free-flow speed. |
| `raw_to_metres(raw)` | Converts raw → metres. Raises unless `f` has been anchored. |
| `road_length_raw()` | Along-road length of the calibrated region, in metres / f. |
| `lane_span_px(y)` | Pixel distance between the outer lane centerlines at row y. |
| `mean_lane_width_m()` | Mean lane spacing in metres — a calibration sanity check. |

`raw_to_metres` raising rather than silently assuming a default is intentional: it makes
it impossible to report a metric distance without having explicitly chosen an anchor.

---

## 3. `analyze_traffic.py` — detection, tracking, parameters

### Algorithm

**Detect and track.** YOLOv8m runs over each clip with ByteTrack maintaining identities
between frames. Frames are processed at `IMGSZ=640` (the native 320×240 upscaled 2×),
because vehicles in the far field are only a few pixels across at native resolution.
Both the model and the resolution were selected by measurement over all 254 clips, and
both larger models and larger inputs proved worse — see `scene_calibration.md` §5.
Frame 0 of every clip is skipped — it is corrupted throughout this dataset.

Tracking is what makes the rest possible: without persistent identities there is no
way to measure a speed, a trajectory, or a lane change, only per-frame counts.

**Restrict to measurable road.** Every detection is tested with `scene.in_roi()` and
discarded if outside. Nothing is measured where the calibration does not apply.

**Assign a lane.** Each surviving detection gets the nearest lane centerline.

**Estimate speed.** For each track, along-road position is regressed against time by
least squares, and the slope is the speed. A fit is used rather than differencing the
endpoints so that one noisy detection cannot dominate. A track needs
`MIN_TRACK_SAMPLES` frames to qualify; about **70%** of tracks do. The rest are too
short-lived, and are counted but carry no speed.

The gate is on **duration, never displacement**. An earlier version also required ~6 px
of movement, to suppress fits on jittering boxes — but that discarded 13.8% of
heavy-traffic tracks against 3.4% of light ones, because stopped vehicles are what heavy
congestion *is*. Filtering on movement selectively removed the slowest vehicles and
biased heavy-traffic speeds upward, exactly where near-zero speed is most diagnostic.
Removing it raised held-out accuracy 93.3% → 93.7% and heavy recall 79.5% → 84.1%.

**Aggregate.** Per-track records roll up into per-clip summaries.

### Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `FPS` | `10.0` | Dataset frame rate, per `README_TRAFFICDB`. Converts frame indices to seconds. |
| `IMGSZ` | `cal.IMGSZ` (640) | Inference resolution; 2× the native frame. Imported so both stages match. |
| `CONF_THRESH` | `cal.DETECT_CONF` (0.15) | **Deliberately more permissive than calibration's 0.25.** Here a missed vehicle is lost outright, while loose box edges wash out of the least-squares speed fit. Calibration makes the opposite trade — see `scene_calibration.md` §5. |
| `TRACKER` | `bytetrack.yaml` | ByteTrack — association only, no appearance model, nothing trained. |
| `MIN_TRACK_SAMPLES` | `6` | Minimum detections before a speed fit is attempted. The **only** gate — see above on why movement is deliberately not one. |

The model weights, input size and vehicle class list are imported from
`calibrate_scene.py` rather than redeclared, so both stages see the same detector. The
confidence threshold is the one thing that intentionally differs.

### Functions

| Function | Purpose |
|---|---|
| `track_clip(model, scene, filename, start_frame)` | Runs detect+track on one clip; returns in-ROI detections tagged with track id, lane, and along-road position. |
| `summarise_tracks(det)` | One row per track: class, dominant lane, lane changes, fitted speed. |
| `summarise_clips(det, tracks, info, scene)` | One row per clip: counts, class mix, speed statistics, lane occupancy, joined to the ground-truth label. |
| `main(limit)` | Iterates clips, concatenates, writes the three tables. |

Lane changes are counted on the **de-duplicated** lane sequence, so per-frame jitter
across a lane boundary is not mistaken for a manoeuvre.

### Output

**`outputs/analysis/detections.csv`** — one row per tracked detection per frame (78,563)

`clip, frame_idx, t_s, track_id, cls, conf, x_center, y_center, width_px, height_px, lane, along_raw`

**`outputs/analysis/tracks.csv`** — one row per vehicle track (4,630)

`clip, track_id, cls, n_frames, duration_s, y_span_px, lane, lane_changes, speed_raw, mean_conf`

**`outputs/analysis/clips.csv`** — one row per clip (253; one clip produced no tracks at all)

`clip, congestion_true, weather, date, hour, n_tracks, n_tracks_with_speed,
mean_vehicles_in_roi, max_vehicles_in_roi, median_speed_raw, mean_speed_raw,
p15_speed_raw, total_lane_changes, n_car, n_truck, n_bus, n_motorcycle, lane0_share …
lane4_share`

`mean_vehicles_in_roi` is the density proxy: the average number of vehicles visible in
the region at any instant. It is preferred to a raw track count because it does not
depend on how long each vehicle happens to stay in frame.

**All speeds in these tables are raw** (metres / f per second). See §2.

---

## 4. `evaluate_congestion.py` — classification and accuracy

### Why this is the only metric that can be validated

The dataset carries **no per-vehicle ground truth** — no boxes, counts, speeds, or lane
labels (established in `docs/data_profile.md`). The single light/medium/heavy label per
clip is the only thing any of our outputs can be scored against. Every other parameter
the pipeline produces is a real measurement but an unvalidatable one.

### Method

`README_TRAFFICDB` defines the labels relative to free flow — "free-flowing", "traffic at
reduced speed", "stopped or very slow speed" — so the classifier is built in those terms:

1. Divide each clip's median track speed by a **free-flow reference** (the 85th
   percentile of training-clip speeds).
2. Split the resulting ratio into three classes with **two thresholds**, chosen by
   searching for the best training accuracy.
3. Where a clip has no speed estimate, fall back to the count-only classifier.

Two properties make the result trustworthy:

- **No absolute-calibration dependency.** The feature is a speed *ratio*, so the unknown
  focal length cancels exactly (§2).
- **No leakage.** The free-flow reference and both thresholds are fitted only on training
  clips, using the dataset's **own published train/test folds** (`EvalSet.mat`) — the same
  splits the original papers used. Verified to be a true partition: the four test folds
  are pairwise disjoint and cover all 254 clips, so every clip is predicted exactly once
  while held out.

A **count-only classifier** is fitted identically as a baseline, to test whether the
speed machinery earns its keep.

### Functions

| Function | Purpose |
|---|---|
| `load_clip_table()` | Per-clip metrics joined onto the full clip list, so clips with no detections appear as rows with missing features rather than disappearing. |
| `load_splits()` | The dataset's 4 train/test folds, mapped from 1-based indices to clip names. |
| `fit_thresholds(x, y, higher_is_lighter)` | Searches observed feature values for the two cut points maximising training accuracy. |
| `predict(x, thresholds, higher_is_lighter)` | Applies thresholds. The flag is True for speed (fast = free-flowing), False for count (busy = congested). |
| `confusion(y_true, y_pred)` | 3×3 confusion matrix. |
| `run_fold(train, test)` | Fits reference and thresholds on train, scores on test, returns both classifiers' results. |
| `make_plots(...)` | The two figures below. |

### Results

**93.7%** held-out accuracy (sd 1.9% across folds), against **79.9%** for count-only.

| | speed classifier | count-only |
|---|---|---|
| accuracy | **93.7%** | 79.9% |
| light recall | 98.2% | 99.4% |
| **medium recall** | **86.7%** | **46.7%** |
| heavy recall | 84.1% | 40.9% |

Two things stand out:

**Counting cannot distinguish medium from heavy.** The baseline files 49% of medium
clips as heavy, and its medium recall collapses to 46.7%. The two classes sit at 11.8 and
11.0 median vehicles in the region — medium is if anything the *busier* of the two, so
counting cannot even order them correctly. By speed they separate cleanly (0.44 vs 0.23
of free-flow). This is the traffic-flow *fundamental diagram* visible in our own data: past
jam density the road cannot hold more vehicles, so further congestion shows up as
vehicles slowing rather than accumulating. It is also the concrete payoff for the
perspective-correct speed work in §2 — an isotropic approximation would have blurred
precisely this boundary.

**Every error is between adjacent classes.** The speed classifier never confuses light
with heavy in either direction.

For reference, fitting the thresholds on all 254 clips and scoring on those same clips
gives 95.2%. The **1.5-point gap** to the held-out figure is the optimism that
cross-validation removes.

### Output

| File | Contents |
|---|---|
| `outputs/evaluation/predictions.csv` | Per-clip held-out prediction from both classifiers (254 rows). |
| `outputs/evaluation/confusion_matrix.csv` | Pooled confusion matrix. |
| `outputs/evaluation/confusion_matrices.png` | Both classifiers side by side, coloured by recall. |
| `outputs/evaluation/feature_separation.png` | Each clip by feature, grouped by true label — shows directly why speed separates and counting does not. |

Charts follow the project's visualisation rules: congestion is an *ordered* variable, so
it uses a single-hue ordinal ramp rather than unrelated categorical hues, validated with
the palette validator (`--ordinal`, all checks pass; pairwise separation well clear of
the colour-vision floors). Because the lightest step sits below the 3:1 contrast
threshold, every class is directly labelled and the CSVs serve as the table view.

---

## 5. Assumptions and limitations

1. **Absolute speeds are anchored, not measured** (§2). Ratios are exact; km/h figures
   require assuming a free-flow speed and should be read as indicative. The 93.7%
   accuracy does **not** depend on this.
2. **30% of tracks yield no speed** — 1,367 last fewer than `MIN_TRACK_SAMPLES` frames.
   They still contribute to counts and lane occupancy, and clips where *no* track yields
   a speed fall back to the count classifier. Tracker fragmentation is part of this: 18%
   of tracks last two frames or fewer, and a vehicle re-acquired under a new id counts
   twice in `n_tracks` (though not in the per-frame counts the classifier uses).

   One residual bias remains, in the opposite direction to the one removed: fast vehicles
   cross the region in fewer frames, so the duration gate drops proportionally more of
   them. Light-traffic coverage is 62% against 78-80% for medium and heavy, which would
   bias light speeds slightly *downward*. It does not affect separability — light recall
   is 98.2% — but the free-flow reference is drawn from that population.
3. **One clip produced no tracks at all** and appears with zeroed features. An empty
   road is unambiguously light traffic, so the count fallback handles them correctly.
4. **Only the calibrated region is measured.** Vehicles outside it are invisible to every
   metric here, so counts are counts *within the region*, not counts on the road.
5. **Tracking is not evaluated.** There is no ground-truth track data, so identity
   switches and fragmentation are unmeasured. A switch would corrupt that track's speed.
6. **The classifier's design was not chosen blind.** The thresholds are honestly fitted
   and held-out validated, but the choice of feature (speed ratio, 85th-percentile
   reference) was made by us with the whole dataset in view. That follows the published
   label definitions rather than test performance, but it is not a fully blind protocol.
7. **Detection quality is unmeasured** for the same reason — no ground-truth boxes. Class
   labels come from pretrained COCO weights with no domain fine-tuning, so the
   car/truck/bus split (86.9% / 11.0% / 2.1% of detections) is plausible but unvalidated.
   Motorcycles are vanishingly rare: **12 detections across 78,563, forming 3 tracks**
   (0.02%). Under the previous, weaker detector there were none at all. Whether that
   reflects the road or a detector blind spot at this resolution cannot be determined
   from this data, so `n_motorcycle` should be treated as unreliable either way.
