# Traffic Flow Analysis with Computer Vision

Detects, tracks and measures vehicles in highway camera footage, then classifies how
congested each clip is — and scores that classification against ground truth.

**Live demo:** Open index.html in the output of the pipeline after running scripts/analyze_traffic.py and render_demo.py

| | |
|---|---|
| **Congestion accuracy** | **93.7%**, held out on the dataset's own 4 train/test folds |
| Baseline (counting vehicles only) | 79.9% |
| Clips analysed | 254 (the full dataset) |
| Vehicles tracked | 4,630, from 78,563 detections |
| Training performed | **None** — pretrained YOLOv8 weights throughout |

---

## What it measures

For every clip, on a calibrated region of road:

- **Vehicle count and classification** — car / truck / bus / motorcycle, per frame and per clip
- **Speed** — per vehicle, from along-road motion under a perspective model
- **Traffic density** — mean vehicles present in the measured region
- **Lane occupancy** — every vehicle assigned to one of 5 lanes, with per-lane counts
- **Lane changes** — counted per track
- **Congestion level** — light / medium / heavy per clip, scored against the ground-truth label

---

## Quick start

Requires a CUDA GPU (detection is set to device 0) and Python 3.11+.

```bash
# 1. install  (uv is recommended; pip works too)
uv sync                          # or:  pip install -r requirements.txt

# 2. fetch the dataset (~90 MB, needs Kaggle credentials)
uv run python scripts/download_dataset.py

# optional: pre-fetch the detector weights (~50 MB). Ultralytics downloads them
# automatically on first use, so this is only needed to run offline afterwards.
uv run python scripts/download_model.py

# 3. run the pipeline
uv run python scripts/recover_patch_roi.py     # ~1 min   locate the study region
uv run python scripts/calibrate_scene.py       # ~2 min   work out lane geometry
uv run python scripts/analyze_traffic.py       # ~3 min   detect, track, measure
uv run python scripts/evaluate_congestion.py   # seconds  classify and score
uv run python scripts/render_demo.py           # ~1 min   build the demo page
```

Then open `outputs/demo/index.html`.

To try it quickly on a handful of clips: `uv run python scripts/analyze_traffic.py --limit 6`.

### Models and libraries

| | |
|---|---|
| Detector | **YOLOv8m**, pretrained COCO weights via [Ultralytics](https://docs.ultralytics.com/) |
| Tracker | **ByteTrack** (bundled with Ultralytics) — association only, no learned component |
| Weights | Downloaded automatically on first use, from a release URL pinned to the `ultralytics` version. Nothing is trained or fine-tuned. |

Run the scripts from the repository root: Ultralytics resolves a bare weights filename
against the working directory. `scripts/download_model.py` pre-fetches the weights if you
need the pipeline itself to run without network access.

`yolov8m` was chosen by measurement, not by size — see [Why the middle-sized model](#results-worth-knowing).

---

## Repository layout

```
scripts/
  download_dataset.py     fetch the dataset from Kaggle
  download_model.py       pre-fetch the detector weights (optional)
  recover_patch_roi.py    recover the study region from the dataset's own .mat files
  calibrate_scene.py      derive lane geometry and the pixel->metre scale
  scene.py                shared module: applies the calibration (not runnable)
  analyze_traffic.py      detect + track + measure -> three CSV tables
  evaluate_congestion.py  classify congestion, score it, plot it
  render_demo.py          build the self-contained demo page
config/
  lanes_cctv052.json      the calibration: lanes, road region, scale
documentation/
  scene_calibration.md    how the geometry is derived, and why
  traffic_analysis.md     how vehicles are measured and classified
outputs/                  generated; not in version control
  calibration/  analysis/  evaluation/  demo/
```

Each stage writes files the next one reads, so any stage can be re-run alone. The
expensive step (detection) is separated from the cheap ones (classification, rendering)
for exactly that reason.

---

## How it works

**1 · Find the measurable road.** Rather than hand-drawing a region, lane geometry is
recovered from where vehicles are actually detected: detections are clustered into lanes
band by band, and the region is cut off where the lanes stop being separable. Perspective
compresses lanes together with distance, and past that point any lane assignment would be
invented rather than measured. The calibrated region is deliberately smaller than the frame.

The cut-off is found without a hand-tuned threshold, by requiring that the road appear
*narrower* the further away it is. On this camera that automatically excluded two regions
for two unrelated physical reasons: an on-ramp merging in at the top, and the carriageway
running off the right edge of the frame at the bottom.

**2 · Detect and track.** YOLOv8m finds vehicles; ByteTrack keeps their identity between
frames. Without persistent identities there is no speed, no trajectory and no lane change —
only per-frame counts.

**3 · Measure speed.** Each vehicle's position along the road is regressed against time.
The conversion from pixels to distance uses a perspective model fitted to the scene, not a
flat pixels-per-metre constant, because a fixed constant is wrong by a factor that varies
across the frame.

**4 · Classify congestion.** Each clip's median vehicle speed is split into three classes by
two thresholds. Both the thresholds and the reference speed are fitted **only on training
clips**, then applied to held-out clips, using the four train/test folds the dataset itself
publishes.

Full detail in [`documentation/`](documentation/).

---

## Results worth knowing

**Counting vehicles cannot tell medium from heavy traffic.** The count-only baseline
misfiles 49% of medium clips as heavy, and its medium recall collapses to 47%. The two
classes sit at 11.8 and 11.0 median vehicles in the region — medium is if anything the
*busier* one, so counting cannot even order them correctly.

Speed separates them cleanly. This is the traffic-flow *fundamental diagram* showing up in
the data: past jam density the road cannot hold more vehicles, so further congestion appears
as vehicles slowing, not accumulating.

| | speed classifier | counting only |
|---|---|---|
| accuracy | **93.7%** | 79.9% |
| light recall | 98.2% | 99.4% |
| **medium recall** | **86.7%** | **46.7%** |
| heavy recall | 84.1% | 40.9% |

**No error is severe.** No light clip was ever called heavy, or the reverse.

**A bigger detector is worse here.** Model size traces an inverted U peaking in the middle:
`yolov8m` beats `yolov8s`, `yolov8l` *and* `yolov8x`, and is also the fastest of them.
Larger models are better calibrated and therefore more conservative — on blurry, 320×240,
out-of-distribution footage they decline exactly the marginal detections this task needs.
Raising the input resolution to 1280 was likewise worse on every axis.

---

## Assumptions and limitations

Stated plainly, because several of these bound what the numbers can mean.

1. **The dataset has no per-vehicle ground truth** — no boxes, counts, speeds or lane
   labels. The only thing any output can be scored against is the one light/medium/heavy
   label per clip. Counts, speeds, lane occupancy and trajectories are real measurements but
   **unvalidated** ones.
2. **Absolute speeds are anchored, not measured.** Speed *ratios* are exact, but converting
   to km/h needs one reference, since the camera's focal length is not recorded in the
   dataset. We calibrate against the posted 60 mph limit. The 93.7% accuracy uses ratios
   only and does not depend on this.
3. **Only part of the frame is measured.** Vehicles outside the calibrated region are
   invisible to every metric, so counts are counts *within the region*, not on the road.
4. **Tracking quality is unmeasured** — no ground-truth tracks exist. 18% of tracks last two
   frames or fewer, and a vehicle re-acquired under a new identity is counted twice in the
   track total (though not in the per-frame counts the classifier uses).
5. **The classifier's design was not chosen blind.** Its thresholds are honestly fitted and
   held-out validated, but the choice of feature was made with the whole dataset in view.
6. **One camera, one viewpoint.** All geometry is fixed to this scene; any pan or zoom
   invalidates the calibration.

---

## Dataset

UCSD **TrafficDB** — 254 clips from a fixed WSDOT camera on I-5 at S 188th St, Seattle,
recorded over two days in August 2004. 320×240, 10 fps, roughly 5 seconds each (~22 minutes
in total). Each clip carries a human label of light, medium or heavy traffic.

Chan & Vasconcelos, *"Classification and Retrieval of Traffic Video using Auto-Regressive
Stochastic Processes"*, IEEE Intelligent Vehicles Symposium, 2005.

Distributed on Kaggle as
[`aryashah2k/highway-traffic-videos-dataset`](https://www.kaggle.com/datasets/aryashah2k/highway-traffic-videos-dataset).
