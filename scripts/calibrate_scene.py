"""Offline scene calibration for the fixed cctv052 camera.

Core idea: only calibrate the region where the lanes are actually distinguishable
in the detector's output, and make that region the ROI. Nothing is extrapolated
into the far field, where perspective compresses the lanes together and any lane
assignment (and therefore any speed or occupancy figure) would be invented rather
than measured.

Method:
  1. Pool vehicle detections over a sample of clips, weighted toward light
     traffic since that is what the lane geometry is read from (cached to CSV).
  2. Keep only detections inside the patch ROI box recovered from the dataset's
     own supplementary files (see scripts/recover_patch_roi.py), then drop stray
     off-road detections per row-band with a gap-based outlier filter.
  3. For each horizontal band, cluster detection x-positions into NUM_LANES
     groups (agglomerative, average linkage) and record the lane centers.
  4. Keep the longest run of consecutive bands whose lane span (rightmost minus
     leftmost center) is perspective-consistent: the road must appear narrower
     the further away it is. A band where the span jumps back up has absorbed
     something that is not a lane; a band whose span collapses has lost part of
     the road. On this camera that excludes exactly two regions, for two
     different physical reasons - an incoming merge lane joins from the right
     near the top (y<=66), and the carriageway runs off the right-hand side of
     the frame near the bottom (y>=210), truncating the outer lane. Neither
     needed a hand-tuned threshold to find.
  5. Fit one low-degree polynomial per lane over that run. The run's y-extent is
     the calibrated region - lanes are neither defined nor used outside it.
  6. Derive the ROI from the outer lane curves, shifted outward just enough to
     enclose the detections beyond them, clipped to the patch box.
  7. Fit a pixel -> meter scale curve over the same region, from median detected
     car bbox width against an assumed real car width.

Outputs:
  config/lanes_cctv052.json               - the calibration
  outputs/calibration/detections.csv      - pooled detections (cache/reproducibility)
  outputs/calibration/lane_clusters.png   - band centers + fitted lanes + ROI
  outputs/calibration/roi_lanes_scatter.png
  outputs/calibration/roi_lanes_on_frame.png
  outputs/calibration/band_span_diagnostic.png - why the region ends where it does
  outputs/calibration/scale_curve.png

Usage:
    uv run python scripts/calibrate_scene.py [--no-cache]
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import curve_fit
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "highway-traffic-videos-dataset"
VIDEO_DIR = DATASET_DIR / "video"
OUT_DIR = REPO_ROOT / "outputs" / "calibration"
CONFIG_PATH = REPO_ROOT / "config" / "lanes_cctv052.json"
PATCH_ROI_PATH = OUT_DIR / "patch_roi_recovery.json"

FRAME_W, FRAME_H = 320, 240
# Model and resolution, shared with scripts/analyze_traffic.py. Both were chosen by
# measuring over all 254 clips, not by assuming bigger is better - which is wrong here.
#
# Model size traces an inverted U, peaking in the middle. Held-out accuracy and
# detections at imgsz 640:
#     yolov8s  65,896 dets  92.5%
#     yolov8m  78,563 dets  93.3%   <- best on every axis, and the fastest of the three
#     yolov8l  71,904 dets  92.1%
#     yolov8x  fewest dets (sampled only) - finds ~20% FEWER vehicles than yolov8s
# Larger models are more conservative on this blurry, low-resolution, out-of-
# distribution footage: they decline the marginal detections this task depends on.
MODEL_NAME = "yolov8m.pt"
# 640 = the 320x240 source upscaled 2x. Going to 1280 was measured over all 254 clips
# and was WORSE on every axis: fewer detections (70,269 vs 78,563), more fragmentation
# (25% vs 18% of tracks lasting <=2 frames), lower accuracy (92.5% vs 93.3%), at ~4x
# the runtime. Past 2x the detector is fed interpolated pixels, not information.
IMGSZ = 640
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Two thresholds, because the two stages want opposite things.
#
# Detection runs at the permissive value, so the cache is a superset that either
# stage can filter down from without re-running inference.
#
# Calibration then filters UP: placing a lane centerline needs well-localised boxes,
# since a loose box shifts the centroid and blurs the per-band cluster. Measured
# max lane-fit residual against this threshold: 0.15 -> 15.2px, 0.25 -> 6.9px,
# 0.35 -> 4.3px. Going above 0.25 keeps improving the fit but starves the far bands
# and shrinks the calibrated region (0.35 loses y=198; 0.50 collapses it to y=126),
# so 0.25 is the point where the fit is clean and the region is still full length.
#
# Analysis keeps the permissive value instead (see analyze_traffic.CONF_THRESH):
# there, a missed vehicle is lost outright, whereas per-box localisation noise
# averages out in the least-squares speed fit.
DETECT_CONF = 0.15
CONF_THRESH = 0.25

SEED = 42
# Light clips carry the lane geometry and need volume: in light traffic the
# near-field bands are sparse, and a band skipped for lack of detections gets
# excluded from the calibrated region for the wrong reason. Medium and heavy
# clips only feed the scale curve and the ROI edge shift, both of which are flat
# past ~8 clips per label, so they are sampled thinly.
LANE_SAMPLE_LIGHT_CLIPS = 20
CONTEXT_CLIPS_PER_CLASS = 8
CLIP_SAMPLE = {
    "light": LANE_SAMPLE_LIGHT_CLIPS,
    "medium": CONTEXT_CLIPS_PER_CLASS,
    "heavy": CONTEXT_CLIPS_PER_CLASS,
}

NUM_LANES = 5
BAND_HEIGHT_PX = 12
MIN_DETECTIONS_PER_BAND = 40  # below this a band's cluster centers are too noisy to trust
SPAN_TOLERANCE_PX = 3.0  # slack when checking perspective consistency of the span
LANE_POLY_DEGREE = 2

OUTLIER_GAP_PX = 25.0
# Scale reference. Lane width is used rather than vehicle width: it is an
# engineering standard (US Interstate lanes are 12 ft) rather than a population
# average, it is measured from lane geometry we have already validated, and it is
# immune to how tightly the detector draws boxes. Calibrating from car widths
# instead put the implied lane width at 2.70 m - 26% under standard - because the
# camera sees each car's rear plus part of its side, so its box spans about 2.46 m
# rather than the car's true ~1.8 m width.
LANE_WIDTH_M = 3.66
MIN_CAR_DETECTIONS_FOR_SCALE_BAND = 10  # only for the implied-car-width diagnostic

LANE_COLORS_RGB = ["tab:blue", "tab:green", "gold", "tab:red", "tab:purple"]
LANE_COLORS_BGR = [(255, 0, 0), (0, 200, 0), (0, 200, 200), (0, 0, 200), (200, 0, 128)]


# --------------------------------------------------------------------------- #
# dataset access
# --------------------------------------------------------------------------- #


def load_info_table() -> pd.DataFrame:
    """info.txt columns (tab-separated, per README_TRAFFICDB): filename, date,
    timestamp, direction, day/night, weather, start_frame, num_frames, class, notes."""
    lines = (DATASET_DIR / "info.txt").read_text().splitlines()[1:]
    rows = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 9 or not parts[0].strip():
            continue
        rows.append(
            {
                "filename": parts[0],
                "date": parts[1],
                "timestamp": parts[2],
                "direction": parts[3],
                "day_night": parts[4],
                "weather": parts[5],
                "start_frame": int(parts[6]),
                "num_frames": int(parts[7]),
                "congestion": parts[8],
            }
        )
    return pd.DataFrame(rows)


def sample_clips(info: pd.DataFrame, per_label: dict[str, int], seed: int) -> list[str]:
    """Sample clips per congestion label, with a separate count for each because
    the labels serve different purposes here (see CLIP_SAMPLE)."""
    rng = random.Random(seed)
    chosen: list[str] = []
    for label, group in info.groupby("congestion"):
        if label not in per_label:
            raise KeyError(f"No sample size configured for congestion label {label!r}")
        names = group["filename"].tolist()
        rng.shuffle(names)
        chosen.extend(names[: per_label[label]])
    return chosen


def detect_in_clip(model: YOLO, filename: str, start_frame: int) -> list[dict]:
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{filename}.avi"))
    detections = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx < start_frame:  # frame 0 of every clip is corrupted, see README_TRAFFICDB
            frame_idx += 1
            continue
        # Ultralytics rescales internally and returns native-frame coordinates, so
        # imgsz is the single knob both stages share - no manual upscale, no
        # coordinate conversion to keep in sync between them.
        result = model.predict(frame, imgsz=IMGSZ, conf=DETECT_CONF, verbose=False, device=0)[0]
        for box in result.boxes:
            cls_id = int(box.cls.item())
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "clip": filename,
                    "frame_idx": frame_idx,
                    "cls": VEHICLE_CLASS_IDS[cls_id],
                    "conf": float(box.conf.item()),
                    "x_center": (x1 + x2) / 2,
                    "y_center": (y1 + y2) / 2,
                    "width_px": x2 - x1,
                    "height_px": y2 - y1,
                }
            )
        frame_idx += 1
    cap.release()
    return detections


def load_patch_roi_rect() -> tuple[int, int, int, int]:
    """Rectangle recovered by scripts/recover_patch_roi.py by template-matching
    traffic_patches.mat against the raw video: the crop region the UCSD TrafficDB
    authors themselves used."""
    if not PATCH_ROI_PATH.exists():
        raise RuntimeError(f"{PATCH_ROI_PATH} not found - run scripts/recover_patch_roi.py first")
    return tuple(json.loads(PATCH_ROI_PATH.read_text())["median_rect"])


def rect_to_polygon(rect: tuple[int, int, int, int]) -> list[list[float]]:
    x0, y0, w, h = rect
    return [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]


def filter_to_rect(df: pd.DataFrame, rect: tuple[int, int, int, int]) -> pd.DataFrame:
    x0, y0, w, h = rect
    return df[
        (df["x_center"] >= x0) & (df["x_center"] <= x0 + w)
        & (df["y_center"] >= y0) & (df["y_center"] <= y0 + h)
    ]


# --------------------------------------------------------------------------- #
# banding, outlier rejection, clustering
# --------------------------------------------------------------------------- #


def band_key(y: float) -> int:
    return int(y // BAND_HEIGHT_PX)


def band_center_y(band: int) -> float:
    return (band + 0.5) * BAND_HEIGHT_PX


def filter_main_cluster(df: pd.DataFrame, gap: float = OUTLIER_GAP_PX) -> pd.DataFrame:
    """Drop stray off-road detections (misdetections in trees/background, or the
    opposite carriageway) by keeping, per row-band, only the largest contiguous run
    of x-positions. The real carriageway has far more detections per band than any
    spurious cluster, so this isolates it without hand-picked bounds."""
    keep_idx = []
    for _, sub in df.groupby("_band"):
        ordered = sub.sort_values("x_center")
        vals = ordered["x_center"].to_numpy()
        idxs = ordered.index.to_numpy()
        groups, start = [], 0
        for i in range(1, len(vals)):
            if vals[i] - vals[i - 1] > gap:
                groups.append((start, i))
                start = i
        groups.append((start, len(vals)))
        lo, hi = max(groups, key=lambda g: g[1] - g[0])
        keep_idx.extend(idxs[lo:hi])
    return df.loc[keep_idx]


def hierarchical_1d_labels(values: np.ndarray, k: int) -> np.ndarray:
    """Agglomerative average-linkage clustering on 1D data, relabelled left to
    right. Average linkage rather than k-means (which assumes equal-variance,
    equal-density clusters and drifts toward the busier lanes) or single linkage
    (which lets one stray point chain two lanes together)."""
    n = len(values)
    if n < k:
        return np.zeros(n, dtype=int)
    labels = fcluster(linkage(values.reshape(-1, 1), method="average"), t=k, criterion="maxclust")
    ids = np.unique(labels)
    order = np.argsort([values[labels == c].mean() for c in ids])
    remap = {ids[order[i]]: i for i in range(len(ids))}
    return np.array([remap[c] for c in labels])


def measure_bands(df: pd.DataFrame, num_lanes: int) -> list[dict]:
    """Cluster each band independently and record its lane centers and span."""
    out = []
    for band in sorted(df["_band"].unique()):
        sub = df[df["_band"] == band]
        if len(sub) < MIN_DETECTIONS_PER_BAND:
            continue
        vals = sub["x_center"].to_numpy()
        labels = hierarchical_1d_labels(vals, num_lanes)
        ids = np.unique(labels)
        if len(ids) < num_lanes:
            continue
        # sorted so lane_id is always the left-to-right rank, which is what makes
        # centers[i] comparable across bands
        centers = np.sort(np.array([vals[labels == i].mean() for i in ids]))
        out.append(
            {
                "band": int(band),
                "y": band_center_y(band),
                "n": int(len(sub)),
                "centers": centers,
                "span": float(centers[-1] - centers[0]),
                "min_gap": float(np.diff(centers).min()),
            }
        )
    return out


def longest_perspective_consistent_run(bands: list[dict], tol: float = SPAN_TOLERANCE_PX) -> list[dict]:
    """Keep the longest run of consecutive bands whose lane span grows with y.

    Under perspective the road must look narrower the further away it is, so span
    must increase monotonically as y increases (toward the camera). A band that
    breaks this has not measured the road: either it absorbed something that is
    not a lane (a merge lane inflates the span) or it lost part of the road (the
    span collapses, e.g. once the carriageway starts running off the side of the
    frame). Selecting the longest self-consistent run excludes both without
    needing a threshold tuned to this particular camera.
    """
    if not bands:
        return []
    ordered = sorted(bands, key=lambda b: b["y"])
    best_start, best_len = 0, 1
    cur_start = 0
    for i in range(1, len(ordered)):
        consistent = ordered[i]["span"] >= ordered[i - 1]["span"] - tol
        contiguous = ordered[i]["band"] == ordered[i - 1]["band"] + 1
        if not (consistent and contiguous):
            cur_start = i
        if i - cur_start + 1 > best_len:
            best_start, best_len = cur_start, i - cur_start + 1
    return ordered[best_start : best_start + best_len]


# --------------------------------------------------------------------------- #
# lane and ROI geometry
# --------------------------------------------------------------------------- #


def fit_lane_polys(run: list[dict], num_lanes: int, degree: int = LANE_POLY_DEGREE) -> list[np.ndarray]:
    if len(run) < degree + 1:
        raise RuntimeError(f"Only {len(run)} usable bands, need {degree + 1} to fit degree {degree}")
    ys = np.array([b["y"] for b in run])
    return [
        np.polyfit(ys, np.array([b["centers"][lane_id] for b in run]), deg=degree)
        for lane_id in range(num_lanes)
    ]


def lane_fit_residuals(run: list[dict], polys: list[np.ndarray]) -> list[float]:
    ys = np.array([b["y"] for b in run])
    return [
        float(np.abs(np.array([b["centers"][i] for b in run]) - np.polyval(polys[i], ys)).max())
        for i in range(len(polys))
    ]


def derive_roi(df: pd.DataFrame, polys: list[np.ndarray], y_min: float, y_max: float,
               rect: tuple[int, int, int, int], enclose_percentile: float = 99.0):
    """The outer lane centerlines run down the middle of the outer lanes, not along
    the road edge, so shift them outward by enough to enclose the detections that
    fall outside them (a high percentile rather than the max, so a few strays do
    not balloon the ROI), then clip to the patch box. Only the calibrated y-range
    is covered."""
    x0, _, w, _ = rect
    inside = df[(df["y_center"] >= y_min) & (df["y_center"] <= y_max)]
    ys, xs = inside["y_center"].to_numpy(), inside["x_center"].to_numpy()

    left_over = np.polyval(polys[0], ys) - xs
    right_over = xs - np.polyval(polys[-1], ys)
    left_over = left_over[left_over > 0]
    right_over = right_over[right_over > 0]
    shift_left = float(np.percentile(left_over, enclose_percentile)) if len(left_over) else 0.0
    shift_right = float(np.percentile(right_over, enclose_percentile)) if len(right_over) else 0.0

    y_samples = np.linspace(y_min, y_max, 60)
    left_curve = np.clip(np.polyval(polys[0], y_samples) - shift_left, x0, x0 + w)
    right_curve = np.clip(np.polyval(polys[-1], y_samples) + shift_right, x0, x0 + w)
    return y_samples, left_curve, right_curve, shift_left, shift_right


def fit_scale_model(polys: list[np.ndarray], num_lanes: int, y_min: float, y_max: float) -> dict:
    """Fit lateral metres-per-pixel against image row, from the spacing between
    lane centerlines against the standard lane width.

    Model: mpp(y) = h / (y - y_horizon). This is the form a pinhole camera viewing
    a flat ground plane produces, and it fits ~2.6x better than a straight line,
    which matters because a linear fit is off by ~25% at the near edge of the
    region and would bias every speed computed there.

    Two consequences are recorded for downstream use:
      - `h` and `y_horizon` are fit parameters of the projection. `h` has units of
        metres and is the right order for a mast-mounted camera, but it absorbs
        model error (the road curves, the plane is not perfectly flat), so it
        should not be read as a surveyed camera height.
      - Along-road position is D(y) = f * mpp(y) for the (unknown) focal length f.
        So along-road *distances* are proportional to differences in mpp, with f as
        a single global multiplier - see documentation/scene_calibration.md. Note
        this means any constant factor in mpp is absorbed by f: for speed and
        density it is only the SHAPE of mpp(y) that matters, not its absolute
        scale.
    """
    ys = np.arange(y_min, y_max + 1e-9, BAND_HEIGHT_PX)
    spacing_px = np.abs(np.polyval(polys[-1], ys) - np.polyval(polys[0], ys)) / (num_lanes - 1)
    mpp = LANE_WIDTH_M / spacing_px

    (h, y_horizon), _ = curve_fit(
        lambda y, h, yh: h / (y - yh), ys, mpp, p0=[10.0, 20.0], maxfev=20000
    )
    if y_horizon >= y_min:
        raise RuntimeError(
            f"Fitted horizon y={y_horizon:.1f} is inside the calibrated region "
            f"(starts at {y_min}); the scale model would diverge or change sign"
        )
    rms = float(np.sqrt((((h / (ys - y_horizon)) - mpp) ** 2).mean()))
    return {"h": float(h), "y_horizon": float(y_horizon), "rms": rms, "ys": ys, "mpp": mpp}


def metres_per_pixel(scale: dict, y):
    """Lateral scale at image row y, from the fitted ground-plane model."""
    return scale["h"] / (np.asarray(y, dtype=float) - scale["y_horizon"])


def implied_car_box_width_m(df: pd.DataFrame, scale: dict, y_min: float, y_max: float) -> float:
    """What the median detected car box measures in metres under the fitted scale.

    A consistency check, not an input. A real car is ~1.8 m wide, so a value
    meaningfully above that is expected here and quantifies how much the oblique
    view inflates boxes: the camera sees each car's rear plus part of its side.
    """
    cars = df[(df["cls"] == "car") & (df["y_center"] >= y_min) & (df["y_center"] <= y_max)]
    widths = []
    for band in sorted(cars["_band"].unique()):
        sub = cars[cars["_band"] == band]
        if len(sub) < MIN_CAR_DETECTIONS_FOR_SCALE_BAND:
            continue
        y = band_center_y(band)
        widths.append(float(np.median(sub["width_px"])) * float(metres_per_pixel(scale, y)))
    return float(np.mean(widths)) if widths else float("nan")


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #


def visualize(df_used, df_dropped, rect, all_bands, run, polys,
              roi_y, roi_left, roi_right, scale, sample_frame_path: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_poly = rect_to_polygon(rect)
    patch_arr = np.array(patch_poly + patch_poly[:1])
    roi_arr = np.array(
        list(zip(roi_left, roi_y)) + list(zip(roi_right[::-1], roi_y[::-1]))
        + [(roi_left[0], roi_y[0])]
    )
    y_fit = np.linspace(roi_y.min(), roi_y.max(), 200)
    lane_curves = [np.polyval(p, y_fit) for p in polys]
    kept_bands = {b["band"] for b in run}

    def draw_frame(ax):
        ax.invert_yaxis()
        ax.set_xlim(0, FRAME_W)
        ax.set_ylim(FRAME_H, 0)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    # 1. detections + final ROI
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if len(df_dropped):
        ax.scatter(df_dropped["x_center"], df_dropped["y_center"], s=4, alpha=0.35,
                   color="gray", label="dropped (off-road)")
    ax.scatter(df_used["x_center"], df_used["y_center"], s=4, alpha=0.2, label="detections")
    ax.plot(patch_arr[:, 0], patch_arr[:, 1], color="red", lw=1.5, label="patch ROI box")
    ax.plot(roi_arr[:, 0], roi_arr[:, 1], color="black", lw=2, label="calibrated ROI")
    draw_frame(ax)
    ax.set_title("Detections and the calibrated ROI")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roi_lanes_scatter.png", dpi=150)
    plt.close(fig)

    # 2. band centers (kept vs excluded) + fitted lanes
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for b in all_bands:
        kept = b["band"] in kept_bands
        for lane_id, cx in enumerate(b["centers"]):
            ax.scatter(cx, b["y"], s=26 if kept else 22,
                       color=LANE_COLORS_RGB[lane_id % len(LANE_COLORS_RGB)] if kept else "lightgray",
                       marker="o" if kept else "x", zorder=3 if kept else 2)
    for lane_id, curve in enumerate(lane_curves):
        ax.plot(curve, y_fit, color=LANE_COLORS_RGB[lane_id % len(LANE_COLORS_RGB)], lw=2,
                label=f"lane {lane_id}")
    ax.plot(patch_arr[:, 0], patch_arr[:, 1], color="red", lw=1, label="patch ROI box")
    ax.plot(roi_arr[:, 0], roi_arr[:, 1], color="black", lw=1.5, ls="--", label="calibrated ROI")
    draw_frame(ax)
    ax.set_title("Band lane centers (x = excluded) and fitted lanes")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lane_clusters.png", dpi=150)
    plt.close(fig)

    # 3. why the region ends where it does
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ys_all = [b["y"] for b in all_bands]
    spans = [b["span"] for b in all_bands]
    ax.plot(spans, ys_all, color="lightgray", lw=1, zorder=1)
    ax.scatter([b["span"] for b in all_bands if b["band"] not in kept_bands],
               [b["y"] for b in all_bands if b["band"] not in kept_bands],
               color="crimson", marker="x", s=45, label="excluded band", zorder=3)
    ax.scatter([b["span"] for b in run], [b["y"] for b in run],
               color="tab:blue", s=30, label="calibrated band", zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel("lane span: rightmost - leftmost center (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Span must shrink with distance;\nbands that break this are excluded", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "band_span_diagnostic.png", dpi=150)
    plt.close(fig)

    # 4. overlay on a real frame
    overlay = cv2.imread(str(sample_frame_path)).copy()
    cv2.polylines(overlay, [patch_arr.astype(int)], True, (0, 0, 255), 1)
    cv2.polylines(overlay, [roi_arr.astype(int)], True, (0, 0, 0), 1)
    for b in run:
        for lane_id, cx in enumerate(b["centers"]):
            cv2.circle(overlay, (int(cx), int(b["y"])), 2,
                       LANE_COLORS_BGR[lane_id % len(LANE_COLORS_BGR)], -1)
    for lane_id, curve in enumerate(lane_curves):
        pts = np.stack([curve, y_fit], axis=1).astype(int)
        cv2.polylines(overlay, [pts], False, LANE_COLORS_BGR[lane_id % len(LANE_COLORS_BGR)], 1)
    cv2.imwrite(str(OUT_DIR / "roi_lanes_on_frame.png"),
                cv2.resize(overlay, (FRAME_W * 3, FRAME_H * 3), interpolation=cv2.INTER_NEAREST))

    # 5. scale model: measured band values against the fitted ground-plane curve,
    #    with the straight-line fit shown for comparison
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ys = np.linspace(roi_y.min(), roi_y.max(), 100)
    ax.scatter(scale["mpp"], scale["ys"], s=28, color="tab:blue", zorder=3, label="measured (lane spacing)")
    ax.plot(metres_per_pixel(scale, ys), ys, color="tab:blue", lw=2,
            label=f"h/(y-y0) fit, RMS={scale['rms']:.4f}")
    lin = np.polyfit(scale["ys"], scale["mpp"], 1)
    lin_rms = np.sqrt(((np.polyval(lin, scale["ys"]) - scale["mpp"]) ** 2).mean())
    ax.plot(np.polyval(lin, ys), ys, color="crimson", lw=1.2, ls="--",
            label=f"linear fit (rejected), RMS={lin_rms:.4f}")
    ax.invert_yaxis()
    ax.set_xlabel("lateral metres per pixel")
    ax.set_ylabel("y (px)")
    ax.set_title(f"Scale over calibrated region (lane width = {LANE_WIDTH_M} m)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scale_curve.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #


def main(use_cache: bool = True):
    info = load_info_table()
    clips = sample_clips(info, CLIP_SAMPLE, SEED)
    info_by_name = info.set_index("filename")

    cache_path = OUT_DIR / "detections.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path)
        # the cache records no sampling parameters, so verify it came from this
        # exact clip sample - otherwise a changed SEED/CLIP_SAMPLE would be
        # silently calibrated against the wrong detections
        if set(df["clip"].unique()) != set(clips):
            raise RuntimeError(
                f"{cache_path} was built from a different clip sample "
                f"(SEED/CLIP_SAMPLE changed?). Re-run with --no-cache."
            )
        print(f"Loaded {len(df)} cached detections from {cache_path}")
    else:
        print(f"Sampled {len(clips)} clips (seed={SEED}, per label: {CLIP_SAMPLE})")
        model = YOLO(MODEL_NAME)
        rows = []
        for name in clips:
            dets = detect_in_clip(model, name, int(info_by_name.loc[name, "start_frame"]))
            rows.extend(dets)
            print(f"  {name}: {len(dets)} vehicle detections")
        df = pd.DataFrame(rows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Total pooled detections: {len(df)}")

    # the cache is detected at DETECT_CONF; calibration works from the stricter subset
    n_all = len(df)
    df = df[df["conf"] >= CONF_THRESH]
    print(f"Using {len(df)}/{n_all} detections at conf >= {CONF_THRESH} for calibration")

    rect = load_patch_roi_rect()
    df_in = filter_to_rect(df, rect).copy()
    df_in["_band"] = df_in["y_center"].apply(band_key)
    df_clean = filter_main_cluster(df_in)
    print(f"{len(df_in)}/{len(df)} detections inside patch ROI {rect}; "
          f"{len(df_in) - len(df_clean)} dropped as off-road")

    congestion = info_by_name["congestion"]
    light_clips = [c for c in clips if congestion.get(c) == "light"][:LANE_SAMPLE_LIGHT_CLIPS]
    df_light = df_clean[df_clean["clip"].isin(light_clips)]
    print(f"Reading lane geometry from {len(df_light)} detections in {len(light_clips)} light clips")

    all_bands = measure_bands(df_light, NUM_LANES)
    run = longest_perspective_consistent_run(all_bands)
    if len(run) < LANE_POLY_DEGREE + 1:
        raise RuntimeError(
            f"Only {len(run)} perspective-consistent bands out of {len(all_bands)} measured; "
            "cannot fit lanes"
        )
    y_min, y_max = run[0]["y"], run[-1]["y"]
    kept = {b["band"] for b in run}
    excluded = [b["y"] for b in all_bands if b["band"] not in kept]
    print(f"Lanes distinguishable over y=[{y_min:.0f}, {y_max:.0f}] "
          f"({len(run)}/{len(all_bands)} bands); excluded y={[f'{y:.0f}' for y in excluded]}")

    polys = fit_lane_polys(run, NUM_LANES)
    residuals = lane_fit_residuals(run, polys)
    print("Max lane fit residual (px): " + ", ".join(f"lane{i}={r:.1f}" for i, r in enumerate(residuals)))

    roi_y, roi_left, roi_right, shift_left, shift_right = derive_roi(df_clean, polys, y_min, y_max, rect)
    print(f"ROI from outer lanes, shifted left {shift_left:.1f}px / right {shift_right:.1f}px")

    scale = fit_scale_model(polys, NUM_LANES, y_min, y_max)
    car_box_m = implied_car_box_width_m(df_clean, scale, y_min, y_max)
    print(f"Scale model m/px = h/(y - y_horizon): h={scale['h']:.2f}, "
          f"y_horizon={scale['y_horizon']:.1f}, RMS residual={scale['rms']:.5f}")
    print(f"Consistency check: median car box measures {car_box_m:.2f} m "
          f"(a real car is ~1.8 m; the excess is oblique-view inflation)")

    sample_clip = clips[0]
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{sample_clip}.avi"))
    for _ in range(int(info_by_name.loc[sample_clip, "start_frame"]) + 1):
        ok, frame = cap.read()
    cap.release()
    sample_frame_path = OUT_DIR / "_sample_frame.png"
    cv2.imwrite(str(sample_frame_path), frame)

    visualize(df_clean, df_in.loc[df_in.index.difference(df_clean.index)], rect,
              all_bands, run, polys, roi_y, roi_left, roi_right, scale, sample_frame_path)

    roi_polygon = ([[float(x), float(y)] for x, y in zip(roi_left, roi_y)]
                   + [[float(x), float(y)] for x, y in zip(roi_right[::-1], roi_y[::-1])])

    config = {
        "_comment": (
            "Scene calibration for WSDOT camera cctv052 (I-5 / S 188th St, Seattle). "
            "The calibrated region is the y-range over which the 5 lanes are actually "
            "separable in pooled YOLOv8s detections: per-band agglomerative clustering, "
            "keeping the longest run of bands whose lane span shrinks with distance as "
            "perspective requires. Bands that break that are excluded: near the top an "
            "incoming merge lane joins from the right, and near the bottom the carriageway "
            "runs off the right-hand side of the frame. Lanes are NOT "
            "extrapolated beyond this range - outside it the lanes are not resolvable and "
            "any assignment would be invented. See outputs/calibration/ for diagnostics."
        ),
        "frame_width": FRAME_W,
        "frame_height": FRAME_H,
        "calibrated_y_range": [y_min, y_max],
        "roi_polygon": roi_polygon,
        "roi_polygon_note": "outer lane curves shifted outward to enclose detections, clipped to the patch box; spans calibrated_y_range only",
        "roi_edge_shift_px": {"left": shift_left, "right": shift_right},
        "patch_roi_box": rect_to_polygon(rect),
        "patch_roi_box_note": "recovered from traffic_patches.mat via scripts/recover_patch_roi.py; bounds detections before clustering",
        "num_lanes": NUM_LANES,
        "lane_polynomials": [p.tolist() for p in polys],
        "lane_polynomial_degree": LANE_POLY_DEGREE,
        "lane_curve_note": "x = polyval(lane_polynomials[i], y), valid only for y within calibrated_y_range",
        "lane_max_fit_residual_px": residuals,
        "scale_model": {
            "h": scale["h"],
            "y_horizon": scale["y_horizon"],
            "rms_residual_m_per_px": scale["rms"],
        },
        "scale_note": (
            f"LATERAL metres_per_pixel = h / (y - y_horizon), valid over calibrated_y_range; "
            f"fit from the spacing between lane centerlines against a standard "
            f"{LANE_WIDTH_M} m lane width"
        ),
        "implied_car_box_width_m": car_box_m,
        "implied_car_box_width_note": (
            "Consistency check, not an input: what the median detected car box measures "
            "under this scale. Exceeds a real car's ~1.8 m because the oblique view puts "
            "part of each car's side inside the box."
        ),
        "along_road_note": (
            "Along-road position D(y) is proportional to metres_per_pixel(y): D = f * mpp(y) "
            "for the camera's (unknown) focal length f in pixels. So along-road DISTANCES are "
            "proportional to differences in mpp, with f as a single global multiplier. Ratios "
            "of speeds are therefore independent of f; absolute speeds require anchoring f."
        ),
        "calibration_meta": {
            "model": MODEL_NAME,
            "imgsz": IMGSZ,
            "conf_threshold": CONF_THRESH,
            "seed": SEED,
            "clips_sampled_per_label": CLIP_SAMPLE,
            "lane_sample_clips": light_clips,
            "band_height_px": BAND_HEIGHT_PX,
            "min_detections_per_band": MIN_DETECTIONS_PER_BAND,
            "span_tolerance_px": SPAN_TOLERANCE_PX,
            "outlier_gap_px": OUTLIER_GAP_PX,
            "bands_measured": len(all_bands),
            "bands_calibrated": len(run),
            "bands_excluded_y": excluded,
            "num_detections_raw": len(df),
            "num_detections_in_roi": len(df_clean),
            "detections_per_class": df_clean["cls"].value_counts().to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"Wrote {CONFIG_PATH}\nDiagnostics in {OUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-run detection instead of reusing outputs/calibration/detections.csv")
    main(use_cache=not parser.parse_args().no_cache)
