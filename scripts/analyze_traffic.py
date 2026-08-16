"""Core traffic-flow analysis: detect, track, and compute traffic parameters.

Runs YOLOv8 with ByteTrack over the dataset's clips, applies the scene
calibration (scripts/scene.py), and writes three tables at increasing levels of
aggregation:

  outputs/analysis/detections.csv - one row per tracked detection per frame
  outputs/analysis/tracks.csv     - one row per vehicle track
  outputs/analysis/clips.csv      - one row per clip, joined to its ground-truth label

Speeds are reported in RAW units (metres / focal_px per second). They are exact up
to one unknown global multiplier, so any ratio of speeds is exact, while absolute
km/h requires the anchor described in scripts/scene.py. Congestion analysis
downstream uses ratios, so it never depends on that anchor.

Usage:
    uv run python scripts/analyze_traffic.py              # all clips
    uv run python scripts/analyze_traffic.py --limit 6    # quick smoke test
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene import Scene  # noqa: E402

import calibrate_scene as cal  # noqa: E402  (reuse dataset loading + constants)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "analysis"

FPS = 10.0  # dataset frame rate, per README_TRAFFICDB
IMGSZ = cal.IMGSZ  # same model and resolution as calibration
# Deliberately more permissive than calibration's threshold: here a missed vehicle is
# lost outright, while loose box edges wash out of the least-squares speed fit. The
# reverse trade applies when placing lane geometry - see calibrate_scene.CONF_THRESH.
CONF_THRESH = cal.DETECT_CONF
TRACKER = "bytetrack.yaml"
# A track qualifies for a speed estimate on DURATION alone, never on displacement.
#
# An earlier version also required ~6px of movement, to suppress fits on jittering
# boxes. That was a mistake: it discarded 13.8% of heavy-traffic tracks against 3.4%
# of light ones, because stopped vehicles are what heavy congestion IS. Filtering on
# movement therefore selectively removed the slowest vehicles and biased heavy-traffic
# speeds upward - precisely where near-zero speed is the most diagnostic signal.
#
# Duration is the right proxy for whether a fit is trustworthy: over enough frames a
# stationary vehicle has a well-determined speed of about zero, which is real data.
# Measured (held-out accuracy / heavy speed coverage): 4 frames + 6px -> 93.3% / 71%;
# 6 frames, no movement gate -> 93.7% / 78%.
MIN_TRACK_SAMPLES = 6


def track_clip(model: YOLO, scene: Scene, filename: str, start_frame: int) -> pd.DataFrame:
    """Run detection + tracking over one clip, keeping detections inside the
    calibrated road region and tagging each with its lane."""
    path = cal.VIDEO_DIR / f"{filename}.avi"
    rows = []
    results = model.track(
        source=str(path), stream=True, persist=True, tracker=TRACKER,
        imgsz=IMGSZ, conf=CONF_THRESH, verbose=False, device=0,
    )
    for frame_idx, res in enumerate(results):
        if frame_idx < start_frame:  # frame 0 is corrupted in every clip
            continue
        if res.boxes is None or res.boxes.id is None:
            continue
        for box, tid, cls_id, conf in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.id.cpu().numpy().astype(int),
            res.boxes.cls.cpu().numpy().astype(int),
            res.boxes.conf.cpu().numpy(),
        ):
            if cls_id not in cal.VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = box
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            if not bool(scene.in_roi(xc, yc)):
                continue
            rows.append(
                {
                    "clip": filename,
                    "frame_idx": frame_idx,
                    "t_s": (frame_idx - start_frame) / FPS,
                    "track_id": int(tid),
                    "cls": cal.VEHICLE_CLASS_IDS[cls_id],
                    "conf": float(conf),
                    "x_center": float(xc),
                    "y_center": float(yc),
                    "width_px": float(x2 - x1),
                    "height_px": float(y2 - y1),
                    "lane": int(scene.lane_of(xc, yc)[0]),
                    "along_raw": float(scene.along_road_raw(yc)),
                }
            )
    return pd.DataFrame(rows)


def summarise_tracks(det: pd.DataFrame) -> pd.DataFrame:
    """One row per track: class, lane usage, lane changes, and raw speed.

    Speed comes from a least-squares fit of along-road position against time
    rather than differencing endpoints, so a single noisy detection cannot
    dominate the estimate.
    """
    out = []
    for (clip, tid), g in det.groupby(["clip", "track_id"]):
        g = g.sort_values("frame_idx")
        lanes = [int(v) for v in g["lane"] if v >= 0]
        # a vehicle's lane is its most common assignment; changes are counted on
        # the de-duplicated sequence so per-frame jitter is not a lane change
        squashed = [k for k, _ in itertools.groupby(lanes)]
        n, span = len(g), float(g["y_center"].max() - g["y_center"].min())
        speed_raw = np.nan
        if n >= MIN_TRACK_SAMPLES:
            speed_raw = abs(float(np.polyfit(g["t_s"].to_numpy(), g["along_raw"].to_numpy(), 1)[0]))
        out.append(
            {
                "clip": clip,
                "track_id": tid,
                "cls": Counter(g["cls"]).most_common(1)[0][0],
                "n_frames": n,
                "duration_s": float(g["t_s"].max() - g["t_s"].min()),
                "y_span_px": span,
                "lane": Counter(lanes).most_common(1)[0][0] if lanes else -1,
                "lane_changes": max(0, len(squashed) - 1),
                "speed_raw": speed_raw,
                "mean_conf": float(g["conf"].mean()),
            }
        )
    return pd.DataFrame(out)


def summarise_clips(det: pd.DataFrame, tracks: pd.DataFrame, info: pd.DataFrame,
                    scene: Scene) -> pd.DataFrame:
    """One row per clip: counts, class mix, speed, lane occupancy, density."""
    info_by_name = info.set_index("filename")
    per_frame = det.groupby(["clip", "frame_idx"]).size()
    out = []
    for clip, g in tracks.groupby("clip"):
        dets = det[det["clip"] == clip]
        speeds = g["speed_raw"].dropna()
        counts = per_frame.loc[clip] if clip in per_frame.index.get_level_values(0) else pd.Series(dtype=int)
        lane_counts = dets["lane"].value_counts()
        meta = info_by_name.loc[clip]
        row = {
            "clip": clip,
            "congestion_true": meta["congestion"],
            "weather": meta["weather"],
            "date": meta["date"],
            "hour": int(str(meta["timestamp"]).split(".")[0]),
            "n_tracks": len(g),
            "n_tracks_with_speed": int(speeds.notna().sum()),
            # mean vehicles visible in the region at any instant - the density proxy,
            # independent of how long each vehicle stays in frame
            "mean_vehicles_in_roi": float(counts.mean()) if len(counts) else 0.0,
            "max_vehicles_in_roi": int(counts.max()) if len(counts) else 0,
            "median_speed_raw": float(speeds.median()) if len(speeds) else np.nan,
            "mean_speed_raw": float(speeds.mean()) if len(speeds) else np.nan,
            "p15_speed_raw": float(speeds.quantile(0.15)) if len(speeds) else np.nan,
            "total_lane_changes": int(g["lane_changes"].sum()),
        }
        for cls in ("car", "truck", "bus", "motorcycle"):
            row[f"n_{cls}"] = int((g["cls"] == cls).sum())
        for lane in range(scene.num_lanes):
            # share of in-region detections falling in this lane
            row[f"lane{lane}_share"] = float(lane_counts.get(lane, 0) / max(len(dets), 1))
        out.append(row)
    return pd.DataFrame(out).sort_values("clip").reset_index(drop=True)


def main(limit: int | None = None):
    scene = Scene.from_config()
    info = cal.load_info_table()
    clips = info["filename"].tolist()
    if limit:
        # spread the smoke-test sample across congestion labels
        clips = (
            info.sort_values("congestion").groupby("congestion", group_keys=False)
            .head(max(1, limit // 3))["filename"].tolist()
        )
    print(f"Analysing {len(clips)} clips over calibrated rows "
          f"y=[{scene.y_min:.0f}, {scene.y_max:.0f}]")

    model = YOLO(cal.MODEL_NAME)
    info_by_name = info.set_index("filename")
    frames = []
    for i, name in enumerate(clips, 1):
        df = track_clip(model, scene, name, int(info_by_name.loc[name, "start_frame"]))
        frames.append(df)
        if i % 25 == 0 or i == len(clips):
            print(f"  {i}/{len(clips)} clips, {sum(len(f) for f in frames)} in-region detections")

    det = pd.concat(frames, ignore_index=True)
    if det.empty:
        raise RuntimeError("No detections inside the calibrated region")
    tracks = summarise_tracks(det)
    clip_summary = summarise_clips(det, tracks, info, scene)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    det.to_csv(OUT_DIR / "detections.csv", index=False)
    tracks.to_csv(OUT_DIR / "tracks.csv", index=False)
    clip_summary.to_csv(OUT_DIR / "clips.csv", index=False)

    with_speed = tracks["speed_raw"].notna().sum()
    print(f"\n{len(det)} detections, {len(tracks)} tracks "
          f"({with_speed} with a speed estimate, {with_speed / len(tracks):.0%})")
    print("\nMedian raw speed by ground-truth congestion label:")
    print(clip_summary.groupby("congestion_true")["median_speed_raw"].describe()[["count", "mean", "50%"]])
    print("\nMean vehicles in region by label:")
    print(clip_summary.groupby("congestion_true")["mean_vehicles_in_roi"].mean())
    print(f"\nWrote 3 tables to {OUT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Analyse only a small spread of clips")
    main(limit=p.parse_args().limit)
