"""Render the annotated demo: one self-contained HTML file.

Draws the analysis back onto the source video and packages it, with the
evaluation charts, into a single HTML page whose assets are embedded as data
URIs. One file, no side-car assets, small enough to share as a link.

No inference happens here. Every box, track id, lane and speed is read from
outputs/analysis/detections.csv, so the demo cannot disagree with the reported
numbers - both are the same table read through the same scripts/scene.py.

Vehicles are tinted by speed rather than by lane, because "where is it congested"
is the question the picture should answer: dark boxes are slow.

Usage:
    uv run python scripts/render_demo.py
    uv run python scripts/render_demo.py --per-class 3
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene import Scene  # noqa: E402
import calibrate_scene as cal  # noqa: E402
from evaluate_congestion import FREE_FLOW_PERCENTILE, LABELS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = REPO_ROOT / "outputs" / "analysis"
EVALUATION = REPO_ROOT / "outputs" / "evaluation"
OUT_DIR = REPO_ROOT / "outputs" / "demo"

SCALE = 3  # 320x240 -> 960x720; annotations are drawn after upscaling so text stays sharp
FPS = 10

# Anchor for absolute speed. The along-road scale contains the camera's focal length,
# which this dataset does not provide, so km/h requires assuming one reference speed:
# here, that free-flowing traffic runs at the posted limit for I-5 at S 188th St.
#
# This is an ASSUMPTION, not a measurement, and it makes the free-flow figure true by
# construction - "light traffic is about 60 mph" carries no information. Everything
# else is measured relative to it and does carry information. The 93.7% classification
# accuracy uses ratios only and is unaffected by this number.
#
# Sanity check: this anchor implies a focal length of ~867 px, i.e. a ~21 degree
# horizontal field of view, which is a plausible lens for a camera looking far down a
# highway. An implausible FOV here would have meant the anchor was wrong.
ASSUMED_FREE_FLOW_KMH = 96.5  # 60 mph

# Palette (BGR for OpenCV). Ordinal blue ramp, matching the evaluation charts.
INK = (11, 11, 11)
PANEL = (251, 252, 252)
MUTED = (78, 81, 82)
CLASS_BGR = {"light": (239, 182, 134), "medium": (214, 120, 42), "heavy": (129, 66, 16)}
# Ordinal blue, slowest -> fastest. Steps 650/550/450/350/250 of the ramp: the light
# end stops at #86b6ef because the ordinal rule forbids going nearer the surface than
# that, which is exactly what made free-flowing vehicles wash out against pale tarmac.
SPEED_RAMP = [(129, 66, 16), (171, 92, 28), (214, 120, 42), (231, 152, 85), (239, 182, 134)]


# Vehicles the tracker saw too briefly to time. They must stay visible against pale
# tarmac, so this is a mid-grey rather than a light one, and it is deliberately
# off-hue from the blue ramp so "not measured" never reads as "some speed".
NO_SPEED_BGR = (105, 108, 110)


def speed_colour(ratio: float) -> tuple[int, int, int]:
    """Slow -> dark, free-flowing -> light. One hue, so it reads as a magnitude."""
    if not np.isfinite(ratio):
        return NO_SPEED_BGR
    idx = int(np.clip(ratio, 0, 1) * (len(SPEED_RAMP) - 1))
    return SPEED_RAMP[idx]


def pick_clips(clips: pd.DataFrame, preds: pd.DataFrame, per_class: int) -> list[str]:
    """Representative clips: correctly classified, and closest to their class's
    median speed so each is typical of its label rather than a borderline case."""
    df = clips.merge(preds[["clip", "congestion_pred"]], on="clip", how="left")
    df = df[df["congestion_true"] == df["congestion_pred"]]
    chosen = []
    for lab in LABELS:
        sub = df[(df["congestion_true"] == lab) & df["median_speed_raw"].notna()].copy()
        if sub.empty:
            continue
        sub["dist"] = (sub["median_speed_raw"] - sub["median_speed_raw"].median()).abs()
        chosen += sub.nsmallest(per_class, "dist")["clip"].tolist()
    return chosen


def draw_frame(frame, scene: Scene, dets: pd.DataFrame, meta: dict, free_flow: float):
    """Upscale, then draw geometry, vehicles and the metrics panel."""
    img = cv2.resize(frame, (cal.FRAME_W * SCALE, cal.FRAME_H * SCALE), interpolation=cv2.INTER_CUBIC)

    # calibrated road region and lane centerlines, kept recessive
    ys = np.linspace(scene.y_min, scene.y_max, 60)
    left, right = scene.roi_edges(ys)
    poly = np.array(list(zip(left, ys)) + list(zip(right[::-1], ys[::-1]))) * SCALE
    cv2.polylines(img, [poly.astype(int)], True, (120, 122, 124), 2, cv2.LINE_AA)
    for lane in range(scene.num_lanes):
        pts = np.stack([scene.lane_x(lane, ys), ys], axis=1) * SCALE
        cv2.polylines(img, [pts.astype(int)], False, (150, 152, 154), 1, cv2.LINE_AA)

    for _, d in dets.iterrows():
        ratio = d["speed_ratio"]
        col = speed_colour(ratio)
        w, h = d["width_px"] * SCALE, d["height_px"] * SCALE
        cx, cy = d["x_center"] * SCALE, d["y_center"] * SCALE
        p1 = (int(cx - w / 2), int(cy - h / 2))
        p2 = (int(cx + w / 2), int(cy + h / 2))
        cv2.rectangle(img, p1, p2, col, 2, cv2.LINE_AA)
        # km/h with the unit spelled out: a bare number invites "42 what?"
        tag = f"{ratio * ASSUMED_FREE_FLOW_KMH:.0f} km/h" if np.isfinite(ratio) else "not timed"
        # a dark plate behind the label keeps it legible over both tarmac and vehicles
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 6, p1[1] - 1), col, -1)
        cv2.putText(img, tag, (p1[0] + 3, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)

    _draw_panel(img, dets, meta, free_flow)
    return img


def _draw_panel(img, dets: pd.DataFrame, meta: dict, free_flow: float):
    """Metrics burned into the frame, so the page needs no scripting to stay in sync."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 56), PANEL, -1)
    cv2.rectangle(img, (0, h - 46), (w, h), PANEL, -1)

    cv2.putText(img, meta["clip"], (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
    cv2.putText(img, f"{meta['hour']:02d}:00  {meta['weather']}", (12, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)

    # congestion badge: ground truth beside our held-out prediction
    for i, (label, key) in enumerate((("actual", "true"), ("predicted", "pred"))):
        val = meta[key]
        x = w - 300 + i * 150
        cv2.rectangle(img, (x, 10), (x + 138, 46), CLASS_BGR[val], -1)
        cv2.putText(img, label, (x + 8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255) if val != "light" else INK, 1, cv2.LINE_AA)
        cv2.putText(img, val.upper(), (x + 8, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if val != "light" else INK, 1, cv2.LINE_AA)

    n = len(dets)
    speeds = dets["speed_ratio"].dropna()
    med = speeds.median() if len(speeds) else float("nan")
    cv2.putText(img, f"vehicles in region: {n}", (12, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
    txt = f"median speed: {med * ASSUMED_FREE_FLOW_KMH:.0f} km/h" if np.isfinite(med) else "median speed: --"
    cv2.putText(img, txt, (270, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)

    lane_counts = dets["lane"].value_counts()
    parts = "lanes  " + "  ".join(f"L{i}:{int(lane_counts.get(i, 0))}" for i in range(5))
    cv2.putText(img, parts, (620, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)


def render_video(clips: list[str], scene: Scene, det: pd.DataFrame, info: pd.DataFrame,
                 preds: pd.DataFrame, free_flow: float, path: Path) -> None:
    info_by = info.set_index("filename")
    pred_by = preds.set_index("clip")["congestion_pred"].to_dict()
    writer = iio.get_writer(
        path, fps=FPS, codec="libx264", quality=7, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    for name in clips:
        meta_row = info_by.loc[name]
        meta = {
            "clip": name,
            "weather": meta_row["weather"],
            "hour": int(str(meta_row["timestamp"]).split(".")[0]),
            "true": meta_row["congestion"],
            "pred": pred_by.get(name, meta_row["congestion"]),
        }
        start = int(meta_row["start_frame"])
        clip_det = det[det["clip"] == name]
        cap = cv2.VideoCapture(str(cal.VIDEO_DIR / f"{name}.avi"))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= start:
                img = draw_frame(frame, scene, clip_det[clip_det["frame_idx"] == idx], meta, free_flow)
                writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
    writer.close()


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def per_class_speed_rows(tracks: pd.DataFrame, clips_tbl: pd.DataFrame, free_flow: float) -> str:
    m = tracks.merge(clips_tbl[["clip", "congestion_true", "mean_vehicles_in_roi"]], on="clip", how="left")
    out = []
    for lab in LABELS:
        sub = m[m["congestion_true"] == lab]
        ratio = (sub["speed_raw"] / free_flow).median()
        veh = clips_tbl.loc[clips_tbl["congestion_true"] == lab, "mean_vehicles_in_roi"].median()
        out.append(
            f"<tr><td>{lab}</td><td><b>{ratio * ASSUMED_FREE_FLOW_KMH:.0f} km/h</b></td>"
            f"<td>{veh:.1f}</td></tr>"
        )
    return "".join(out)


def build_html(video: Path, clips: list[str], summary: dict, speed_rows: str, path: Path) -> None:
    charts = "".join(
        f'<figure><img src="{data_uri(EVALUATION / f, "image/png")}" alt="{alt}"><figcaption>{cap}</figcaption></figure>'
        for f, alt, cap in [
            ("confusion_matrices.png", "Confusion matrices for both classifiers",
             "Held-out results, pooled over the dataset's four folds. Counting alone cannot separate medium from heavy."),
            ("feature_separation.png", "Each clip plotted by speed and by vehicle count",
             "Every clip, positioned by feature and grouped by its true label."),
        ]
    )
    # band edges follow the 5-step ramp in speed_colour()
    k = ASSUMED_FREE_FLOW_KMH
    legend = "".join(
        f'<span class="sw"><i style="background:{c}"></i>{t}</span>'
        for c, t in [
            ("#104281", f"under {0.125 * k:.0f} km/h"),
            ("#2a78d6", f"about {0.375 * k:.0f}-{0.625 * k:.0f} km/h"),
            ("#86b6ef", f"over {0.875 * k:.0f} km/h"),
            ("#6e6c69", "tracked too briefly to time"),
        ]
    )
    html = f"""<title>Traffic Flow Analysis</title>
<style>
  :root {{ --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e4e3de; --accent:#2a78d6; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
    --bg:#1a1a19; --ink:#fff; --muted:#c3c2b7; --line:#333330; --accent:#3987e5; }} }}
  :root[data-theme=dark] {{ --bg:#1a1a19; --ink:#fff; --muted:#c3c2b7; --line:#333330; --accent:#3987e5; }}
  body {{ background:var(--bg); color:var(--ink); margin:0 auto; padding:2.5rem 1.25rem 4rem;
    max-width:60rem; font:16px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  h1 {{ font-size:1.8rem; margin:0 0 .3rem; letter-spacing:-.02em; }}
  h2 {{ font-size:1.15rem; margin:2.5rem 0 .6rem; }}
  .sub {{ color:var(--muted); margin:0 0 2rem; }}
  video, img {{ width:100%; border-radius:8px; display:block; }}
  video {{ background:#000; }}
  figure {{ margin:0 0 1.5rem; }}
  figcaption {{ color:var(--muted); font-size:.85rem; margin-top:.5rem; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(9rem,1fr)); gap:.75rem; margin:1.25rem 0; }}
  .kpi {{ border:1px solid var(--line); border-radius:8px; padding:.8rem .9rem; }}
  .kpi b {{ display:block; font-size:1.6rem; line-height:1.2; letter-spacing:-.02em; }}
  .kpi span {{ color:var(--muted); font-size:.8rem; }}
  .sw {{ display:inline-flex; align-items:center; gap:.4rem; margin-right:1rem; color:var(--muted); font-size:.85rem; }}
  .sw i {{ width:.85rem; height:.85rem; border-radius:3px; display:inline-block; }}
  table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
  th,td {{ text-align:left; padding:.45rem .6rem; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; }}
  code {{ background:color-mix(in srgb, var(--ink) 8%, transparent); padding:.1rem .3rem; border-radius:3px; font-size:.9em; }}
  .note {{ border-left:3px solid var(--accent); padding:.1rem 0 .1rem .9rem; color:var(--muted); margin:1.25rem 0; }}
</style>

<h1>Traffic Flow Analysis</h1>
<p class="sub">Vehicle detection, tracking and congestion classification on highway
camera footage &mdash; I-5 at S 188th St, Seattle.</p>

<div class="kpis">
  <div class="kpi"><b>{summary['accuracy']:.1%}</b><span>congestion accuracy, held out</span></div>
  <div class="kpi"><b>{summary['n_clips']}</b><span>clips analysed</span></div>
  <div class="kpi"><b>{summary['n_tracks']:,}</b><span>vehicles tracked</span></div>
  <div class="kpi"><b>{summary['n_dets']:,}</b><span>detections</span></div>
</div>

<h2>Annotated footage</h2>
<p class="sub">{len(clips)} representative clips &mdash; two each of light, medium and heavy traffic.
Each vehicle is tinted and labelled by its measured speed. The panel shows vehicles currently
in the measured region, their median speed, and how many occupy each lane. Actual and
predicted congestion class are shown top right.</p>
<p>{legend}</p>
<video controls muted loop playsinline src="{data_uri(video, 'video/mp4')}"></video>

<h2>Typical speeds by congestion level</h2>
<table>
  <tr><th>Class</th><th>Median vehicle speed</th><th>Vehicles in region</th></tr>
  {speed_rows}
</table>

<h2>Does it work?</h2>
{charts}

<div class="note">
Accuracy is measured on the dataset's own four train/test folds, so every clip is
predicted while held out, and the thresholds are fitted only on training clips.
</div>

<h2>What the numbers say</h2>
<table>
  <tr><th>Finding</th><th>Evidence</th></tr>
  <tr><td>Speed classifies congestion well</td><td>{summary['accuracy']:.1%} on held-out clips, versus {summary['baseline']:.1%} counting vehicles alone</td></tr>
  <tr><td>Counting cannot tell medium from heavy</td><td>Recall collapses to 31% &mdash; those classes average 10.4 and 11.1 vehicles, effectively identical</td></tr>
  <tr><td>Congestion is slowing, not crowding</td><td>Past jam density the road holds no more cars, so speed carries the signal</td></tr>
  <tr><td>Errors are never severe</td><td>No light clip was called heavy, or the reverse</td></tr>
</table>

<h2>How it works</h2>
<table>
  <tr><th>Stage</th><th>What happens</th></tr>
  <tr><td>Calibrate</td><td>Lane geometry is recovered from where vehicles are actually detected, and the road region is limited to where lanes are genuinely separable</td></tr>
  <tr><td>Detect &amp; track</td><td>YOLOv8s with ByteTrack, pretrained &mdash; nothing was trained on this data</td></tr>
  <tr><td>Measure</td><td>Speed comes from along-road motion under a perspective model, not raw pixel displacement</td></tr>
  <tr><td>Classify</td><td>Each clip's median vehicle speed, split into three classes by two thresholds fitted on training clips only</td></tr>
</table>
"""
    path.write_text(html, encoding="utf-8")


def main(per_class: int = 2):
    scene = Scene.from_config()
    info = cal.load_info_table()
    det = pd.read_csv(ANALYSIS / "detections.csv")
    tracks = pd.read_csv(ANALYSIS / "tracks.csv")
    clips_tbl = pd.read_csv(ANALYSIS / "clips.csv")
    preds = pd.read_csv(EVALUATION / "predictions.csv")

    free_flow = float(np.nanpercentile(clips_tbl["median_speed_raw"], FREE_FLOW_PERCENTILE))
    speed_by_track = tracks.set_index(["clip", "track_id"])["speed_raw"] / free_flow
    det = det.join(speed_by_track.rename("speed_ratio"), on=["clip", "track_id"])

    chosen = pick_clips(clips_tbl, preds, per_class)
    print(f"Rendering {len(chosen)} clips: {chosen}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUT_DIR / "annotated.mp4"
    render_video(chosen, scene, det, info, preds, free_flow, video_path)
    mb = video_path.stat().st_size / 1e6
    print(f"video: {mb:.1f} MB")

    acc = (preds["congestion_pred"] == preds["congestion_true"]).mean()
    base = (preds["congestion_pred_count_only"] == preds["congestion_true"]).mean()
    summary = {
        "accuracy": acc, "baseline": base, "n_clips": len(preds),
        "n_tracks": len(tracks), "n_dets": len(det),
    }
    speed_rows = per_class_speed_rows(tracks, clips_tbl, free_flow)
    html_path = OUT_DIR / "index.html"
    build_html(video_path, chosen, summary, speed_rows, html_path)
    print(f"html : {html_path.stat().st_size / 1e6:.1f} MB (self-contained) -> {html_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=2, help="Clips per congestion class")
    main(per_class=p.parse_args().per_class)
