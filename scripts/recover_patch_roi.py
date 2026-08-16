"""Recover the pixel rectangle the original researchers cropped for
traffic_patches.mat, by template-matching each 48x48 patch frame against
candidate crops of the corresponding real video frame (over position, size,
and frame-offset), since no coordinates are stored anywhere in the dataset.

This gives us an ROI grounded in the dataset's own published methodology
(the exact region the UCSD TrafficDB authors used for their traffic
classifier) instead of one we derive ourselves from detections.

Usage:
    uv run python scripts/recover_patch_roi.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "highway-traffic-videos-dataset"
VIDEO_DIR = DATASET_DIR / "video"
OUT_DIR = REPO_ROOT / "outputs" / "calibration"
FRAME_W, FRAME_H = 320, 240
PATCH_SIZE = 48

# Candidate clips to cross-validate the recovered rectangle against (index into
# ImageMaster / imgdb, 0-based). Mix of congestion levels for robustness.
CANDIDATE_INDICES = [0, 10, 100, 200]

# Coarse search grid over (x0, y0, w, h) in native 320x240 coordinates.
X0_RANGE = range(0, 160, 8)
Y0_RANGE = range(40, 160, 8)
W_RANGE = range(120, 321, 12)
H_RANGE = range(80, 201, 12)
FRAME_OFFSETS = [0, 1, 2, 3]  # candidate alignments between patch frame 0 and video frame index


def load_image_master() -> list[dict]:
    d = sio.loadmat(DATASET_DIR / "ImageMaster.mat")
    entries = d["imagemaster"][0]
    out = []
    for e in entries:
        rec = e[0, 0]
        out.append({"num": int(rec["num"][0, 0]), "root": str(rec["root"][0]), "class": str(rec["class"][0])})
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-6:
        return -1.0
    return float(np.dot(a, b) / denom)


def get_video_frame_gray(filename: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{filename}.avi"))
    for i in range(frame_idx + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def score_rect(frame_gray: np.ndarray, patch_gray: np.ndarray, x0: int, y0: int, w: int, h: int) -> float:
    crop = frame_gray[y0 : y0 + h, x0 : x0 + w]
    resized = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    return ncc(resized, patch_gray)


def search_best_rect(frame_gray: np.ndarray, patch_gray: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
    best_score = -2.0
    best_rect = None
    for x0 in X0_RANGE:
        for y0 in Y0_RANGE:
            for w in W_RANGE:
                if x0 + w > FRAME_W:
                    continue
                for h in H_RANGE:
                    if y0 + h > FRAME_H:
                        continue
                    s = score_rect(frame_gray, patch_gray, x0, y0, w, h)
                    if s > best_score:
                        best_score = s
                        best_rect = (x0, y0, w, h)
    # local refinement around the coarse best
    x0, y0, w, h = best_rect
    for dx in range(-10, 11, 2):
        for dy in range(-10, 11, 2):
            for dw in range(-16, 17, 4):
                for dh in range(-16, 17, 4):
                    nx0, ny0, nw, nh = x0 + dx, y0 + dy, w + dw, h + dh
                    if nx0 < 0 or ny0 < 0 or nx0 + nw > FRAME_W or ny0 + nh > FRAME_H:
                        continue
                    if nw < 20 or nh < 20:
                        continue
                    s = score_rect(frame_gray, patch_gray, nx0, ny0, nw, nh)
                    if s > best_score:
                        best_score = s
                        best_rect = (nx0, ny0, nw, nh)
    return best_rect, best_score


def main():
    meta = load_image_master()
    d = sio.loadmat(DATASET_DIR / "traffic_patches.mat")
    imgdb = d["imgdb"][0]

    results = []
    for idx in CANDIDATE_INDICES:
        entry = meta[idx]
        filename = entry["root"]
        patch_stack = imgdb[idx]  # (48, 48, num_patch_frames)
        patch0 = patch_stack[:, :, 0]

        best_overall = (-2.0, None, None)
        for offset in FRAME_OFFSETS:
            frame_gray = get_video_frame_gray(filename, offset)
            if frame_gray is None:
                continue
            rect, score = search_best_rect(frame_gray, patch0)
            print(f"  clip={filename} offset={offset} -> rect={rect} score={score:.4f}")
            if score > best_overall[0]:
                best_overall = (score, rect, offset)

        score, rect, offset = best_overall
        print(f"clip idx={idx} ({filename}, {entry['class']}): best rect={rect} offset={offset} score={score:.4f}")
        results.append({"clip": filename, "class": entry["class"], "rect": rect, "offset": offset, "score": score})

    # aggregate: median rectangle across candidates (robust to any single bad match)
    rects = np.array([r["rect"] for r in results])
    median_rect = tuple(int(v) for v in np.median(rects, axis=0))
    print(f"\nMedian recovered rectangle (x0,y0,w,h): {median_rect}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "patch_roi_recovery.json").write_text(
        json.dumps({"candidates": results, "median_rect": median_rect}, indent=2)
    )

    # visualize: overlay median rect + each candidate rect on a sample frame
    sample_frame = cv2.imread(str(OUT_DIR / "_sample_frame.png"))
    if sample_frame is None:
        sample_frame_gray = get_video_frame_gray(results[0]["clip"], results[0]["offset"])
        sample_frame = cv2.cvtColor(sample_frame_gray, cv2.COLOR_GRAY2BGR)
    overlay = sample_frame.copy()
    for r in results:
        x0, y0, w, h = r["rect"]
        cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 255, 255), 1)
    x0, y0, w, h = median_rect
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), 1)
    big = cv2.resize(overlay, (FRAME_W * 3, FRAME_H * 3), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT_DIR / "patch_roi_recovery.png"), big)

    # side-by-side: best-matching candidate's resized crop vs. the actual patch
    best = max(results, key=lambda r: r["score"])
    frame_gray = get_video_frame_gray(best["clip"], best["offset"])
    x0, y0, w, h = best["rect"]
    crop = frame_gray[y0 : y0 + h, x0 : x0 + w]
    resized = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    idx = [i for i, m in enumerate(meta) if m["root"] == best["clip"]][0]
    patch0 = imgdb[idx][:, :, 0]
    side_by_side = np.hstack([cv2.resize(resized, (192, 192), interpolation=cv2.INTER_NEAREST),
                               cv2.resize(patch0, (192, 192), interpolation=cv2.INTER_NEAREST)])
    cv2.imwrite(str(OUT_DIR / "patch_roi_match_check.png"), side_by_side)

    print(f"\nWrote diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
