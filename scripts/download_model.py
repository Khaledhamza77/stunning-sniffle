"""Fetch the pretrained detector weights.

Strictly optional: Ultralytics downloads the weights by itself the first time a
model is constructed, so the pipeline works from a clean checkout without this.
It exists for two practical reasons:

  - it lets the download happen ahead of time, so the pipeline itself can then be
    run without network access;
  - it makes the model an explicit, inspectable step rather than a side effect,
    and prints exactly which file is used and where it landed.

Nothing is trained or fine-tuned. These are stock COCO-pretrained weights, and the
version is pinned by the `ultralytics` version in requirements.txt.

Ultralytics resolves a bare filename against the current working directory, so run
this (and the rest of the pipeline) from the repository root.

Usage:
    uv run python scripts/download_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_scene as cal  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> Path:
    from ultralytics import YOLO

    target = REPO_ROOT / cal.MODEL_NAME
    if target.exists():
        print(f"Already present: {target} ({target.stat().st_size / 1e6:.0f} MB)")
        return target

    print(f"Fetching {cal.MODEL_NAME} (Ultralytics will download it on construction)...")
    YOLO(str(target) if target.exists() else cal.MODEL_NAME)

    # Ultralytics writes to the working directory; report wherever it actually landed
    found = target if target.exists() else Path(cal.MODEL_NAME).resolve()
    if not found.exists():
        raise RuntimeError(f"Expected weights at {target}, but no file was written")
    print(f"Ready: {found} ({found.stat().st_size / 1e6:.0f} MB)")
    return found


if __name__ == "__main__":
    path = main()
    print(f"\nDetector : {cal.MODEL_NAME} (COCO-pretrained, not fine-tuned)")
    print(f"Inference: imgsz={cal.IMGSZ}, conf={cal.DETECT_CONF}")
    print(f"Tracker  : bytetrack.yaml (ships with ultralytics, nothing to download)")
