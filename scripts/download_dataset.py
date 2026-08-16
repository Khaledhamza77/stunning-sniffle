"""Download the Kaggle highway traffic videos dataset via kagglehub.

Requires Kaggle API credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME /
KAGGLE_KEY env vars). See https://github.com/Kaggle/kagglehub for auth setup.

Usage:
    uv run python scripts/download_dataset.py
"""

import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> Path:
    cache_path = Path(kagglehub.dataset_download("aryashah2k/highway-traffic-videos-dataset"))
    print(f"Downloaded dataset to kagglehub cache: {cache_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "highway-traffic-videos-dataset"
    if dest.exists():
        print(f"Destination already exists, skipping copy: {dest}")
    else:
        shutil.copytree(cache_path, dest)
        print(f"Copied dataset into repo-local (gitignored) folder: {dest}")

    return dest


if __name__ == "__main__":
    main()
