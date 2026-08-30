#!/usr/bin/env python3
"""Download CIFAKE from Kaggle into the repository's ignored raw-data folder.

This is intentionally separate from manifest creation and training. It needs a
Kaggle account plus a configured API token (normally ``~/.kaggle/kaggle.json``).
"""

from __future__ import annotations

import argparse
from pathlib import Path


DATASET_SLUG = "birdy654/cifake-real-and-ai-generated-synthetic-images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/cifake"),
        help="Ignored local directory where Kaggle will unpack CIFAKE.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow the Kaggle client to re-download files already present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Kaggle support is not installed. Run `.venv/bin/python -m pip install "
            "kaggle`, configure your Kaggle API token, then try again."
        ) from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        DATASET_SLUG,
        path=str(args.out_dir),
        unzip=True,
        force=args.force,
        quiet=False,
    )
    print(f"Downloaded CIFAKE to {args.out_dir}")
    print("Next: .venv/bin/python scripts/make_cifake_manifest.py")


if __name__ == "__main__":
    main()
