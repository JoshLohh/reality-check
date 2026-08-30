#!/usr/bin/env python3
"""Stream SID_Set from Hugging Face and prepare binary training manifests.

Only source labels 0 (real) and 1 (full synthetic) are exported. Label 2
(tampered) is deliberately excluded from this binary AIGC detector. Images are
saved as RGB PNG files so file extension is not a shortcut correlated with the
label. The command defaults to a balanced, laptop-manageable subset; use
``--max-per-label 0`` only when you explicitly intend to materialize a split.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.data_preparation import (
    MANIFEST_COLUMNS,
    manifest_split_for_sid_split,
    map_sid_label,
    sid_image_path,
)


DATASET_ID = "saberzl/SID_Set"
LICENSE_NOTE = "SID_Set CC-BY-4.0; retain dataset attribution in submission"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Hugging Face split names to export, e.g. train val or test.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/sid_set"),
        help="Ignored directory where retained RGB PNG images are written.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory where sid_set_<split>.csv manifests are written.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=5000,
        help="Balanced maximum per retained label per split; 0 means no limit.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download through the Hugging Face cache instead of streaming.",
    )
    return parser.parse_args()


def should_stop(counts: Counter[int], max_per_label: int) -> bool:
    return max_per_label > 0 and all(counts[label] >= max_per_label for label in (0, 1))


def row_for_example(
    example: dict[str, Any],
    source_split: str,
    index: int,
    raw_root: Path,
) -> dict[str, str] | None:
    """Save one accepted example and return its manifest row, else exclude it."""

    label = map_sid_label(int(example["label"]))
    if label is None:
        return None

    image_path = sid_image_path(raw_root, source_split, label.class_name, index)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        image = example["image"]
        image.convert("RGB").save(image_path, format="PNG")

    return {
        "image_path": image_path.as_posix(),
        "label": str(label.binary_label),
        "source_dataset": "sid_set",
        "generator": label.generator,
        "split": manifest_split_for_sid_split(source_split),
        "license_notes": LICENSE_NOTE,
    }


def export_split(
    dataset: Iterable[dict[str, Any]],
    source_split: str,
    raw_root: Path,
    max_per_label: int,
) -> tuple[list[dict[str, str]], Counter[int]]:
    """Filter one Hub split to real/full-synthetic images and export it."""

    rows: list[dict[str, str]] = []
    counts: Counter[int] = Counter()
    skipped: Counter[int] = Counter()
    for index, example in enumerate(dataset):
        raw_label = int(example["label"])
        mapped = map_sid_label(raw_label)
        if mapped is None:
            skipped[raw_label] += 1
            continue
        if max_per_label > 0 and counts[mapped.binary_label] >= max_per_label:
            continue

        row = row_for_example(example, source_split, index, raw_root)
        if row is None:
            continue
        rows.append(row)
        counts[mapped.binary_label] += 1
        if should_stop(counts, max_per_label):
            break

    if skipped:
        print(f"{source_split}: excluded labels {dict(skipped)}")
    return rows, counts


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No label 0/1 rows were exported for {path.stem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_per_label < 0:
        raise SystemExit("--max-per-label must be zero or a positive integer")
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Hugging Face Datasets is not installed. Run `.venv/bin/python -m pip "
            "install datasets`, then try again."
        ) from exc

    for source_split in args.splits:
        dataset = load_dataset(
            DATASET_ID,
            split=source_split,
            streaming=not args.no_streaming,
        )
        rows, counts = export_split(
            dataset, source_split, args.raw_root, args.max_per_label
        )
        manifest_split = manifest_split_for_sid_split(source_split)
        manifest_path = args.manifest_dir / f"sid_set_{manifest_split}.csv"
        write_manifest(manifest_path, rows)
        print(
            f"{source_split}: wrote {len(rows)} rows to {manifest_path} "
            f"({counts[0]} real, {counts[1]} full synthetic; label 2 excluded)"
        )


if __name__ == "__main__":
    main()
