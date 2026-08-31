#!/usr/bin/env python3
"""Prepare a safe, balanced WildFake training/validation manifest.

Only official WildFake *training* metadata is accepted.  This is deliberately
more conservative than the hackathon brief: every COCO, DALL-E, and Advanced
row is excluded so the organiser's COCO val2017 / DALL-E Advanced demo set
cannot enter training or model-selection data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


OUTPUT_COLUMNS = (
    "image_path",
    "label",
    "source_dataset",
    "generator",
    "split",
    "license_notes",
)
REQUIRED_COLUMNS = {"Generator", "Architecture", "Category", "IsAdvanced", "IsFake", "Image_path"}
PROTECTED_TERMS = ("coco", "dalle", "dall-e", "dall·e", "dall e", "advanced")
LICENSE_NOTE = (
    "WildFake official ModelScope release; official train metadata only; "
    "COCO, DALL-E, and Advanced rows excluded"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-metadata",
        nargs="+",
        type=Path,
        required=True,
        help="Official WildFake CSV files whose filenames identify them as train splits.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Directory containing WildFake's extracted Images tree.",
    )
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-val", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--max-per-label", type=int, default=1_000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_official_train_metadata(path: Path) -> None:
    name = path.name.lower()
    if "train" not in name or any(term in name for term in ("test", "holdout", "challenge", "demo")):
        raise ValueError(
            f"Refusing '{path.name}'. WildFake metadata must be an official train split, "
            "never test, holdout, challenge, or demo data."
        )


def row_text(row: dict[str, str]) -> str:
    return " ".join(
        row.get(name, "")
        for name in ("Generator", "Architecture", "Category", "IsAdvanced", "Image_path")
    ).lower()


def read_approved_rows(path: Path, images_root: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    ensure_official_train_metadata(path)
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata has no header: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Metadata {path} is missing columns: {sorted(missing)}")

        accepted: list[dict[str, str]] = []
        excluded: Counter[str] = Counter()
        for row in reader:
            text = row_text(row)
            matched = next((term for term in PROTECTED_TERMS if term in text), None)
            if matched:
                excluded[f"protected_{matched}"] += 1
                continue
            if row["IsFake"] not in {"0", "1"}:
                excluded["invalid_label"] += 1
                continue
            relative_path = row["Image_path"].removeprefix("./")
            image_path = images_root / relative_path
            if not image_path.is_file():
                excluded["missing_image"] += 1
                continue
            accepted.append(
                {
                    "image_path": image_path.as_posix(),
                    "label": row["IsFake"],
                    "source_dataset": "wildfake",
                    "generator": row["Architecture"].strip() or row["Generator"].strip(),
                    "split": "",
                    "license_notes": LICENSE_NOTE,
                }
            )
    return accepted, excluded


def select_split(
    rows: list[dict[str, str]], max_per_label: int, val_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if max_per_label <= 0:
        raise ValueError("--max-per-label must be positive")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    rng = random.Random(seed)
    selected_train: list[dict[str, str]] = []
    selected_val: list[dict[str, str]] = []
    for label in ("0", "1"):
        group = [row for row in rows if row["label"] == label]
        rng.shuffle(group)
        group = group[:max_per_label]
        if not group:
            raise ValueError(f"No approved WildFake rows found for label {label}")
        val_count = max(1, round(len(group) * val_fraction))
        selected_val.extend({**row, "split": "val"} for row in group[:val_count])
        selected_train.extend({**row, "split": "train"} for row in group[val_count:])
    rng.shuffle(selected_train)
    rng.shuffle(selected_val)
    return selected_train, selected_val


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, str]] = []
    excluded: Counter[str] = Counter()
    for path in args.train_metadata:
        rows, path_excluded = read_approved_rows(path, args.images_root)
        all_rows.extend(rows)
        excluded.update(path_excluded)

    train_rows, val_rows = select_split(
        all_rows, args.max_per_label, args.val_fraction, args.seed
    )
    selected_text = " ".join(
        row["image_path"].lower() + " " + row["generator"].lower()
        for row in [*train_rows, *val_rows]
    )
    if any(term in selected_text for term in PROTECTED_TERMS):
        raise RuntimeError("Protected WildFake content reached the selected manifest")

    write_csv(args.out_train, train_rows)
    write_csv(args.out_val, val_rows)
    audit = {
        "policy": "official train metadata only; exclude all COCO, DALL-E, and Advanced rows",
        "metadata_inputs": [path.as_posix() for path in args.train_metadata],
        "images_root": args.images_root.as_posix(),
        "excluded_rows": dict(sorted(excluded.items())),
        "selected_train_rows": len(train_rows),
        "selected_val_rows": len(val_rows),
        "selected_by_label": {
            "train": dict(Counter(row["label"] for row in train_rows)),
            "val": dict(Counter(row["label"] for row in val_rows)),
        },
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
