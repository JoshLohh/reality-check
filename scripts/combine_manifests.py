#!/usr/bin/env python3
"""Build balanced combined train/validation manifests from approved sources.

This tool never accepts a test, holdout, or challenge-demo manifest as input.
It deterministically caps each (source_dataset, label) group so that a large
source cannot dominate the training run.  It is intentionally generic: each
source manifest must already have been prepared and safety-audited.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_COLUMNS = (
    "image_path",
    "label",
    "source_dataset",
    "generator",
    "split",
    "license_notes",
)
FORBIDDEN_MANIFEST_TERMS = ("test", "holdout", "challenge", "demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--val-manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-val", type=Path, required=True)
    parser.add_argument(
        "--train-per-source-label",
        type=int,
        default=5_000,
        help="Maximum rows from each (source_dataset, label) training group.",
    )
    parser.add_argument(
        "--val-per-source-label",
        type=int,
        default=1_000,
        help="Maximum rows from each (source_dataset, label) validation group.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def reject_unsafe_name(path: Path) -> None:
    name = path.name.lower()
    if any(term in name for term in FORBIDDEN_MANIFEST_TERMS):
        raise ValueError(
            f"Refusing unsafe manifest name '{path.name}'. Test, holdout, and "
            "challenge/demo data must never enter this combined training run."
        )


def read_rows(path: Path) -> list[dict[str, str]]:
    reject_unsafe_name(path)
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
        rows = list(reader)

    for row in rows:
        if row["label"] not in {"0", "1"}:
            raise ValueError(f"Only binary labels 0/1 are allowed: {path}")
        if not row["source_dataset"].strip():
            raise ValueError(f"Missing source_dataset in {path}")
    return rows


def select_balanced(
    rows: list[dict[str, str]], cap: int, split: str, rng: random.Random
) -> list[dict[str, str]]:
    if cap <= 0:
        raise ValueError("Per-source-label caps must be positive")
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["source_dataset"], row["label"])].append(row)

    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        for row in group[:cap]:
            selected.append({**row, "split": split})
    rng.shuffle(selected)
    return selected


def ensure_disjoint(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> None:
    train_paths = {row["image_path"] for row in train_rows}
    val_paths = {row["image_path"] for row in val_rows}
    overlap = train_paths & val_paths
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(
            "Training and validation manifests overlap. Example paths: "
            + ", ".join(sample)
        )


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter((row["source_dataset"], row["label"]) for row in rows)
    return {f"{source}/label_{label}": count for (source, label), count in sorted(counts.items())}


def main() -> None:
    args = parse_args()
    train_rows = [row for path in args.train_manifests for row in read_rows(path)]
    val_rows = [row for path in args.val_manifests for row in read_rows(path)]
    ensure_disjoint(train_rows, val_rows)

    rng = random.Random(args.seed)
    selected_train = select_balanced(
        train_rows, args.train_per_source_label, "train", rng
    )
    selected_val = select_balanced(val_rows, args.val_per_source_label, "val", rng)
    ensure_disjoint(selected_train, selected_val)
    write_rows(args.out_train, selected_train)
    write_rows(args.out_val, selected_val)

    print("Wrote combined manifests")
    print(f"train: {args.out_train} ({len(selected_train)} rows) {format_counts(selected_train)}")
    print(f"val: {args.out_val} ({len(selected_val)} rows) {format_counts(selected_val)}")


if __name__ == "__main__":
    main()
