#!/usr/bin/env python3
"""Create hackathon-safe manifests for the CIFAKE dataset.

This script only indexes local image files and writes CSV metadata. It does
not train, transform, or copy images.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Row:
    image_path: str
    label: int
    source_dataset: str
    generator: str
    split: str
    license_notes: str


def find_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def rows_for_folder(folder: Path, label: int, generator: str, split: str) -> list[Row]:
    return [
        Row(
            image_path=path.as_posix(),
            label=label,
            source_dataset="cifake",
            generator=generator,
            split=split,
            license_notes="CIFAKE public Kaggle dataset; verify terms before submission",
        )
        for path in find_images(folder)
    ]


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(Row.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def stratified_train_val_split(
    real_rows: list[Row], fake_rows: list[Row], val_fraction: float, seed: int
) -> tuple[list[Row], list[Row]]:
    rng = random.Random(seed)
    real_rows = list(real_rows)
    fake_rows = list(fake_rows)
    rng.shuffle(real_rows)
    rng.shuffle(fake_rows)

    real_val_count = round(len(real_rows) * val_fraction)
    fake_val_count = round(len(fake_rows) * val_fraction)

    val_rows = [
        *[replace_split(row, "val") for row in real_rows[:real_val_count]],
        *[replace_split(row, "val") for row in fake_rows[:fake_val_count]],
    ]
    train_rows = [
        *real_rows[real_val_count:],
        *fake_rows[fake_val_count:],
    ]

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def replace_split(row: Row, split: str) -> Row:
    return Row(
        image_path=row.image_path,
        label=row.label,
        source_dataset=row.source_dataset,
        generator=row.generator,
        split=split,
        license_notes=row.license_notes,
    )


def make_sample(rows: list[Row], per_class: int, seed: int) -> list[Row]:
    rng = random.Random(seed)
    real = [row for row in rows if row.label == 0]
    fake = [row for row in rows if row.label == 1]
    rng.shuffle(real)
    rng.shuffle(fake)
    sample = [*real[:per_class], *fake[:per_class]]
    rng.shuffle(sample)
    return [replace_split(row, "sample") for row in sample]


def summarize(name: str, rows: list[Row]) -> str:
    real = sum(row.label == 0 for row in rows)
    fake = sum(row.label == 1 for row in rows)
    return f"{name}: {len(rows)} rows ({real} real, {fake} fake)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/cifake/archive"),
        help="Path containing CIFAKE train/test folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory for output CSV manifests.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-class", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_real_dir = args.root / "train" / "REAL"
    train_fake_dir = args.root / "train" / "FAKE"
    test_real_dir = args.root / "test" / "REAL"
    test_fake_dir = args.root / "test" / "FAKE"

    required_dirs = [train_real_dir, train_fake_dir, test_real_dir, test_fake_dir]
    missing_dirs = [path for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        missing = "\n".join(f"- {path}" for path in missing_dirs)
        raise SystemExit(f"Missing expected CIFAKE directories:\n{missing}")

    train_real = rows_for_folder(train_real_dir, 0, "cifar10", "train")
    train_fake = rows_for_folder(train_fake_dir, 1, "stable_diffusion", "train")
    test_rows = [
        *rows_for_folder(test_real_dir, 0, "cifar10", "test_clean"),
        *rows_for_folder(test_fake_dir, 1, "stable_diffusion", "test_clean"),
    ]

    train_rows, val_rows = stratified_train_val_split(
        train_real, train_fake, args.val_fraction, args.seed
    )
    sample_rows = make_sample(train_rows, args.sample_per_class, args.seed)

    outputs = {
        "train.csv": train_rows,
        "val.csv": val_rows,
        "test_clean.csv": test_rows,
        "cifake_sample.csv": sample_rows,
    }

    for filename, rows in outputs.items():
        write_csv(args.out_dir / filename, rows)

    datasets_md = args.out_dir / "datasets.md"
    datasets_md.write_text(
        "\n".join(
            [
                "# Dataset Inventory",
                "",
                "## CIFAKE",
                "",
                "- Source: Kaggle dataset `birdy654/cifake-real-and-ai-generated-synthetic-images`",
                "- Local root: `data/raw/cifake/archive`",
                "- Label mapping: `REAL -> 0`, `FAKE -> 1`",
                "- Real generator/source: `cifar10`",
                "- AIGC generator/source: `stable_diffusion`",
                "- Role: smoke test and initial baseline data",
                "- Caveat: images are 32x32, so CIFAKE alone is not enough for the final robustness claim",
                "- Hackathon note: no challenge validation-demo data is included here",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(summarize("train.csv", train_rows))
    print(summarize("val.csv", val_rows))
    print(summarize("test_clean.csv", test_rows))
    print(summarize("cifake_sample.csv", sample_rows))
    print(f"Wrote manifests to {args.out_dir}")


if __name__ == "__main__":
    main()
