#!/usr/bin/env python3
"""Apply robustness transforms to a small manifest and save preview images.

This script is data-pipeline validation only. It does not create a model,
train a model, or change any dataset labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image, ImageDraw

from reality_check.transforms import EVAL_TRANSFORMS, apply_named_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/cifake_sample.csv"),
        help="CSV manifest with image_path and label columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports/transform_smoke_test"),
        help="Directory where transformed previews will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Number of manifest rows to preview.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=list(EVAL_TRANSFORMS),
        help="Named transform conditions to apply.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_balanced_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(rows):
        return rows

    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    if {"0", "1"}.issubset(by_label):
        real_target = limit // 2
        fake_target = limit - real_target
        selected = by_label["0"][:real_target] + by_label["1"][:fake_target]
        if len(selected) < limit:
            selected_paths = {row["image_path"] for row in selected}
            selected.extend(
                row for row in rows if row["image_path"] not in selected_paths
            )
        return selected[:limit]

    return rows[:limit]


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def make_contact_sheet(
    rows: list[dict[str, str]],
    transformed: dict[tuple[int, str], Image.Image],
    conditions: list[str],
    out_path: Path,
) -> None:
    preview_size = 96
    header_height = 24
    label_width = 120
    cell_width = preview_size + 24
    cell_height = preview_size + header_height + 12

    sheet_width = label_width + cell_width * len(conditions)
    sheet_height = header_height + cell_height * len(rows)
    sheet = Image.new("RGB", (sheet_width, sheet_height), color="white")
    draw = ImageDraw.Draw(sheet)

    for col, condition in enumerate(conditions):
        x = label_width + col * cell_width
        draw.text((x + 4, 4), condition, fill="black")

    for row_idx, row in enumerate(rows):
        y = header_height + row_idx * cell_height
        label = "fake" if row["label"] == "1" else "real"
        draw.text((4, y + 36), f"{row_idx}: {label}", fill="black")
        for col, condition in enumerate(conditions):
            x = label_width + col * cell_width
            image = transformed[(row_idx, condition)].resize(
                (preview_size, preview_size), resample=Image.Resampling.NEAREST
            )
            sheet.paste(image, (x + 4, y + header_height))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    unknown_conditions = sorted(set(args.conditions) - set(EVAL_TRANSFORMS))
    if unknown_conditions:
        raise SystemExit(f"Unknown conditions: {', '.join(unknown_conditions)}")

    rows = select_balanced_rows(read_manifest(args.manifest), args.limit)
    if not rows:
        raise SystemExit(f"No rows found in {args.manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    transformed: dict[tuple[int, str], Image.Image] = {}
    summary_rows: list[dict[str, str | int]] = []

    for row_idx, row in enumerate(rows):
        image_path = Path(row["image_path"])
        with Image.open(image_path) as image:
            original = image.convert("RGB")
            original_size = original.size
            for condition in args.conditions:
                output = apply_named_transform(original, condition)
                if output.size != original_size:
                    raise RuntimeError(
                        f"{condition} changed size for {image_path}: "
                        f"{original_size} -> {output.size}"
                    )

                transformed[(row_idx, condition)] = output
                output_name = (
                    f"{row_idx:02d}_label{row['label']}_"
                    f"{condition}_{safe_name(image_path.stem)}.png"
                )
                output_path = args.out_dir / output_name
                output.save(output_path)
                summary_rows.append(
                    {
                        "row_index": row_idx,
                        "image_path": image_path.as_posix(),
                        "label": row["label"],
                        "condition": condition,
                        "output_path": output_path.as_posix(),
                        "width": output.width,
                        "height": output.height,
                    }
                )

    contact_sheet_path = args.out_dir / "contact_sheet.png"
    make_contact_sheet(rows, transformed, list(args.conditions), contact_sheet_path)

    summary_path = args.out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Applied {len(args.conditions)} transforms to {len(rows)} images.")
    print(f"Wrote previews to {args.out_dir}")
    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
