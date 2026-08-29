#!/usr/bin/env python3
"""Evaluate a checkpoint across clean and transformed image conditions.

This produces the compact robustness table requested by the hackathon. Use a
small `--limit` locally to keep laptop inference under control.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.evaluation import evaluate_predictions
from reality_check.prediction import (
    predict_mock,
    predict_with_checkpoint,
    select_balanced_image_paths_from_manifest,
    write_predictions_json,
)
from reality_check.transforms import EVAL_TRANSFORMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/reports/robustness"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=list(EVAL_TRANSFORMS),
        help="Subset of transform names to evaluate.",
    )
    return parser.parse_args()


def write_summary_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(rows: list[dict[str, float | int | str]], path: Path) -> None:
    lines = [
        "| Condition | Count | ROC AUC | Accuracy | Precision | Recall | F1 | AUC Drop vs Clean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {count} | {roc_auc:.4f} | {accuracy:.4f} | "
            "{precision:.4f} | {recall:.4f} | {f1:.4f} | {auc_drop_vs_clean:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    unknown_conditions = sorted(set(args.conditions) - set(EVAL_TRANSFORMS))
    if unknown_conditions:
        raise SystemExit(f"Unknown conditions: {', '.join(unknown_conditions)}")
    if not args.mock and args.checkpoint is None:
        raise SystemExit("Real robustness evaluation requires --checkpoint, or use --mock.")

    image_paths = select_balanced_image_paths_from_manifest(args.manifest, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.out_dir / "predictions"

    rows: list[dict[str, float | int | str]] = []
    clean_auc: float | None = None
    for condition in args.conditions:
        prediction_path = predictions_dir / f"{condition}_preds.json"
        print(f"Evaluating condition: {condition}")

        if args.mock:
            predictions = predict_mock(image_paths)
        else:
            predictions = predict_with_checkpoint(
                image_paths=image_paths,
                checkpoint_path=args.checkpoint,
                transform_name=condition,
                image_size=args.image_size,
                batch_size=args.batch_size,
            )

        write_predictions_json(predictions, prediction_path)
        result = evaluate_predictions(
            manifest_path=args.manifest,
            predictions_path=prediction_path,
            threshold=args.threshold,
        )

        if condition == "clean":
            clean_auc = result.roc_auc
        auc_drop = (clean_auc - result.roc_auc) if clean_auc is not None else 0.0
        rows.append(
            {
                "condition": condition,
                "count": result.count,
                "positives": result.positives,
                "negatives": result.negatives,
                "roc_auc": result.roc_auc,
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "auc_drop_vs_clean": auc_drop,
            }
        )

    clean_row = next(row for row in rows if row["condition"] == "clean")
    transformed_rows = [row for row in rows if row["condition"] != "clean"]
    robust_auc = sum(float(row["roc_auc"]) for row in transformed_rows) / max(
        len(transformed_rows), 1
    )
    final_score = 0.5 * float(clean_row["roc_auc"]) + 0.5 * robust_auc
    summary = {
        "manifest": args.manifest.as_posix(),
        "checkpoint": args.checkpoint.as_posix() if args.checkpoint else None,
        "limit": args.limit,
        "count_per_condition": rows[0]["count"],
        "clean_auc": clean_row["roc_auc"],
        "mean_transformed_auc": robust_auc,
        "final_score": final_score,
        "worst_condition": min(rows, key=lambda row: float(row["roc_auc"]))[
            "condition"
        ],
        "worst_condition_auc": min(float(row["roc_auc"]) for row in rows),
    }

    write_summary_csv(rows, args.out_dir / "robustness_summary.csv")
    write_summary_markdown(rows, args.out_dir / "robustness_summary.md")
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote robustness outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

