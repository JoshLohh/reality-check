"""Evaluation helpers for prediction JSON files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from reality_check.dataset import read_manifest
from reality_check.prediction import load_predictions_json


@dataclass(frozen=True)
class EvaluationResult:
    count: int
    positives: int
    negatives: int
    threshold: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def roc_auc_score(labels: list[int], scores: list[float]) -> float:
    """Compute ROC AUC with average ranks for tied scores."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")

    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires at least one positive and one negative label")

    sorted_pairs = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(sorted_pairs):
        tie_end = index + 1
        while tie_end < len(sorted_pairs) and sorted_pairs[tie_end][1] == sorted_pairs[index][1]:
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        for rank_index in range(index, tie_end):
            original_index = sorted_pairs[rank_index][0]
            ranks[original_index] = average_rank
        index = tie_end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def binary_metrics(
    labels: list[int], scores: list[float], threshold: float
) -> tuple[float, float, float, float]:
    predictions = [1 if score >= threshold else 0 for score in scores]
    tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, labels))
    tn = sum(pred == 0 and label == 0 for pred, label in zip(predictions, labels))
    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, labels))
    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, labels))

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return accuracy, precision, recall, f1


def evaluate_predictions(
    manifest_path: Path, predictions_path: Path, threshold: float = 0.5
) -> EvaluationResult:
    records = read_manifest(manifest_path)
    labels_by_path = {record.image_path.as_posix(): record.label for record in records}
    prediction_rows = load_predictions_json(predictions_path)

    labels: list[int] = []
    scores: list[float] = []
    missing_paths: list[str] = []
    for row in prediction_rows:
        image_path = str(row["image_path"])
        if image_path not in labels_by_path:
            missing_paths.append(image_path)
            continue
        score = float(row["pred"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Prediction must be in [0, 1], got {score}: {image_path}")
        labels.append(labels_by_path[image_path])
        scores.append(score)

    if missing_paths:
        preview = ", ".join(missing_paths[:3])
        raise ValueError(
            f"{len(missing_paths)} prediction paths were not found in manifest. "
            f"First examples: {preview}"
        )
    if not labels:
        raise ValueError("No overlapping prediction paths found")

    auc = roc_auc_score(labels, scores)
    accuracy, precision, recall, f1 = binary_metrics(labels, scores, threshold)
    return EvaluationResult(
        count=len(labels),
        positives=sum(label == 1 for label in labels),
        negatives=sum(label == 0 for label in labels),
        threshold=threshold,
        roc_auc=auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def write_evaluation_outputs(result: EvaluationResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.__dict__
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)

