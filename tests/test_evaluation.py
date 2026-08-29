from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.evaluation import evaluate_predictions, roc_auc_score


class EvaluationTests(unittest.TestCase):
    def test_roc_auc_perfect_ranking(self) -> None:
        self.assertEqual(roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_evaluate_predictions_against_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            predictions = root / "preds.json"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "image_path",
                        "label",
                        "source_dataset",
                        "generator",
                        "split",
                        "license_notes",
                    ],
                )
                writer.writeheader()
                for path, label in [("a.jpg", 0), ("b.jpg", 1)]:
                    writer.writerow(
                        {
                            "image_path": path,
                            "label": label,
                            "source_dataset": "unit",
                            "generator": "unit",
                            "split": "sample",
                            "license_notes": "unit",
                        }
                    )
            predictions.write_text(
                json.dumps(
                    [
                        {"image_path": "a.jpg", "pred": 0.1},
                        {"image_path": "b.jpg", "pred": 0.9},
                    ]
                ),
                encoding="utf-8",
            )

            result = evaluate_predictions(manifest, predictions)

        self.assertEqual(result.count, 2)
        self.assertEqual(result.roc_auc, 1.0)
        self.assertEqual(result.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()

