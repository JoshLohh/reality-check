from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.training import (
    TrainingSafetyError,
    build_training_plan,
    enforce_training_safety,
    source_label_sample_weights,
)
from reality_check.dataset import ManifestRecord


def base_config(enabled: bool = False) -> dict:
    return {
        "data": {
            "manifests": {
                "train": "data/manifests/train.csv",
                "val": "data/manifests/val.csv",
            },
            "input_size": 224,
            "batch_size": 32,
        },
        "model": {
            "backbone": "efficientnet_b0",
            "pretrained": True,
            "classifier": {"dropout": 0.2},
        },
        "training": {
            "enabled": enabled,
            "max_epochs": 10,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "class_balance": "weighted_sampler",
            "early_stopping_metric": "val_roc_auc",
        },
    }


class TrainingSafetyTests(unittest.TestCase):
    def test_training_is_blocked_when_config_disabled(self) -> None:
        with self.assertRaisesRegex(TrainingSafetyError, "disabled in the config"):
            enforce_training_safety(base_config(enabled=False), allow_training=True)

    def test_training_is_blocked_without_cli_flag(self) -> None:
        with self.assertRaisesRegex(TrainingSafetyError, "--allow-training"):
            enforce_training_safety(base_config(enabled=True), allow_training=False)

    def test_training_safety_passes_only_with_both_switches(self) -> None:
        enforce_training_safety(base_config(enabled=True), allow_training=True)

    def test_build_training_plan_reads_expected_config_fields(self) -> None:
        plan = build_training_plan(base_config())

        self.assertEqual(plan.train_manifest, Path("data/manifests/train.csv"))
        self.assertEqual(plan.val_manifest, Path("data/manifests/val.csv"))
        self.assertEqual(plan.image_size, 224)
        self.assertEqual(plan.model.backbone, "efficientnet_b0")

    def test_source_label_sampler_equalizes_dataset_label_groups(self) -> None:
        records = [
            ManifestRecord(Path("a.jpg"), 0, "a", "real", "train", "test"),
            ManifestRecord(Path("b.jpg"), 0, "a", "real", "train", "test"),
            ManifestRecord(Path("c.jpg"), 1, "a", "fake", "train", "test"),
        ]
        self.assertEqual(source_label_sample_weights(records), [0.5, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
