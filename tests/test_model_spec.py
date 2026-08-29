from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.model import (
    ModelSpec,
    SUPPORTED_BACKBONES,
    default_model_spec,
    describe_spec,
)


class ModelSpecTests(unittest.TestCase):
    def test_default_model_is_scoped_binary_baseline(self) -> None:
        spec = default_model_spec()

        self.assertEqual(spec.backbone, "efficientnet_b0")
        self.assertEqual(spec.num_classes, 1)
        self.assertEqual(spec.parameter_limit, "<2B")

    def test_supported_backbones_are_small_public_baselines(self) -> None:
        self.assertEqual(SUPPORTED_BACKBONES, ("efficientnet_b0", "resnet50"))

    def test_describe_spec_does_not_claim_training_started(self) -> None:
        description = describe_spec(ModelSpec(backbone="resnet50", pretrained=False))

        self.assertEqual(description["backbone"], "resnet50")
        self.assertFalse(description["pretrained"])
        self.assertFalse(description["training_started"])


if __name__ == "__main__":
    unittest.main()

