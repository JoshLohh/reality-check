from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.transforms import EVAL_TRANSFORMS, apply_named_transform


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        x = np.linspace(0, 255, 32, dtype=np.uint8)
        red = np.tile(x, (32, 1))
        green = red.T
        blue = np.full((32, 32), 128, dtype=np.uint8)
        self.image = Image.fromarray(np.stack([red, green, blue], axis=-1))

    def test_all_eval_transforms_preserve_size_and_mode(self) -> None:
        for condition in EVAL_TRANSFORMS:
            with self.subTest(condition=condition):
                output = apply_named_transform(self.image, condition)
                self.assertEqual(output.size, self.image.size)
                self.assertEqual(output.mode, "RGB")

    def test_noise_transform_is_deterministic_for_evaluation(self) -> None:
        first = np.asarray(apply_named_transform(self.image, "noise_s0_10"))
        second = np.asarray(apply_named_transform(self.image, "noise_s0_10"))
        np.testing.assert_array_equal(first, second)

    def test_unknown_transform_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown transform"):
            apply_named_transform(self.image, "not_a_real_condition")


if __name__ == "__main__":
    unittest.main()
