from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.data_preparation import (
    manifest_split_for_sid_split,
    map_sid_label,
    sid_image_path,
)


class SidSetPreparationTests(unittest.TestCase):
    def test_binary_label_mapping_excludes_tampered_examples(self) -> None:
        self.assertEqual(map_sid_label(0).binary_label, 0)
        self.assertEqual(map_sid_label(1).binary_label, 1)
        self.assertIsNone(map_sid_label(2))

    def test_unexpected_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected SID_Set label"):
            map_sid_label(9)

    def test_split_and_path_conventions_are_reproducible(self) -> None:
        self.assertEqual(manifest_split_for_sid_split("test"), "test_clean")
        path = sid_image_path(Path("data/raw/sid_set"), "train", "real", 12)
        self.assertEqual(path.as_posix(), "data/raw/sid_set/train/real/sid_train_0000012.png")


if __name__ == "__main__":
    unittest.main()
