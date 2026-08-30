from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.dataset import ManifestImageDataset, batch_samples


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.image_path = self.root / "example.jpg"
        self.manifest_path = self.root / "manifest.csv"

        image = Image.fromarray(np.full((16, 20, 3), 128, dtype=np.uint8))
        image.save(self.image_path)

        with self.manifest_path.open("w", newline="", encoding="utf-8") as file:
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
            writer.writerow(
                {
                    "image_path": self.image_path.as_posix(),
                    "label": "1",
                    "source_dataset": "unit_test",
                    "generator": "synthetic_fixture",
                    "split": "sample",
                    "license_notes": "local test fixture",
                }
            )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_dataset_loads_manifest_row_as_model_ready_array(self) -> None:
        dataset = ManifestImageDataset(
            self.manifest_path, transform_name="jpeg_q70", image_size=32
        )
        sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(sample.label, 1)
        self.assertEqual(sample.original_size, (20, 16))
        self.assertEqual(sample.tensor.shape, (3, 32, 32))
        self.assertEqual(sample.tensor.dtype, np.float32)

    def test_batch_samples_stacks_images_labels_and_paths(self) -> None:
        dataset = ManifestImageDataset(self.manifest_path, image_size=32)
        images, labels, paths = batch_samples([dataset[0]])

        self.assertEqual(images.shape, (1, 3, 32, 32))
        np.testing.assert_array_equal(labels, np.array([1]))
        self.assertEqual(paths, [self.image_path])


if __name__ == "__main__":
    unittest.main()

