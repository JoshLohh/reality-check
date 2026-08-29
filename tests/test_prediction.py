from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.prediction import (
    find_image_files,
    mock_score_for_path,
    select_balanced_image_paths_from_manifest,
    write_predictions_json,
)


class PredictionTests(unittest.TestCase):
    def test_mock_score_is_deterministic_probability(self) -> None:
        path = Path("example.jpg")

        self.assertEqual(mock_score_for_path(path), mock_score_for_path(path))
        self.assertGreaterEqual(mock_score_for_path(path), 0.0)
        self.assertLessEqual(mock_score_for_path(path), 1.0)

    def test_find_image_files_recurses_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            Image.new("RGB", (4, 4)).save(root / "a.jpg")
            Image.new("RGB", (4, 4)).save(nested / "b.png")
            (root / "notes.txt").write_text("ignore me", encoding="utf-8")

            files = find_image_files(root)

        self.assertEqual([path.name for path in files], ["a.jpg", "b.png"])

    def test_write_predictions_json_uses_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preds.json"
            write_predictions_json([{"image_path": "a.jpg", "pred": 0.25}], out)
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(payload, [{"image_path": "a.jpg", "pred": 0.25}])

    def test_select_balanced_image_paths_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "image_path,label,source_dataset,generator,split,license_notes",
                        "real1.jpg,0,unit,unit,test,unit",
                        "real2.jpg,0,unit,unit,test,unit",
                        "fake1.jpg,1,unit,unit,test,unit",
                        "fake2.jpg,1,unit,unit,test,unit",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            paths = select_balanced_image_paths_from_manifest(manifest, limit=2)

        self.assertEqual([path.as_posix() for path in paths], ["real1.jpg", "fake1.jpg"])


if __name__ == "__main__":
    unittest.main()
