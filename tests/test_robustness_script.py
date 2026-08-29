from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RobustnessScriptTests(unittest.TestCase):
    def test_help_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/evaluate_robustness.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--conditions", result.stdout)

    def test_mock_robustness_run_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "robustness"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_robustness.py",
                    "--manifest",
                    "data/manifests/test_clean.csv",
                    "--out-dir",
                    str(out_dir),
                    "--mock",
                    "--limit",
                    "20",
                    "--conditions",
                    "clean",
                    "jpeg_q30",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out_dir / "robustness_summary.csv").is_file())
            self.assertTrue((out_dir / "robustness_summary.md").is_file())
            self.assertTrue((out_dir / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()

