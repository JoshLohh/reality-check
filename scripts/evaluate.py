#!/usr/bin/env python3
"""Evaluate prediction JSON against a labelled manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.evaluation import evaluate_predictions, write_evaluation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/reports/evaluation"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_predictions(
        manifest_path=args.manifest,
        predictions_path=args.predictions,
        threshold=args.threshold,
    )
    write_evaluation_outputs(result, args.out_dir)
    print(json.dumps(result.__dict__, indent=2))
    print(f"Wrote evaluation outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

