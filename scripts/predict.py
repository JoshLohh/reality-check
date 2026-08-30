#!/usr/bin/env python3
"""Write AIGC prediction JSON for an image directory.

Use `--mock` before a trained checkpoint exists. Mock mode validates the
required submission interface without training or loading a model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.prediction import (
    find_image_files,
    predict_mock,
    predict_with_checkpoint,
    select_balanced_image_paths_from_manifest,
    write_predictions_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=Path)
    input_group.add_argument(
        "--manifest",
        type=Path,
        help="Labelled manifest to select image paths from. Useful with --limit.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--transform", default="clean")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest is not None:
        image_paths = select_balanced_image_paths_from_manifest(args.manifest, args.limit)
    else:
        if not args.input_dir.is_dir():
            raise SystemExit(f"Input directory not found: {args.input_dir}")
        image_paths = find_image_files(args.input_dir)
    if args.limit is not None and args.manifest is None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit("No supported images found")

    if args.mock:
        predictions = predict_mock(image_paths)
    else:
        if args.checkpoint is None:
            raise SystemExit("Real inference requires --checkpoint, or use --mock.")
        predictions = predict_with_checkpoint(
            image_paths=image_paths,
            checkpoint_path=args.checkpoint,
            transform_name=args.transform,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )

    write_predictions_json(predictions, args.output)
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
