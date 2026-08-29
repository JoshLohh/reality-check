#!/usr/bin/env python3
"""Guarded training entrypoint.

Default behavior is dry-run planning. It prints the intended training setup and
refuses to train unless both `training.enabled: true` and `--allow-training`
are provided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.training import (
    TrainingSafetyError,
    load_yaml_config,
    run_training,
    summarize_training_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the experiment config.",
    )
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Required safety flag for actual training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training plan and exit without training.",
    )
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or mps. Auto falls back to CPU if needed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    summary = summarize_training_plan(config)
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print("Dry run requested. Training was not started.")
        return 0

    try:
        run_training(
            config,
            allow_training=args.allow_training,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
            max_epochs=args.max_epochs,
            device_name=args.device,
        )
    except TrainingSafetyError as exc:
        print(f"Training blocked: {exc}")
        print("Training was not started.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
