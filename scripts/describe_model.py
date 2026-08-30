#!/usr/bin/env python3
"""Print the planned baseline model architecture.

By default this only describes the architecture config and does not instantiate
the model, download weights, or train anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.model import (
    ModelSpec,
    build_model,
    count_parameters,
    describe_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Instantiate without downloading pretrained weights.",
    )
    parser.add_argument(
        "--instantiate",
        action="store_true",
        help="Build the PyTorch model to count parameters. Does not train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ModelSpec(
        backbone=args.backbone,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    )
    description = describe_spec(spec)

    if args.instantiate:
        model = build_model(spec)
        description["parameter_count"] = count_parameters(model)

    print(json.dumps(description, indent=2))


if __name__ == "__main__":
    main()
