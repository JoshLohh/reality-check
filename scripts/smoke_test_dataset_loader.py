#!/usr/bin/env python3
"""Smoke test the manifest image loader.

This validates image loading, labels, deterministic transforms, resizing, and
normalization. It does not build, train, or evaluate a model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reality_check.dataset import ManifestImageDataset, batch_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/cifake_sample.csv"),
    )
    parser.add_argument("--transform", default="clean")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def balanced_indices(dataset: ManifestImageDataset, limit: int) -> list[int]:
    if limit <= 0 or limit >= len(dataset):
        return list(range(len(dataset)))

    by_label: dict[int, list[int]] = {}
    for index, record in enumerate(dataset.records):
        by_label.setdefault(record.label, []).append(index)

    if {0, 1}.issubset(by_label):
        real_target = limit // 2
        fake_target = limit - real_target
        selected = by_label[0][:real_target] + by_label[1][:fake_target]
        if len(selected) < limit:
            seen = set(selected)
            selected.extend(index for index in range(len(dataset)) if index not in seen)
        return selected[:limit]

    return list(range(limit))


def main() -> None:
    args = parse_args()
    dataset = ManifestImageDataset(
        manifest_path=args.manifest,
        transform_name=args.transform,
        image_size=args.image_size,
    )
    samples = [dataset[index] for index in balanced_indices(dataset, args.limit)]
    images, labels, paths = batch_samples(samples)

    print(f"Manifest: {args.manifest}")
    print(f"Transform: {args.transform}")
    print(f"Samples loaded: {len(samples)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Label counts: real={(labels == 0).sum()}, fake={(labels == 1).sum()}")
    print(f"First path: {paths[0]}")
    print(f"First original size: {samples[0].original_size}")
    print(f"Value range after normalization: {images.min():.3f} to {images.max():.3f}")


if __name__ == "__main__":
    main()
