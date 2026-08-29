"""Manifest-based image loading for Reality Check.

The loader reads CSV manifests, applies one named evaluation transform, resizes
images to the future model input size, and returns normalized NumPy arrays.
It deliberately avoids model/training dependencies so the data pipeline can be
validated before any training starts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from reality_check.transforms import apply_named_transform, ensure_rgb


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "label",
    "source_dataset",
    "generator",
    "split",
    "license_notes",
}


@dataclass(frozen=True)
class ManifestRecord:
    image_path: Path
    label: int
    source_dataset: str
    generator: str
    split: str
    license_notes: str


@dataclass(frozen=True)
class ImageSample:
    image_path: Path
    label: int
    source_dataset: str
    generator: str
    split: str
    transform_name: str
    original_size: tuple[int, int]
    tensor: np.ndarray


def read_manifest(manifest_path: Path) -> list[ManifestRecord]:
    """Read a manifest CSV into typed records."""

    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")

        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Manifest {manifest_path} is missing: {missing_cols}")

        records = [
            ManifestRecord(
                image_path=Path(row["image_path"]),
                label=int(row["label"]),
                source_dataset=row["source_dataset"],
                generator=row["generator"],
                split=row["split"],
                license_notes=row["license_notes"],
            )
            for row in reader
        ]

    return records


def preprocess_for_model(
    image: Image.Image,
    image_size: int = 224,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Resize and normalize an image into a channel-first float32 array."""

    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    image = ensure_rgb(image).resize(
        (image_size, image_size), resample=Image.Resampling.BICUBIC
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (array - mean) / std
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


class ManifestImageDataset:
    """Small dataset abstraction for manifest rows and deterministic transforms."""

    def __init__(
        self,
        manifest_path: Path | str,
        transform_name: str = "clean",
        image_size: int = 224,
        limit: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform_name = transform_name
        self.image_size = image_size
        self.records = read_manifest(self.manifest_path)
        if limit is not None:
            if limit < 0:
                raise ValueError(f"limit must be non-negative, got {limit}")
            self.records = self.records[:limit]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ImageSample:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            rgb = ensure_rgb(image)
            original_size = rgb.size
            transformed = apply_named_transform(rgb, self.transform_name)
            tensor = preprocess_for_model(transformed, image_size=self.image_size)

        return ImageSample(
            image_path=record.image_path,
            label=record.label,
            source_dataset=record.source_dataset,
            generator=record.generator,
            split=record.split,
            transform_name=self.transform_name,
            original_size=original_size,
            tensor=tensor,
        )

    def __iter__(self) -> Iterator[ImageSample]:
        for index in range(len(self)):
            yield self[index]


def batch_samples(samples: list[ImageSample]) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Stack samples into arrays a future model can consume."""

    if not samples:
        raise ValueError("Cannot batch an empty sample list")

    images = np.stack([sample.tensor for sample in samples], axis=0)
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    paths = [sample.image_path for sample in samples]
    return images, labels, paths

