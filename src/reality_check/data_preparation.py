"""Shared metadata helpers for downloading and preparing public datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MANIFEST_COLUMNS = (
    "image_path",
    "label",
    "source_dataset",
    "generator",
    "split",
    "license_notes",
)


@dataclass(frozen=True)
class SidLabel:
    """The binary label policy for one SID_Set example."""

    binary_label: int
    class_name: str
    generator: str


def map_sid_label(raw_label: int) -> SidLabel | None:
    """Map SID_Set labels to the binary task, excluding tampered images.

    SID_Set defines 0 as real, 1 as fully synthetic, and 2 as tampered. The
    hackathon baseline detects complete AIGC images, so label 2 intentionally
    returns ``None`` rather than being silently treated as synthetic.
    """

    if raw_label == 0:
        return SidLabel(0, "real", "openimages_v7")
    if raw_label == 1:
        return SidLabel(1, "full_synthetic", "unknown_full_synthetic")
    if raw_label == 2:
        return None
    raise ValueError(f"Unexpected SID_Set label {raw_label}; expected 0, 1, or 2")


def manifest_split_for_sid_split(source_split: str) -> str:
    """Map the Hub split names into this repository's manifest convention."""

    mapping = {
        "train": "train",
        "val": "val",
        "validation": "val",
        "test": "test_clean",
    }
    try:
        return mapping[source_split]
    except KeyError as exc:
        known = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unknown SID_Set split '{source_split}'. Expected one of: {known}"
        ) from exc


def sid_image_path(root: Path, source_split: str, class_name: str, index: int) -> Path:
    """Return a deterministic, collision-resistant local PNG path."""

    return root / source_split / class_name / f"sid_{source_split}_{index:07d}.png"
