"""Prediction helpers for the required hackathon JSON output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from reality_check.dataset import preprocess_for_model, read_manifest
from reality_check.transforms import apply_named_transform, ensure_rgb


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_files(input_dir: Path) -> list[Path]:
    """Find supported image files recursively under an input directory."""

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_balanced_image_paths_from_manifest(
    manifest_path: Path, limit: int | None = None
) -> list[Path]:
    """Select image paths from a labelled manifest, balanced when limited."""

    records = read_manifest(manifest_path)
    if limit is None or limit <= 0 or limit >= len(records):
        return [record.image_path for record in records]

    by_label: dict[int, list[Path]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record.image_path)

    if {0, 1}.issubset(by_label):
        real_target = limit // 2
        fake_target = limit - real_target
        selected = by_label[0][:real_target] + by_label[1][:fake_target]
        if len(selected) < limit:
            seen = set(selected)
            selected.extend(
                record.image_path for record in records if record.image_path not in seen
            )
        return selected[:limit]

    return [record.image_path for record in records[:limit]]


def mock_score_for_path(path: Path) -> float:
    """Return a deterministic fake score for interface testing."""

    digest = hashlib.sha256(path.as_posix().encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return round(integer / float(2**64 - 1), 6)


def write_predictions_json(predictions: list[dict[str, Any]], output_path: Path) -> None:
    """Write predictions using the required `image_path` and `pred` fields."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2)
        file.write("\n")


def load_predictions_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Predictions JSON must contain a list: {path}")
    for index, row in enumerate(data):
        if not isinstance(row, dict) or "image_path" not in row or "pred" not in row:
            raise ValueError(f"Prediction row {index} must contain image_path and pred")
    return data


def predict_mock(image_paths: list[Path]) -> list[dict[str, float | str]]:
    return [
        {"image_path": path.as_posix(), "pred": mock_score_for_path(path)}
        for path in image_paths
    ]


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Real checkpoint inference requires PyTorch. Install it with "
            "`.venv/bin/python -m pip install torch torchvision`."
        ) from exc
    return torch


def predict_with_checkpoint(
    image_paths: list[Path],
    checkpoint_path: Path,
    transform_name: str,
    image_size: int,
    batch_size: int,
) -> list[dict[str, float | str]]:
    """Run real checkpoint inference.

    This path is intentionally separate from mock mode. It assumes a future
    training loop will save a checkpoint containing a `model_state_dict`.
    """

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    torch = _require_torch()
    from reality_check.model import ModelSpec, build_model

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = build_model(ModelSpec(pretrained=False))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    predictions: list[dict[str, float | str]] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            arrays = []
            kept_paths = []
            for path in batch_paths:
                with Image.open(path) as image:
                    rgb = ensure_rgb(image)
                    transformed = apply_named_transform(rgb, transform_name)
                    arrays.append(preprocess_for_model(transformed, image_size=image_size))
                    kept_paths.append(path)

            inputs = torch.from_numpy(np.stack(arrays, axis=0)).to(device)
            logits = model(inputs).view(-1)
            scores = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            predictions.extend(
                {"image_path": path.as_posix(), "pred": round(float(score), 6)}
                for path, score in zip(kept_paths, scores)
            )

    return predictions
