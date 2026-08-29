"""Guarded training skeleton for Reality Check.

This module documents and wires the future training path without making
training easy to start by accident. Actual optimization only proceeds when both
the config and command line explicitly allow it.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from reality_check.dataset import ManifestRecord, preprocess_for_model
from reality_check.evaluation import binary_metrics, roc_auc_score
from reality_check.dataset import read_manifest
from reality_check.model import (
    ModelSpec,
    build_model,
    count_parameters,
    count_trainable_parameters,
    describe_spec,
    freeze_backbone,
)
from reality_check.transforms import apply_random_training_transform, ensure_rgb


class TrainingSafetyError(RuntimeError):
    """Raised when a training run is intentionally blocked."""


@dataclass(frozen=True)
class TrainingPlan:
    train_manifest: Path
    val_manifest: Path
    image_size: int
    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    class_balance: str
    early_stopping_metric: str
    model: ModelSpec
    freeze_backbone: bool
    checkpoint_dir: Path
    num_workers: int


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config while keeping PyYAML as a runtime dependency."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading training configs requires PyYAML. Install it with "
            "`.venv/bin/python -m pip install PyYAML`, or install all planned "
            "requirements when ready."
        ) from exc

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def build_training_plan(config: dict[str, Any]) -> TrainingPlan:
    """Extract the planned training settings from the project config."""

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    manifests = data_config.get("manifests", {})
    classifier = model_config.get("classifier", {})

    return TrainingPlan(
        train_manifest=Path(manifests["train"]),
        val_manifest=Path(manifests["val"]),
        image_size=int(data_config.get("input_size", 224)),
        batch_size=int(data_config.get("batch_size", 32)),
        max_epochs=int(training_config.get("max_epochs", 10)),
        learning_rate=float(training_config.get("learning_rate", 0.0001)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        class_balance=str(training_config.get("class_balance", "weighted_sampler")),
        early_stopping_metric=str(
            training_config.get("early_stopping_metric", "val_roc_auc")
        ),
        model=ModelSpec(
            backbone=str(model_config.get("backbone", "efficientnet_b0")),
            pretrained=bool(model_config.get("pretrained", True)),
            dropout=float(classifier.get("dropout", 0.2)),
        ),
        freeze_backbone=bool(training_config.get("freeze_backbone", True)),
        checkpoint_dir=Path(training_config.get("checkpoint_dir", "checkpoints")),
        num_workers=int(data_config.get("num_workers", 0)),
    )


def enforce_training_safety(config: dict[str, Any], allow_training: bool) -> None:
    """Block accidental training unless both explicit switches are enabled."""

    config_enabled = bool(config.get("training", {}).get("enabled", False))
    if not config_enabled:
        raise TrainingSafetyError(
            "Training is disabled in the config. Set `training.enabled: true` "
            "only when you intentionally want to train."
        )
    if not allow_training:
        raise TrainingSafetyError(
            "Training also requires the command-line flag `--allow-training`."
        )


def summarize_training_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable summary without importing PyTorch or training."""

    plan = build_training_plan(config)
    train_records = read_manifest(plan.train_manifest)
    val_records = read_manifest(plan.val_manifest)

    return {
        "training_started": False,
        "train_manifest": plan.train_manifest.as_posix(),
        "train_rows": len(train_records),
        "val_manifest": plan.val_manifest.as_posix(),
        "val_rows": len(val_records),
        "image_size": plan.image_size,
        "batch_size": plan.batch_size,
        "max_epochs": plan.max_epochs,
        "learning_rate": plan.learning_rate,
        "weight_decay": plan.weight_decay,
        "class_balance": plan.class_balance,
        "early_stopping_metric": plan.early_stopping_metric,
        "freeze_backbone": plan.freeze_backbone,
        "checkpoint_dir": plan.checkpoint_dir.as_posix(),
        "model": describe_spec(plan.model),
    }


def require_training_dependencies() -> tuple[Any, Any]:
    """Import heavy training dependencies only after safety checks pass."""

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Training requires PyTorch. Install the full requirements when you "
            "are ready: `.venv/bin/python -m pip install -r requirements.txt`."
        ) from exc

    return torch, (DataLoader, Dataset)


class TorchManifestDataset:
    """PyTorch-compatible dataset backed by manifest records."""

    def __init__(
        self,
        records: list[ManifestRecord],
        image_size: int,
        training: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.training = training
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch, _ = require_training_dependencies()
        record = self.records[index]
        with Image.open(record.image_path) as image:
            rgb = ensure_rgb(image)
            if self.training:
                rng = random.Random(self.seed + index + int(time.time() // 60))
                transformed, _ = apply_random_training_transform(rgb, rng)
            else:
                transformed = rgb
            tensor = preprocess_for_model(transformed, image_size=self.image_size)

        image_tensor = torch.from_numpy(tensor)
        label_tensor = torch.tensor(float(record.label), dtype=torch.float32)
        return image_tensor, label_tensor


def select_records(records: list[ManifestRecord], limit: int | None) -> list[ManifestRecord]:
    """Return a balanced subset when a limit is provided."""

    if limit is None or limit <= 0 or limit >= len(records):
        return records

    by_label: dict[int, list[ManifestRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)

    if {0, 1}.issubset(by_label):
        half = limit // 2
        selected = by_label[0][:half] + by_label[1][: limit - half]
        if len(selected) < limit:
            seen = {record.image_path for record in selected}
            selected.extend(record for record in records if record.image_path not in seen)
        return selected[:limit]

    return records[:limit]


def resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def run_one_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


def validate(model: Any, loader: Any, criterion: Any, device: Any) -> dict[str, float]:
    torch, _ = require_training_dependencies()
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels_all: list[int] = []
    scores_all: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).view(-1)
            loss = criterion(logits, labels)
            scores = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            label_values = labels.detach().cpu().numpy().astype(int).tolist()
            batch_size = labels.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            labels_all.extend(label_values)
            scores_all.extend(float(score) for score in scores)

    auc = roc_auc_score(labels_all, scores_all)
    accuracy, precision, recall, f1 = binary_metrics(labels_all, scores_all, threshold=0.5)
    return {
        "val_loss": total_loss / max(total_examples, 1),
        "val_roc_auc": auc,
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
    }


def run_training(
    config: dict[str, Any],
    allow_training: bool,
    limit_train: int | None = None,
    limit_val: int | None = None,
    max_epochs: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run a guarded training job once both safety switches are enabled.

    This is intentionally sized by CLI limits for laptop smoke runs.
    """

    enforce_training_safety(config, allow_training)
    torch, (DataLoader, _) = require_training_dependencies()

    seed = int(config.get("project", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    plan = build_training_plan(config)
    epochs = max_epochs if max_epochs is not None else plan.max_epochs
    train_records = select_records(read_manifest(plan.train_manifest), limit_train)
    val_records = select_records(read_manifest(plan.val_manifest), limit_val)
    if not train_records or not val_records:
        raise ValueError("Training and validation records must both be non-empty")

    os.environ.setdefault("TORCH_HOME", str(plan.checkpoint_dir / "torch_cache"))

    train_dataset = TorchManifestDataset(
        train_records, image_size=plan.image_size, training=True, seed=seed
    )
    val_dataset = TorchManifestDataset(
        val_records, image_size=plan.image_size, training=False, seed=seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=plan.batch_size,
        shuffle=True,
        num_workers=plan.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=plan.batch_size,
        shuffle=False,
        num_workers=plan.num_workers,
    )

    device = resolve_device(torch, device_name)
    model = build_model(plan.model)
    if plan.freeze_backbone:
        freeze_backbone(model, plan.model.backbone)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=plan.learning_rate,
        weight_decay=plan.weight_decay,
    )

    plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    history: list[dict[str, float | int]] = []
    started_at = time.time()

    print(
        json.dumps(
            {
                "training_started": True,
                "device": str(device),
                "train_rows": len(train_records),
                "val_rows": len(val_records),
                "epochs": epochs,
                "batch_size": plan.batch_size,
                "freeze_backbone": plan.freeze_backbone,
                "parameter_count": count_parameters(model),
                "trainable_parameter_count": count_trainable_parameters(model),
            },
            indent=2,
        )
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = run_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = validate(model, val_loader, criterion, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "seconds": round(time.time() - epoch_start, 2),
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if metrics["val_roc_auc"] > best_auc:
            best_auc = metrics["val_roc_auc"]
            checkpoint_path = plan.checkpoint_dir / "best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_spec": describe_spec(plan.model),
                    "epoch": epoch,
                    "val_roc_auc": best_auc,
                    "train_rows": len(train_records),
                    "val_rows": len(val_records),
                    "image_size": plan.image_size,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to {checkpoint_path}")

    summary = {
        "training_started": True,
        "training_finished": True,
        "seconds_total": round(time.time() - started_at, 2),
        "best_val_roc_auc": best_auc,
        "checkpoint": (plan.checkpoint_dir / "best.pt").as_posix(),
        "history": history,
    }
    history_path = plan.checkpoint_dir / "training_history.json"
    history_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote training history to {history_path}")
    return summary
