"""Baseline model definition for AIGC image detection.

This module defines the planned classifier architecture but does not train it.
Heavy ML imports are lazy so the rest of the data pipeline can be tested
without installing PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_BACKBONES = ("efficientnet_b0", "resnet50")


@dataclass(frozen=True)
class ModelSpec:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.2
    num_classes: int = 1
    output_activation: str = "sigmoid_probability_at_inference"
    parameter_limit: str = "<2B"


def default_model_spec() -> ModelSpec:
    """Return the first planned hackathon baseline."""

    return ModelSpec()


def _require_torchvision() -> tuple[Any, Any]:
    try:
        import torch.nn as nn
        from torchvision import models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Building the model requires PyTorch and torchvision. "
            "Install the full requirements when you are ready for model work: "
            "`.venv/bin/python -m pip install -r requirements.txt`. "
            "This does not start training by itself."
        ) from exc
    return nn, models


def build_model(spec: ModelSpec | None = None) -> Any:
    """Build the binary classifier architecture described by ``spec``.

    The model returns one raw logit per image. Training code should use
    BCEWithLogitsLoss. Inference code should apply sigmoid to convert logits
    into the required AIGC probability score.
    """

    spec = spec or default_model_spec()
    if spec.backbone not in SUPPORTED_BACKBONES:
        supported = ", ".join(SUPPORTED_BACKBONES)
        raise ValueError(f"Unsupported backbone '{spec.backbone}'. Use one of: {supported}")
    if spec.num_classes != 1:
        raise ValueError("This hackathon prototype is scoped to binary detection.")

    nn, models = _require_torchvision()

    if spec.backbone == "efficientnet_b0":
        weights = (
            models.EfficientNet_B0_Weights.DEFAULT if spec.pretrained else None
        )
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=spec.dropout),
            nn.Linear(in_features, spec.num_classes),
        )
        return model

    weights = models.ResNet50_Weights.DEFAULT if spec.pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=spec.dropout),
        nn.Linear(in_features, spec.num_classes),
    )
    return model


def count_parameters(model: Any) -> int:
    """Count trainable and frozen parameters for a PyTorch-like model."""

    return sum(parameter.numel() for parameter in model.parameters())


def freeze_backbone(model: Any, backbone: str) -> None:
    """Freeze all feature layers while leaving the binary classifier trainable."""

    if backbone == "efficientnet_b0":
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        return

    if backbone == "resnet50":
        for name, parameter in model.named_parameters():
            if not name.startswith("fc."):
                parameter.requires_grad = False
        return

    raise ValueError(f"Unsupported backbone for freezing: {backbone}")


def count_trainable_parameters(model: Any) -> int:
    """Count only parameters that will be updated by the optimizer."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def describe_spec(spec: ModelSpec | None = None) -> dict[str, str | int | float | bool]:
    """Return a serializable description of the planned architecture."""

    spec = spec or default_model_spec()
    return {
        "backbone": spec.backbone,
        "pretrained": spec.pretrained,
        "dropout": spec.dropout,
        "num_output_logits": spec.num_classes,
        "output_activation": spec.output_activation,
        "parameter_limit": spec.parameter_limit,
        "training_started": False,
    }
