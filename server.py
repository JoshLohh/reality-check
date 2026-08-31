"""Serve the Reality Check upload UI and its local PyTorch inference API.

Add one compatible .pt checkpoint anywhere inside checkpoints/, then run:
    .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from reality_check.dataset import preprocess_for_model
from reality_check.model import ModelSpec, build_model
from reality_check.transforms import ensure_rgb

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def find_checkpoint() -> Path:
    """Find the only model file the user needs to add to this project."""
    checkpoints = sorted((ROOT / "checkpoints").rglob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            "No .pt model found. Add your checkpoint under checkpoints/, for example "
            "checkpoints/best.pt."
        )
    if len(checkpoints) > 1:
        preferred = ROOT / "checkpoints" / "best.pt"
        if preferred.is_file():
            return preferred
        raise FileNotFoundError(
            "Found more than one .pt model. Keep only one, or name the checkpoint "
            "you want to use checkpoints/best.pt."
        )
    return checkpoints[0]


def load_model() -> tuple[Any, Any, Path]:
    """Load a checkpoint trained with this repository's EfficientNet-B0 model."""
    import torch

    checkpoint_path = find_checkpoint()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("The .pt file must contain a model state dictionary.")
    saved_spec = checkpoint.get("model_spec", {})
    if not isinstance(saved_spec, dict):
        saved_spec = {}
    # Training checkpoints from this project record their architecture. Falling
    # back to the baseline keeps plain state-dictionary checkpoints supported.
    model = build_model(
        ModelSpec(
            backbone=str(saved_spec.get("backbone", "efficientnet_b0")),
            pretrained=False,
            dropout=float(saved_spec.get("dropout", 0.2)),
            num_classes=int(saved_spec.get("num_output_logits", 1)),
        )
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device, checkpoint_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Do not prevent the UI from opening when a teammate has not yet copied the
    # checkpoint; /api/predict will return the actionable error instead.
    app.state.model = None
    app.state.device = None
    app.state.checkpoint_path = None
    app.state.load_error = None
    try:
        app.state.model, app.state.device, app.state.checkpoint_path = load_model()
    except (FileNotFoundError, RuntimeError, ValueError, ModuleNotFoundError) as exc:
        app.state.load_error = str(exc)
    yield


app = FastAPI(title="Reality Check", lifespan=lifespan)


@app.get("/")
def homepage() -> FileResponse:
    return FileResponse(ROOT / "ui" / "image-upload.html")


@app.get("/api/health")
def health() -> dict[str, str | bool | None]:
    return {
        "ready": app.state.model is not None,
        "checkpoint": app.state.checkpoint_path.name if app.state.checkpoint_path else None,
        "message": app.state.load_error,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | str]:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail=app.state.load_error)
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Use a JPG, PNG, WebP, or BMP image.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened_image:
            image = ensure_rgb(opened_image)
            pixels = preprocess_for_model(image, image_size=224)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="That file could not be read as an image.") from exc

    import torch

    with torch.no_grad():
        inputs = torch.from_numpy(np.stack([pixels])).to(app.state.device)
        probability = float(torch.sigmoid(app.state.model(inputs).view(-1))[0].cpu())

    return {
        "ai_probability": round(probability, 6),
        "label": "Likely AI-generated" if probability >= 0.5 else "Likely authentic",
    }
