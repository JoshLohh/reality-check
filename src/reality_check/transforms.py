"""Image transforms for robustness training and evaluation.

Evaluation transforms are deterministic and named so clean-vs-robust metrics
can be reproduced exactly. Training code can later sample from the same
primitive operations stochastically.
"""

from __future__ import annotations

import io
import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


EVAL_TRANSFORMS: tuple[str, ...] = (
    "clean",
    "jpeg_q90",
    "jpeg_q70",
    "jpeg_q50",
    "jpeg_q30",
    "blur_s0_5",
    "blur_s1_0",
    "blur_s2_0",
    "resize_0_5",
    "resize_0_25",
    "noise_s0_02",
    "noise_s0_05",
    "noise_s0_10",
    "color_jitter_20",
    "center_crop_80",
)


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Return a detached RGB copy of an image."""

    return image.convert("RGB").copy()


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip an image through JPEG compression at the requested quality."""

    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be between 1 and 100, got {quality}")

    image = ensure_rgb(image)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur using sigma as the blur radius."""

    if sigma < 0:
        raise ValueError(f"Blur sigma must be non-negative, got {sigma}")
    return ensure_rgb(image).filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_down_up(image: Image.Image, scale: float) -> Image.Image:
    """Downscale an image and resize it back to the original dimensions."""

    if not 0 < scale <= 1:
        raise ValueError(f"Resize scale must be in (0, 1], got {scale}")

    image = ensure_rgb(image)
    width, height = image.size
    down_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    downsampled = image.resize(down_size, resample=Image.Resampling.BICUBIC)
    return downsampled.resize((width, height), resample=Image.Resampling.BICUBIC)


def gaussian_noise(image: Image.Image, sigma: float, seed: int | None = 0) -> Image.Image:
    """Add zero-mean Gaussian noise where sigma is measured on a 0-1 scale."""

    if sigma < 0:
        raise ValueError(f"Noise sigma must be non-negative, got {sigma}")

    rng = np.random.default_rng(seed)
    image = ensure_rgb(image)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    noisy = np.clip(arr + rng.normal(0.0, sigma, size=arr.shape), 0.0, 1.0)
    return Image.fromarray(np.rint(noisy * 255.0).astype(np.uint8))


def color_jitter(
    image: Image.Image,
    brightness: float = 1.2,
    contrast: float = 1.2,
    saturation: float = 1.2,
) -> Image.Image:
    """Apply deterministic brightness, contrast, and saturation adjustment."""

    image = ensure_rgb(image)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return ImageEnhance.Color(image).enhance(saturation)


def center_crop_then_resize(image: Image.Image, crop_percent: float) -> Image.Image:
    """Center crop by a percentage and resize back to the original dimensions."""

    if not 0 < crop_percent <= 100:
        raise ValueError(f"Crop percent must be in (0, 100], got {crop_percent}")

    image = ensure_rgb(image)
    width, height = image.size
    scale = crop_percent / 100.0
    crop_width = max(1, round(width * scale))
    crop_height = max(1, round(height * scale))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), resample=Image.Resampling.BICUBIC)


def _noise_transform(sigma: float) -> Callable[[Image.Image], Image.Image]:
    return lambda image: gaussian_noise(image, sigma=sigma, seed=0)


NAMED_EVAL_TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean": ensure_rgb,
    "jpeg_q90": lambda image: jpeg_compress(image, quality=90),
    "jpeg_q70": lambda image: jpeg_compress(image, quality=70),
    "jpeg_q50": lambda image: jpeg_compress(image, quality=50),
    "jpeg_q30": lambda image: jpeg_compress(image, quality=30),
    "blur_s0_5": lambda image: gaussian_blur(image, sigma=0.5),
    "blur_s1_0": lambda image: gaussian_blur(image, sigma=1.0),
    "blur_s2_0": lambda image: gaussian_blur(image, sigma=2.0),
    "resize_0_5": lambda image: resize_down_up(image, scale=0.5),
    "resize_0_25": lambda image: resize_down_up(image, scale=0.25),
    "noise_s0_02": _noise_transform(0.02),
    "noise_s0_05": _noise_transform(0.05),
    "noise_s0_10": _noise_transform(0.10),
    "color_jitter_20": lambda image: color_jitter(
        image, brightness=1.2, contrast=1.2, saturation=1.2
    ),
    "center_crop_80": lambda image: center_crop_then_resize(image, crop_percent=80),
}


def apply_named_transform(image: Image.Image, name: str) -> Image.Image:
    """Apply one deterministic evaluation transform by name."""

    try:
        transform = NAMED_EVAL_TRANSFORMS[name]
    except KeyError as exc:
        known = ", ".join(EVAL_TRANSFORMS)
        raise ValueError(f"Unknown transform '{name}'. Known transforms: {known}") from exc
    return transform(image)


def apply_random_training_transform(
    image: Image.Image,
    rng: random.Random,
    condition_names: tuple[str, ...] = EVAL_TRANSFORMS,
) -> tuple[Image.Image, str]:
    """Sample and apply one robustness transform for training."""

    name = rng.choice(condition_names)
    return apply_named_transform(image, name), name
