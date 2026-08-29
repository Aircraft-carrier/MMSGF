"""Wire-format validation and image serialization for online evaluation."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@dataclass(frozen=True, slots=True)
class OnlineObservation:
    step: int
    images: dict[str, np.ndarray]
    state: np.ndarray


def decode_jpeg(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def encode_jpeg(image: np.ndarray, *, quality: int = 90) -> str:
    array = np.asarray(image, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_observation(payload: dict[str, Any]) -> OnlineObservation:
    step = int(payload["step"])
    state = np.asarray(payload["state"], dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("observation state must contain 16 finite values")
    encoded = payload.get("images")
    if not isinstance(encoded, dict):
        raise ValueError("observation images must be an object")
    missing = [key for key in CAMERA_KEYS if key not in encoded]
    if missing:
        raise ValueError(f"observation is missing cameras: {missing}")
    images = {key: decode_jpeg(encoded[key]) for key in CAMERA_KEYS}
    shape = images[CAMERA_KEYS[0]].shape
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"camera images must be HWC RGB, got {shape}")
    if any(image.shape != shape for image in images.values()):
        raise ValueError("all camera images must have the same shape")
    return OnlineObservation(step=step, images=images, state=state)
