"""Shared flip/rotation helpers."""
from __future__ import annotations

import math

import numpy as np


def decode_flip(value: object) -> tuple[bool, bool]:
    """Parse a flip string into (left_right_flip, up_down_flip)."""
    text = str(value).strip().upper().replace(" ", "")
    if text in {"", "NAN", "NONE", "0", "FALSE"}:
        return False, False
    return "LR" in text, "UP" in text or "UD" in text


def encode_flip(lr_flip: bool, up_flip: bool) -> str:
    if lr_flip and up_flip:
        return "LR+UP"
    if lr_flip:
        return "LR"
    if up_flip:
        return "UP"
    return ""


def normalize_flip(value: object) -> str:
    return encode_flip(*decode_flip(value))


def normalize_rotation(value: object) -> int:
    """Normalize a rotation to a count of 90 deg CCW steps (0-3, or degrees)."""
    text = str(value).strip()
    if text == "" or text.upper() in {"NAN", "NONE"}:
        return 0
    try:
        rotation = float(text)
    except ValueError:
        return 0
    if not math.isfinite(rotation):
        return 0
    rotation = int(round(rotation))
    if abs(rotation) > 3 and rotation % 90 == 0:
        rotation //= 90
    return rotation % 4


def apply_transform(data: np.ndarray, flip_value: object, rotation_steps: object) -> np.ndarray:
    """Apply LR flip, then UP flip, then rotation to a 2D array."""
    lr_flip, up_flip = decode_flip(flip_value)
    transformed = data
    if lr_flip:
        transformed = np.fliplr(transformed)
    if up_flip:
        transformed = np.flipud(transformed)
    rotation_steps = normalize_rotation(rotation_steps)
    if rotation_steps:
        transformed = np.rot90(transformed, k=rotation_steps)
    return np.ascontiguousarray(transformed)
