"""Pure image and mask operations used by the segmentation GUI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .settings import WORK_MAX

try:  # Pillow >= 9.1 moved the resampling filters onto Image.Resampling.
    NEAREST = Image.Resampling.NEAREST
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    NEAREST = Image.NEAREST
    LANCZOS = Image.LANCZOS


def load_gray_work(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image as 2D uint8 downsampled to WORK_MAX, plus its native (h, w)."""
    with Image.open(path) as img:
        img = img.convert("L")
        native_hw = (img.height, img.width)
        scale = min(WORK_MAX / max(img.width, img.height), 1.0)
        if scale < 1.0:
            size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(size, LANCZOS)
        return np.asarray(img, dtype=np.uint8).copy(), native_hw


def load_mask_work(path: Path, work_hw: tuple[int, int]) -> np.ndarray:
    """Load a mask PNG (any value > 0 is foreground) as 0/255 at working size."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"))
    binary = (arr > 0).astype(np.uint8) * 255
    resized = Image.fromarray(binary).resize((work_hw[1], work_hw[0]), NEAREST)
    return np.asarray(resized, dtype=np.uint8).copy()


def otsu_threshold(arr: np.ndarray) -> int:
    """Otsu level t such that ``arr >= t`` is the upper class."""
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    sigma_b = np.nan_to_num(sigma_b, nan=0.0, posinf=0.0)
    return int(min(255, np.argmax(sigma_b) + 1))


def bright_background(arr: np.ndarray, threshold: int) -> bool:
    """True if the image border falls in the upper (>= threshold) class."""
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return float(np.median(border)) >= threshold


def build_threshold_mask(
    base_img: np.ndarray, low: int, high: int, invert: bool
) -> Image.Image:
    """Return a 0/255 image for pixels inside (or outside, if invert) the range."""
    if invert:
        binary = (base_img < low) | (base_img > high)
    else:
        binary = (base_img >= low) & (base_img <= high)
    return Image.fromarray(binary.astype(np.uint8) * 255)


def apply_contrast(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Window the 0-255 display values between lo and hi."""
    if lo <= 0 and hi >= 255:
        return arr
    if hi <= lo:
        hi = lo + 1
    scaled = (arr.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def mask_diff(before: np.ndarray, after: np.ndarray) -> tuple[int, int, list[int] | None]:
    """Return added/removed pixel counts and the changed bounding box."""
    changed = before != after
    if not changed.any():
        return 0, 0, None
    added = int(((before == 0) & (after != 0)).sum())
    removed = int(((before != 0) & (after == 0)).sum())
    rows, cols = np.where(changed)
    bbox = [
        int(rows.min()),
        int(cols.min()),
        int(rows.max()) + 1,
        int(cols.max()) + 1,
    ]
    return added, removed, bbox
