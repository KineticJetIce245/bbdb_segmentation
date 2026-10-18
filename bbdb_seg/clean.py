"""Pure mask clean-up operations: background exclusion and edge snapping.

Thresholding picks up a thin rim of blurred / JPEG-halo pixels at tissue edges
and in narrow sulcal gaps. Two tools deal with that:

* ``background_mask`` estimates the slide background level from the image
  border and marks every pixel close to it (plus a small margin for the blurry
  rim). Background is never cortex.
* ``snap_to_edges`` is marker-controlled watershed on the image gradient:
  pixels deep inside the mask are "sure cortex", pixels well outside it (or on
  background) are "sure not cortex", and the narrow band in between is assigned
  by flooding the gradient. The boundary therefore settles on the strongest
  nearby edge and never crosses it ("dilate past the edge, erode back to it").

All functions take/return plain numpy arrays (uint8 gray image, bool masks).
Edge snapping uses scikit-image's watershed when it is installed (exact
boundary placement); otherwise it falls back to scipy's ``watershed_ift``,
which can leave the boundary 1-3 px off.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .settings import BG_MARGIN, BG_TOLERANCE, CLEAN_BAND, CLEAN_MIN_ISLAND

try:  # exact watershed; optional dependency
    from skimage.segmentation import watershed as _sk_watershed
except ImportError:  # pragma: no cover - depends on the environment
    _sk_watershed = None


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def background_mask(
    gray: np.ndarray,
    tolerance: int | None = BG_TOLERANCE,
    margin: int = BG_MARGIN,
) -> np.ndarray:
    """Bool mask of pixels within ``tolerance`` of the border (background) level.

    The background level is the median of the image border, so white, grey or
    dark backgrounds all work. ``tolerance`` widens automatically when the
    border is noisy. ``margin`` grows the result by that many pixels to absorb
    the blurry tissue rim.
    """
    gray = np.asarray(gray)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]]).astype(np.float32)
    level = float(np.median(border))
    spread = float(np.median(np.abs(border - level)))  # robust border noise
    tol = max(float(tolerance or 0), 3.0 * spread)
    bg = np.abs(gray.astype(np.float32) - level) <= tol
    if margin > 0:
        bg = ndimage.binary_dilation(bg, structure=_disk(1), iterations=margin)
    return bg


def remove_small_islands(mask: np.ndarray, min_size: int = CLEAN_MIN_ISLAND) -> np.ndarray:
    """Drop connected components with fewer than ``min_size`` pixels."""
    mask = np.asarray(mask, dtype=bool)
    if min_size <= 1 or not mask.any():
        return mask
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def _ridge(mask: np.ndarray) -> np.ndarray:
    """Approximate medial axis: local maxima of the distance transform."""
    dist = ndimage.distance_transform_edt(mask)
    local_max = dist >= ndimage.maximum_filter(dist, size=3)
    return mask & local_max & (dist >= 1)


def snap_to_edges(
    gray: np.ndarray,
    mask: np.ndarray,
    band: int = CLEAN_BAND,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Move the mask boundary (within ``band`` px) onto the strongest image edge."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or band <= 0:
        return mask
    gray_f = np.asarray(gray, dtype=np.float32)
    grad = ndimage.gaussian_gradient_magnitude(gray_f, sigma=1.0)

    structure = _disk(band)
    sure_fg = ndimage.binary_erosion(mask, structure=structure)
    sure_fg |= _ridge(mask)  # thin cortex that erosion would delete keeps a seed
    sure_bg = ~ndimage.binary_dilation(mask, structure=structure)
    if background is not None:
        sure_fg &= ~background
        sure_bg |= background

    markers = np.zeros(mask.shape, dtype=np.int16)
    markers[sure_bg] = 1
    markers[sure_fg] = 2
    if not (markers == 2).any():
        return mask & ~background if background is not None else mask
    if _sk_watershed is not None:
        labels = _sk_watershed(grad, markers)
    else:
        peak = float(grad.max()) or 1.0
        labels = ndimage.watershed_ift(np.round(grad / peak * 65535).astype(np.uint16), markers)
    return labels == 2


def clean_mask(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    band: int = CLEAN_BAND,
    min_island: int = CLEAN_MIN_ISLAND,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Background exclusion + edge snapping + small-island removal."""
    if background is None:
        background = background_mask(gray)
    out = snap_to_edges(gray, mask, band=band, background=background)
    out &= ~background
    return remove_small_islands(out, min_island)
