"""Rectangular ROI model and base/display coordinate mapping."""
from __future__ import annotations

from dataclasses import dataclass

from common import decode_flip


@dataclass
class Roi:
    """A rectangular selection in base-image pixel coordinates (r1/c1 exclusive)."""

    roi_id: int
    r0: int
    c0: int
    r1: int
    c1: int
    threshold_min: int
    threshold_max: int
    invert: bool

    @property
    def bbox(self) -> list[int]:
        return [self.r0, self.c0, self.r1, self.c1]


def base_to_display(
    row: int, col: int, height: int, width: int, flip: str, rotation: int
) -> tuple[int, int]:
    """Map a base-image pixel to its display coordinate for the current transform."""
    lr_flip, up_flip = decode_flip(flip)
    if lr_flip:
        col = width - 1 - col
    if up_flip:
        row = height - 1 - row
    if rotation == 1:
        return width - 1 - col, row
    if rotation == 2:
        return height - 1 - row, width - 1 - col
    if rotation == 3:
        return col, height - 1 - row
    return row, col


def roi_screen_rect(
    roi: Roi,
    height: int,
    width: int,
    flip: str,
    rotation: int,
    scale: float,
    offset: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Return the ROI rectangle in canvas coordinates."""
    corners = [
        (roi.r0, roi.c0),
        (roi.r0, roi.c1 - 1),
        (roi.r1 - 1, roi.c0),
        (roi.r1 - 1, roi.c1 - 1),
    ]
    display = [base_to_display(row, col, height, width, flip, rotation) for row, col in corners]
    rows = [point[0] for point in display]
    cols = [point[1] for point in display]
    offset_x, offset_y = offset
    return (
        offset_x + min(cols) * scale,
        offset_y + min(rows) * scale,
        offset_x + (max(cols) + 1) * scale,
        offset_y + (max(rows) + 1) * scale,
    )


def roi_at_point(rois: list[Roi], row: int, col: int) -> Roi | None:
    """Return the topmost ROI containing a base-image pixel."""
    for roi in reversed(rois):
        if roi.r0 <= row < roi.r1 and roi.c0 <= col < roi.c1:
            return roi
    return None
