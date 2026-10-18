"""ROI model and base/display coordinate mapping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .transforms import decode_flip


@dataclass
class Roi:
    """A rectangular or polygon selection in base-image coordinates.

    ``r1``/``c1`` are exclusive for rectangles. For polygons they are the
    exclusive bounds of the polygon's bounding box, while ``points`` holds the
    inclusive ``(row, col)`` vertices.
    """

    roi_id: int
    r0: int
    c0: int
    r1: int
    c1: int
    threshold_min: int
    threshold_max: int
    invert: bool
    shape: str = "rectangle"
    points: list[tuple[int, int]] | None = None

    @property
    def bbox(self) -> list[int]:
        return [self.r0, self.c0, self.r1, self.c1]

    def contains(self, row: int, col: int) -> bool:
        """Return True when a base-image pixel is inside this ROI."""
        if self.shape != "polygon" or not self.points:
            return self.r0 <= row < self.r1 and self.c0 <= col < self.c1
        points = self.points
        inside = False
        previous_row, previous_col = points[-1]
        for point_row, point_col in points:
            if (point_col > col) != (previous_col > col):
                crossing = (
                    (previous_row - point_row) * (col - point_col)
                    / (previous_col - point_col)
                    + point_row
                )
                if row < crossing:
                    inside = not inside
            previous_row, previous_col = point_row, point_col
        return inside

    def mask_array(self, height: int, width: int) -> np.ndarray:
        """Rasterize the ROI as a boolean mask at working-image resolution."""
        mask = np.zeros((height, width), dtype=bool)
        if self.shape != "polygon" or not self.points:
            mask[self.r0:self.r1, self.c0:self.c1] = True
            return mask
        if len(self.points) < 3:
            return mask
        stencil = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(stencil)
        draw.polygon([(col, row) for row, col in self.points], fill=255)
        return np.asarray(stencil, dtype=np.uint8) > 0


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
    """Return the ROI bounding rectangle in canvas coordinates."""
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


def roi_screen_points(
    roi: Roi,
    height: int,
    width: int,
    flip: str,
    rotation: int,
    scale: float,
    offset: tuple[float, float],
) -> list[tuple[float, float]]:
    """Return polygon vertices in canvas coordinates."""
    if not roi.points:
        return []
    offset_x, offset_y = offset
    screen = []
    for row, col in roi.points:
        display_row, display_col = base_to_display(row, col, height, width, flip, rotation)
        screen.append((offset_x + display_col * scale, offset_y + display_row * scale))
    return screen


def roi_at_point(rois: list[Roi], row: int, col: int) -> Roi | None:
    """Return the topmost ROI containing a base-image pixel."""
    for roi in reversed(rois):
        if roi.contains(row, col):
            return roi
    return None
