#!/usr/bin/env python3
"""GUI to manually segment Brain Biodiversity Bank sections.

Reads one ``csv/<specimen>_<stain>.csv`` file produced by
``make_segmentation_csvs.py``. Every row whose ``segmentation`` file does not
exist yet is opened one by one. Each image starts with an Otsu-based intensity
band shown as a red overlay; the min/max threshold sliders adjust that band,
and the mask can then be edited by hand.

Features
  * Two threshold sliders select the intensity range ``min <= value <= max``.
    "Invert" selects pixels outside that range instead. The initial range is
    picked from the Otsu level and the image border (dark or bright background).
  * Rectangular (Shift+left-drag), lasso (Shift+right-drag) and brush-shaped
    (L, then left-drag) ROIs, each with its own threshold settings. An active
    ROI limits threshold changes to its shape; outside pixels are preserved.
    Shift+click activates a ROI; Backspace deletes it. In brush-select mode,
    right-drag erases part of the active brush ROI.
  * Every mask/ROI operation is appended to ``<mask>.ops.jsonl``. PNG masks are
    written on Enter, s, q or when switching sections, not on every edit.
  * Paint (left drag) / erase (right drag), clear, erode, dilate.
  * Rotate / flip the view; the flip/rotation is recorded to the CSV, while the
    saved mask always stays aligned with the image on disk.
  * Zoom with the mouse wheel or +/- and pan with the middle mouse button or
    Ctrl+left-drag. The view transform never changes the saved mask.
  * Masks are saved as 0/1 PNGs at the image's native resolution.

Usage
  python segment_gui.py csv/lion_nissl.csv
  python segment_gui.py            # launches a file picker to choose the CSV
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import tkinter as tk

import numpy as np
from PIL import Image

from .session import IOMixin
from .status import StatusMixin
from .brush_erase import BrushEraseMixin
from .cleaning import CleanMixin
from .smoothing import StrokeSmoothMixin
from .brush_select import BrushSelectMixin
from .lasso import LassoMixin
from .painting import PaintMixin
from .work_queue import QueueMixin
from .rendering import RenderMixin
from .selection import RoiMixin
from .threshold import ThresholdMixin
from .tools import ToolsMixin
from .ui import UiMixin
from .paths import REPO_ROOT
from .settings import (
    SMOOTH_STROKES,
    APP_TITLE,
    DEFAULT_BRUSH,
    OVERLAY_ALPHA,
    ZOOM_STEP,
)


class SegmentGui(QueueMixin, IOMixin, StatusMixin, ThresholdMixin, RoiMixin, LassoMixin, CleanMixin, StrokeSmoothMixin, PaintMixin, BrushSelectMixin, BrushEraseMixin, ToolsMixin, RenderMixin, UiMixin, tk.Tk):
    def __init__(
        self,
        csv_path: Path | None,
        data_root_override: str | None = None,
        seg_root_override: str | None = None,
    ) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.withdraw()  # stay hidden until we know we have a valid CSV to show
        self.aborted = False

        self.seg_root = Path(seg_root_override).expanduser().resolve() if seg_root_override else REPO_ROOT
        resolved = self.resolve_inputs(csv_path, data_root_override)
        if resolved is None:
            self.aborted = True
            return
        self.csv_path, self.data_root, self.records, self.queue = resolved

        self._init_document_state()
        self._init_edit_state()
        self._init_clean_state()
        self.reset_smoothing()
        self._init_brush_select_state()
        self._init_view_state()
        self._init_tk_vars()
        self._init_window()

    def _init_document_state(self) -> None:
        self.pos = 0
        self.index = self.queue[0]
        self.cache: OrderedDict[int, tuple[np.ndarray, tuple[int, int]]] = OrderedDict()
        self.base_img: np.ndarray | None = None
        self.base_mask_img: Image.Image | None = None
        self.base_index: np.ndarray | None = None
        self.native_hw: tuple[int, int] = (0, 0)

    def _init_edit_state(self) -> None:
        self.flip = ""
        self.rotation = 0
        self.tool = "draw"
        self.undo_stack: list[Image.Image] = []
        self.last_action: str | None = None
        self.stroke_pushed = False
        self.brush = DEFAULT_BRUSH
        self.overlay_alpha = OVERLAY_ALPHA
        self.contrast_min = 0
        self.contrast_max = 255
        self.threshold_min = 0
        self.threshold_max = 255
        self.global_threshold_min = 0
        self.global_threshold_max = 255
        self.global_invert = False
        self.bright_border = False
        self.otsu = 0
        self.rois: list[Roi] = []
        self.active_roi_id: int | None = None
        self.next_roi_id = 1
        self.roi_items: list[int] = []
        self.roi_drag_item: int | None = None
        self.roi_drag_start: tuple[int, int] | None = None
        self.roi_dragging = False
        self.roi_drag_moved = False
        self.lasso_dragging = False
        self.lasso_points: list[tuple[int, int]] = []
        self.lasso_item: int | None = None
        self.lasso_moved = False
        self.stroke_before: np.ndarray | None = None
        self.stroke_op = "draw"
        self.stroke_roi_id: int | None = None
        self.threshold_pending = False
        self.alpha_pending = False
        self.contrast_pending = False
        self.overlay_pending = False
        self.slider_after_id: str | None = None
        self.ui_ready = False
        self.mask_dirty = False
        self.mask_source = ""
        self.segmented_status = False
        self.stop_auto_confirm = False
        self.status_cache: dict[int, tuple[bool, bool]] = {}
        self.last_pt: tuple[int, int] | None = None

    def _init_brush_select_state(self) -> None:
        self.brush_select_dragging = False
        self.brush_select_mask: np.ndarray | None = None
        self.brush_select_points: list[tuple[int, int]] = []
        self.brush_select_items: list[int] = []
        self.brush_select_last: tuple[int, int] | None = None
        self.brush_select_target_id: int | None = None
        self.brush_select_after_id: str | None = None
        self.brush_preview_item: int | None = None
        self.brush_preview_photo: ImageTk.PhotoImage | None = None
        self.brush_erase_dragging = False
        self.brush_erase_roi_id: int | None = None
        self.brush_erase_before: np.ndarray | None = None
        self.brush_erase_last: tuple[int, int] | None = None
        self.brush_erase_changed = False

    def _init_view_state(self) -> None:
        self.disp_index: np.ndarray | None = None
        self.disp_shape: tuple[int, int] = (0, 0)
        self.display_scale = 1.0
        self.display_offset: tuple[float, float] = (0.0, 0.0)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.panning = False
        self.pan_last: tuple[int, int] | None = None
        self.last_mouse: tuple[int, int] | None = None
        self.bg_photo: ImageTk.PhotoImage | None = None
        self.ov_photo: ImageTk.PhotoImage | None = None
        self.bg_id: int | None = None
        self.ov_id: int | None = None
        self.roi_photos: list[ImageTk.PhotoImage] = []
        self.cursor_id: int | None = None
        self.resize_after_id: str | None = None
        self.pending_save_id: str | None = None

    def _init_tk_vars(self) -> None:
        self.position_var = tk.StringVar()
        self.file_var = tk.StringVar()
        self.state_var = tk.StringVar()
        self.brush_var = tk.StringVar()
        self.brush_input_var = tk.StringVar()
        self.mask_src_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.done_var = tk.StringVar()
        self.otsu_var = tk.StringVar()
        self.invert_var = tk.BooleanVar(value=False)
        self.exclude_bg_var = tk.BooleanVar(value=True)
        self.exclude_legend_var = tk.BooleanVar(value=True)
        self.smooth_var = tk.BooleanVar(value=SMOOTH_STROKES)
        self.segmented_var = tk.BooleanVar(value=False)
        self.stop_auto_confirm_var = tk.BooleanVar(value=False)
        self.queue_filter_var = tk.StringVar(value="all")

    def _init_window(self) -> None:
        self.geometry("1500x950")
        self.minsize(1000, 700)
        self.deiconify()
        self.build_ui()
        self.ui_ready = True
        self.bind_all("<Key>", self.on_key)
        self.bind_all("<Control-z>", lambda _e: self.undo())
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.load_current()
        self.after(100, self.focus_force)

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym
        char = event.char.lower() if event.char else ""
        if key in ("Return", "KP_Enter"):
            self.accept_and_next()
        elif key in ("Left", "Up"):
            self.move(-1)
        elif key in ("Right", "Down"):
            self.move(1)
        elif key in ("plus", "KP_Add", "equal") or char in ("+", "="):
            self.zoom_center(ZOOM_STEP)
        elif key in ("minus", "KP_Subtract") or char == "-":
            self.zoom_center(1.0 / ZOOM_STEP)
        elif key in ("0", "KP_0") or char == "0":
            self.reset_view()
        elif key == "E":
            self.erode_mask()
        elif key == "D":
            self.dilate_mask()
        elif char == "k":
            self.clean_current()
        elif key in ("BackSpace", "Delete"):
            self.delete_active_roi()
        elif char == "t":
            self.reset_threshold()
        elif char == "i":
            self.toggle_invert()
        elif char == "r":
            self.rotate(1)
        elif char == "h":
            self.flip_h()
        elif char == "v":
            self.flip_v()
        elif char == "c":
            self.clear_mask()
        elif char == "e":
            self.toggle_erase()
        elif char == "f":
            self.toggle_fill()
        elif char == "l":
            self.toggle_brush_select()
        elif char == "u":
            self.undo()
        elif char == "s":
            self.save_now()
        elif char == "[":
            self.change_brush(-2)
        elif char == "]":
            self.change_brush(2)
        elif key == "Escape" or char == "q":
            self.on_close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually segment Brain Biodiversity Bank sections.")
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="group CSV to segment, e.g. csv/lion_nissl.csv (if omitted, a file picker opens)",
    )
    parser.add_argument("--data-root", default=None, help="brain_biodiversity_bank folder containing v.1/ (auto-detected by default)")
    parser.add_argument("--seg-root", default=None, help="folder the CSV 'segmentation' paths are relative to (default: this folder)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None
    app = SegmentGui(csv_path, args.data_root, args.seg_root)
    if app.aborted:
        app.destroy()
        return
    app.mainloop()


if __name__ == "__main__":
    main()
