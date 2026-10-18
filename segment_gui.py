#!/usr/bin/env python3
"""GUI to manually segment Brain Biodiversity Bank sections.

Reads one ``csv/<specimen>_<stain>.csv`` file produced by
``make_segmentation_csvs.py``. Every row whose ``segmentation`` file does not
exist yet is opened one by one. Each image starts with an Otsu-threshold mask
(pixels >= threshold are 1, below are 0) shown as a red overlay; the
threshold slider moves that level, and the mask can then be edited by hand.

Features
  * Threshold slider (default: Otsu level) regenerates the mask from the image.
    "Invert" selects pixels below the level instead (useful for dark tissue on a
    bright background; auto-detected per image).
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
import math
from collections import OrderedDict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageDraw, ImageFilter
from scipy import ndimage

from common import (
    apply_transform,
    decode_flip,
    encode_flip,
    normalize_flip,
    normalize_rotation,
    resolve_path,
)

TOOL_DIR = Path(__file__).resolve().parent
APP_TITLE = "BBDB Segmentation GUI"
WORK_MAX = 1600          # longest side of the in-memory working image
OVERLAY_ALPHA = 110      # 0..255 opacity of the red overlay
MAX_CACHE_ITEMS = 4
SAVE_DEBOUNCE_MS = 400
DEFAULT_BRUSH = 14       # brush radius in on-screen pixels
MIN_BRUSH = 2
MAX_BRUSH = 120
UNDO_LIMIT = 30
TOOLS = ("draw", "erase", "fill")
ZOOM_STEP = 1.25         # one wheel notch / key press
MIN_ZOOM = 0.25          # relative to the fit-to-window scale
MAX_ZOOM = 8.0
MAX_VIEW_PIXELS = 12_000_000  # cap the rendered image so PhotoImage stays small
SHIFT_MASK = 0x0001
CONTROL_MASK = 0x0004

try:  # Pillow >= 9.1 moved the resampling filters onto Image.Resampling.
    NEAREST = Image.Resampling.NEAREST
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    NEAREST = Image.NEAREST
    LANCZOS = Image.LANCZOS


# --------------------------------------------------------------------------- #
# IO / image helpers
# --------------------------------------------------------------------------- #
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


def find_data_root(override: str | None) -> Path | None:
    if override:
        return Path(override).expanduser().resolve()
    for parent in [TOOL_DIR, *TOOL_DIR.parents]:
        for candidate in [
            parent,
            *sorted(parent.glob("brain_biodiversity_bank*")),
            *sorted(parent.glob("brain_biodiversity_bank*/brain_biodiversity_bank")),
        ]:
            if (candidate / "v.1" / "sections").is_dir():
                return candidate
    return None


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class SegmentGui(tk.Tk):
    def __init__(
        self,
        csv_path: Path | None,
        data_root_override: str | None = None,
        seg_root_override: str | None = None,
        review_all: bool = False,
    ) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.withdraw()  # stay hidden until we know we have a valid CSV to show
        self.aborted = False

        self.seg_root = Path(seg_root_override).expanduser().resolve() if seg_root_override else TOOL_DIR
        resolved = self.resolve_inputs(csv_path, data_root_override, review_all)
        if resolved is None:
            self.aborted = True
            return
        self.csv_path, self.data_root, self.records, self.queue = resolved

        self.pos = 0
        self.index = self.queue[0]
        self.cache: OrderedDict[int, tuple[np.ndarray, tuple[int, int]]] = OrderedDict()
        self.base_img: np.ndarray | None = None
        self.base_mask_img: Image.Image | None = None
        self.base_index: np.ndarray | None = None
        self.native_hw: tuple[int, int] = (0, 0)

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
        self.threshold = 0
        self.otsu = 0
        self.mask_dirty = False
        self.mask_source = ""
        self.last_pt: tuple[int, int] | None = None

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
        self.cursor_id: int | None = None
        self.resize_after_id: str | None = None
        self.pending_save_id: str | None = None
        self.mask_save_after_id: str | None = None

        self.position_var = tk.StringVar()
        self.file_var = tk.StringVar()
        self.state_var = tk.StringVar()
        self.brush_var = tk.StringVar()
        self.mask_src_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.done_var = tk.StringVar()
        self.otsu_var = tk.StringVar()
        self.invert_var = tk.BooleanVar(value=False)

        self.geometry("1500x950")
        self.minsize(1000, 700)
        self.deiconify()

        self.build_ui()
        self.bind_all("<Key>", self.on_key)
        self.bind_all("<Control-z>", lambda _e: self.undo())
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.load_current()
        self.after(100, self.focus_force)

    def resolve_inputs(
        self,
        csv_path: Path | None,
        data_root_override: str | None,
        review_all: bool,
    ) -> tuple[Path, Path, pd.DataFrame, list[int]] | None:
        """Pick a CSV and data root (dialogs if needed) and build the work queue."""
        if csv_path is None:
            chosen = filedialog.askopenfilename(
                parent=self,
                title="Select a segmentation CSV",
                initialdir=str(TOOL_DIR / "csv"),
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not chosen:
                return None
            csv_path = Path(chosen).expanduser().resolve()
        if not csv_path.exists():
            messagebox.showerror(APP_TITLE, f"CSV not found:\n{csv_path}", parent=self)
            return None

        data_root = find_data_root(data_root_override)
        if data_root is None:
            chosen = filedialog.askdirectory(
                parent=self, title="Select the brain_biodiversity_bank data folder (contains v.1/)",
            )
            if not chosen:
                return None
            data_root = Path(chosen).expanduser().resolve()

        try:
            records = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
            missing = {"base", "image", "segmentation"} - set(records.columns)
            if missing:
                raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        except Exception as exc:  # malformed/unreadable CSV -- report instead of crashing
            messagebox.showerror(APP_TITLE, f"Could not read CSV:\n{csv_path}\n\n{exc}", parent=self)
            return None
        if records.empty:
            messagebox.showerror(APP_TITLE, f"No rows found in:\n{csv_path}", parent=self)
            return None
        for column, default in (("flip", ""), ("rotation", "0")):
            if column not in records.columns:
                records[column] = default
        records["flip"] = records["flip"].apply(normalize_flip)
        records["rotation"] = records["rotation"].apply(normalize_rotation).astype(int)

        pending = [
            i for i in range(len(records))
            if not resolve_path(self.seg_root, records.at[i, "segmentation"]).exists()
        ]
        if review_all:
            queue = list(range(len(records)))
        elif pending:
            queue = pending
        elif messagebox.askyesno(
            APP_TITLE,
            f"All {len(records)} images in {csv_path.name} are already segmented.\n\nReview them anyway?",
            parent=self,
        ):
            queue = list(range(len(records)))
        else:
            return None
        if not (data_root / str(records.at[queue[0], "image"])).exists():
            messagebox.showerror(
                APP_TITLE,
                f"Images not found under data folder:\n{data_root}\n\n"
                "Pass --data-root /path/to/brain_biodiversity_bank",
                parent=self,
            )
            return None
        return csv_path, data_root, records, queue

    # --- UI construction ---------------------------------------------------- #
    def build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.grid(row=0, column=0, sticky="ew")

        ttk.Button(toolbar, text="< Prev", command=lambda: self.move(-1)).pack(side="left")
        ttk.Button(toolbar, text="Next >", command=lambda: self.move(1)).pack(side="left", padx=(2, 10))

        ttk.Button(toolbar, text="Rotate -90", command=lambda: self.rotate(-1)).pack(side="left")
        ttk.Button(toolbar, text="Rotate +90", command=lambda: self.rotate(1)).pack(side="left", padx=(2, 10))

        ttk.Button(toolbar, text="Flip H", command=self.flip_h).pack(side="left")
        ttk.Button(toolbar, text="Flip V", command=self.flip_v).pack(side="left", padx=(2, 10))

        self.mode_button = ttk.Button(toolbar, text="Mode: Draw", width=12, command=self.cycle_tool)
        self.mode_button.pack(side="left")
        ttk.Button(toolbar, text="Fill (f)", command=lambda: self.set_tool("fill")).pack(side="left", padx=(2, 0))
        ttk.Button(toolbar, text="Undo (Ctrl+Z)", command=self.undo).pack(side="left", padx=(2, 0))
        ttk.Button(toolbar, text="Clear mask", command=self.clear_mask).pack(side="left", padx=(2, 10))

        ttk.Button(toolbar, text="Erode (E)", command=self.erode_mask).pack(side="left")
        ttk.Button(toolbar, text="Dilate (D)", command=self.dilate_mask).pack(side="left", padx=(2, 10))

        ttk.Button(toolbar, text="Brush -", command=lambda: self.change_brush(-2)).pack(side="left")
        ttk.Label(toolbar, textvariable=self.brush_var, width=10).pack(side="left")
        ttk.Button(toolbar, text="Brush +", command=lambda: self.change_brush(2)).pack(side="left", padx=(0, 10))

        ttk.Button(toolbar, text="Accept & next (Enter)", command=self.accept_and_next).pack(side="left")
        self.done_label = tk.Label(toolbar, textvariable=self.done_var, font=("TkDefaultFont", 10, "bold"))
        self.done_label.pack(side="left", padx=(8, 0))

        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right")

        controls = ttk.Frame(self, padding=(8, 2))
        controls.grid(row=1, column=0, sticky="ew")
        self.threshold_scale = tk.Scale(
            controls, from_=0, to=255, orient="horizontal", length=320,
            label="Threshold", command=self.on_threshold_change,
            takefocus=False, highlightthickness=0,
        )
        self.threshold_scale.pack(side="left", padx=(0, 4))
        ttk.Button(controls, text="Otsu (t)", command=self.reset_threshold).pack(side="left")
        ttk.Checkbutton(
            controls, text="Invert (i)", variable=self.invert_var, command=self.apply_threshold,
            takefocus=False,
        ).pack(side="left", padx=(6, 4))
        ttk.Label(controls, textvariable=self.otsu_var, width=10).pack(side="left", padx=(0, 18))

        self.alpha_scale = tk.Scale(
            controls, from_=0, to=255, orient="horizontal", length=160,
            label="Overlay opacity", command=self.on_alpha_change,
            takefocus=False, highlightthickness=0,
        )
        self.alpha_scale.pack(side="left", padx=(0, 14))
        self.min_scale = tk.Scale(
            controls, from_=0, to=255, orient="horizontal", length=180,
            label="Contrast min", command=self.on_contrast_change,
            takefocus=False, highlightthickness=0,
        )
        self.min_scale.pack(side="left", padx=(0, 8))
        self.max_scale = tk.Scale(
            controls, from_=0, to=255, orient="horizontal", length=180,
            label="Contrast max", command=self.on_contrast_change,
            takefocus=False, highlightthickness=0,
        )
        self.max_scale.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Auto", command=self.auto_contrast).pack(side="left", padx=(4, 4))
        ttk.Button(controls, text="Reset", command=self.reset_contrast).pack(side="left")
        self.alpha_scale.set(self.overlay_alpha)
        self.min_scale.set(self.contrast_min)
        self.max_scale.set(self.contrast_max)

        self.canvas = tk.Canvas(self, background="#101010", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", lambda e: self.on_press(e, right=False))
        self.canvas.bind("<B1-Motion>", lambda e: self.on_drag(e, right=False))
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.canvas.bind("<ButtonPress-3>", lambda e: self.on_press(e, right=True))
        self.canvas.bind("<B3-Motion>", lambda e: self.on_drag(e, right=True))
        self.canvas.bind("<ButtonRelease-3>", self.end_stroke)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        self.canvas.bind("<Control-ButtonPress-1>", self.start_pan)
        self.canvas.bind("<Control-B1-Motion>", self.pan_drag)
        self.canvas.bind("<Control-ButtonRelease-1>", self.end_pan)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, ZOOM_STEP))
        self.canvas.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 1.0 / ZOOM_STEP))
        self.canvas.bind("<Shift-Button-4>", lambda e: self.change_brush(2))
        self.canvas.bind("<Shift-Button-5>", lambda e: self.change_brush(-2))

        statusbar = ttk.Frame(self, padding=(8, 4))
        statusbar.grid(row=3, column=0, sticky="ew")
        statusbar.columnconfigure(1, weight=1)
        ttk.Label(statusbar, textvariable=self.position_var, width=24).grid(row=0, column=0, sticky="w")
        ttk.Label(statusbar, textvariable=self.file_var).grid(row=0, column=1, sticky="w")
        ttk.Label(statusbar, textvariable=self.mask_src_var).grid(row=0, column=2, sticky="e", padx=(0, 12))
        ttk.Label(statusbar, textvariable=self.state_var).grid(row=0, column=3, sticky="e")

        help_text = (
            "Enter: accept & next    Left/Right: section    t: Otsu    i: invert    "
            "r: rotate 90    h/v: flip    Left-drag: draw    Right-drag: erase    e: draw/erase\n"
            "f: fill    Ctrl+Z/u: undo    c: clear    E/D: erode/dilate    [ ]: brush    s: save    "
            "Wheel: zoom    Shift+Wheel: brush    +/-: zoom    0: fit    Middle/Ctrl-drag: pan    Esc/q: quit\n"
            "(moving the threshold replaces hand edits)"
        )
        ttk.Label(self, text=help_text, padding=(8, 2)).grid(row=4, column=0, sticky="w")
        self.update_brush_label()
        self.update_mode_label()

    # --- data loading ------------------------------------------------------- #
    def seg_path(self, index: int) -> Path:
        return resolve_path(self.seg_root, self.records.at[index, "segmentation"])

    def get_base_img(self, index: int) -> tuple[np.ndarray, tuple[int, int]]:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        path = resolve_path(self.data_root, self.records.at[index, "image"])
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")
        data, native_hw = load_gray_work(path)
        self.cache[index] = (data, native_hw)
        if len(self.cache) > MAX_CACHE_ITEMS:
            self.cache.popitem(last=False)
        return data, native_hw

    def load_current(self) -> None:
        try:
            base_img, native_hw = self.get_base_img(self.index)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self.base_img = base_img
        self.native_hw = native_hw
        height, width = base_img.shape
        self.base_index = np.arange(height * width, dtype=np.int32).reshape(height, width)

        self.otsu = otsu_threshold(base_img)
        self.otsu_var.set(f"Otsu: {self.otsu}")
        self.invert_var.set(bright_background(base_img, self.otsu))
        self.threshold = self.otsu
        self.threshold_scale.set(self.otsu)

        seg_path = self.seg_path(self.index)
        self.base_mask_img = None
        if seg_path.exists():
            try:
                self.base_mask_img = Image.fromarray(load_mask_work(seg_path, (height, width)))
                self.mask_source = "saved"
            except Exception:
                self.base_mask_img = None
        if self.base_mask_img is None:
            self.base_mask_img = self.threshold_mask()
            self.mask_source = "threshold"

        self.flip = normalize_flip(self.records.at[self.index, "flip"])
        self.rotation = normalize_rotation(self.records.at[self.index, "rotation"])

        self.last_pt = None
        self.mask_dirty = False
        self.undo_stack.clear()
        self.last_action = None
        self.disp_index = None
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.panning = False
        self.pan_last = None
        self.canvas.configure(cursor="crosshair" if self.tool == "fill" else "")
        self.render_all()

    # --- threshold ---------------------------------------------------------- #
    def threshold_mask(self) -> Image.Image:
        assert self.base_img is not None
        if self.invert_var.get():
            binary = self.base_img < self.threshold
        else:
            binary = self.base_img >= self.threshold
        return Image.fromarray(binary.astype(np.uint8) * 255)

    def apply_threshold(self) -> None:
        if self.base_img is None:
            return
        # A slider drag fires many updates; keep one undo step for the whole drag.
        if self.last_action != "threshold":
            self.push_undo("threshold")
        self.base_mask_img = self.threshold_mask()
        self.mask_source = "threshold"
        self.mark_edited()

    def on_threshold_change(self, value: str) -> None:
        level = int(float(value))
        # Scale.set() also fires this callback; ignore programmatic no-op updates.
        if level == self.threshold:
            return
        self.threshold = level
        self.apply_threshold()

    def reset_threshold(self) -> None:
        self.threshold = self.otsu
        self.threshold_scale.set(self.otsu)
        self.apply_threshold()

    def toggle_invert(self) -> None:
        self.invert_var.set(not self.invert_var.get())
        self.apply_threshold()

    # --- rendering ---------------------------------------------------------- #
    def render_all(self) -> None:
        if self.base_img is None:
            return
        self.render_background()
        self.render_overlay()
        self.update_status()

    def canvas_size(self) -> tuple[int, int]:
        return max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)

    @staticmethod
    def clamp_pan(pan: float, canvas: int, image: int) -> float:
        """Keep the image on screen; centre it when it is smaller than the canvas."""
        if image <= canvas:
            return 0.0
        limit = (image - canvas) / 2.0
        return max(-limit, min(limit, pan))

    def max_zoom(self, disp_w: int, disp_h: int) -> float:
        """Largest zoom relative to fit that respects MAX_VIEW_PIXELS."""
        canvas_w, canvas_h = self.canvas_size()
        fit = max(min(canvas_w / disp_w, canvas_h / disp_h), 1e-3)
        pixel_limit = math.sqrt(MAX_VIEW_PIXELS / max(disp_w * disp_h, 1)) / fit
        return max(MIN_ZOOM, min(MAX_ZOOM, max(1.0, pixel_limit)))

    def render_background(self) -> None:
        if self.base_img is None or self.base_index is None:
            return
        disp_img = apply_transform(self.base_img, self.flip, self.rotation)
        self.disp_index = apply_transform(self.base_index, self.flip, self.rotation)
        disp_h, disp_w = disp_img.shape
        self.disp_shape = (disp_h, disp_w)

        canvas_w, canvas_h = self.canvas_size()
        fit_scale = max(min(canvas_w / disp_w, canvas_h / disp_h), 1e-3)
        self.zoom = max(MIN_ZOOM, min(self.zoom, self.max_zoom(disp_w, disp_h)))
        scale = fit_scale * self.zoom
        out_w = max(1, int(disp_w * scale))
        out_h = max(1, int(disp_h * scale))
        self.display_scale = scale

        self.pan_x = self.clamp_pan(self.pan_x, canvas_w, out_w)
        self.pan_y = self.clamp_pan(self.pan_y, canvas_h, out_h)
        offset_x = (canvas_w - out_w) / 2.0 + self.pan_x
        offset_y = (canvas_h - out_h) / 2.0 + self.pan_y
        self.display_offset = (offset_x, offset_y)

        disp_img = self.apply_contrast(disp_img)
        pil = Image.fromarray(disp_img, "L").convert("RGB").resize((out_w, out_h), LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(pil)
        if self.bg_id is None:
            self.bg_id = self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.bg_photo)
        else:
            self.canvas.itemconfigure(self.bg_id, image=self.bg_photo)
            self.canvas.coords(self.bg_id, offset_x, offset_y)

    def render_overlay(self) -> None:
        if self.base_mask_img is None:
            return
        mask_np = np.asarray(self.base_mask_img)
        disp_mask = apply_transform(mask_np, self.flip, self.rotation)
        out_w = max(1, int(self.disp_shape[1] * self.display_scale))
        out_h = max(1, int(self.disp_shape[0] * self.display_scale))
        stencil = Image.fromarray(disp_mask, "L").resize((out_w, out_h), NEAREST)
        rgba = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        rgba.paste(Image.new("RGBA", (out_w, out_h), (255, 40, 40, self.overlay_alpha)), (0, 0), stencil)
        self.ov_photo = ImageTk.PhotoImage(rgba)

        offset_x, offset_y = self.display_offset
        if self.ov_id is None:
            self.ov_id = self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.ov_photo)
        else:
            self.canvas.itemconfigure(self.ov_id, image=self.ov_photo)
            self.canvas.coords(self.ov_id, offset_x, offset_y)
        self.canvas.tag_raise(self.ov_id)
        if self.cursor_id is not None:
            self.canvas.tag_raise(self.cursor_id)

    # --- display controls --------------------------------------------------- #
    def apply_contrast(self, arr: np.ndarray) -> np.ndarray:
        """Window the 0-255 display values between contrast_min and contrast_max."""
        lo, hi = self.contrast_min, self.contrast_max
        if lo <= 0 and hi >= 255:
            return arr
        if hi <= lo:
            hi = lo + 1
        scaled = (arr.astype(np.float32) - lo) * (255.0 / (hi - lo))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def on_alpha_change(self, value: str) -> None:
        self.overlay_alpha = int(float(value))
        self.render_overlay()

    def on_contrast_change(self, _value: str | None = None) -> None:
        self.contrast_min = int(self.min_scale.get())
        self.contrast_max = int(self.max_scale.get())
        self.render_background()

    def auto_contrast(self) -> None:
        """Stretch the window to the current image's actual value range."""
        if self.base_img is None:
            return
        lo = int(self.base_img.min())
        hi = int(self.base_img.max())
        if hi <= lo:
            hi = min(255, lo + 1)
        self.min_scale.set(lo)
        self.max_scale.set(hi)

    def reset_contrast(self) -> None:
        self.min_scale.set(0)
        self.max_scale.set(255)

    # --- zoom / pan --------------------------------------------------------- #
    def zoom_at(self, x: float, y: float, factor: float) -> None:
        """Zoom by ``factor`` while keeping the image point under (x, y) fixed."""
        if self.base_img is None or self.display_scale <= 0:
            return
        disp_h, disp_w = self.disp_shape
        if disp_w <= 0 or disp_h <= 0:
            return
        new_zoom = max(MIN_ZOOM, min(self.max_zoom(disp_w, disp_h), self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-9:
            return

        old_scale = self.display_scale
        old_x, old_y = self.display_offset
        image_x = (x - old_x) / old_scale
        image_y = (y - old_y) / old_scale

        self.zoom = new_zoom
        canvas_w, canvas_h = self.canvas_size()
        fit_scale = max(min(canvas_w / disp_w, canvas_h / disp_h), 1e-3)
        new_scale = fit_scale * new_zoom
        out_w = max(1, int(disp_w * new_scale))
        out_h = max(1, int(disp_h * new_scale))
        self.pan_x = x - image_x * new_scale - (canvas_w - out_w) / 2.0
        self.pan_y = y - image_y * new_scale - (canvas_h - out_h) / 2.0
        self.pan_x = self.clamp_pan(self.pan_x, canvas_w, out_w)
        self.pan_y = self.clamp_pan(self.pan_y, canvas_h, out_h)
        self.render_all()

    def zoom_center(self, factor: float) -> None:
        if self.last_mouse is not None:
            x, y = self.last_mouse
        else:
            canvas_w, canvas_h = self.canvas_size()
            x, y = canvas_w / 2.0, canvas_h / 2.0
        self.zoom_at(x, y, factor)

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.render_all()

    # --- painting ----------------------------------------------------------- #
    def canvas_to_base(self, event_x: int, event_y: int) -> tuple[int, int] | None:
        if self.disp_index is None or self.base_img is None:
            return None
        offset_x, offset_y = self.display_offset
        scale = self.display_scale
        disp_x = (event_x - offset_x) / scale
        disp_y = (event_y - offset_y) / scale
        disp_h, disp_w = self.disp_shape
        if disp_x < 0 or disp_y < 0 or disp_x >= disp_w or disp_y >= disp_h:
            return None
        flat = int(self.disp_index[int(disp_y), int(disp_x)])
        base_w = self.base_img.shape[1]
        return flat // base_w, flat % base_w

    def stamp(self, r0: int, c0: int, r1: int, c1: int, value: int) -> None:
        if self.base_mask_img is None:
            return
        radius = max(1, int(round(self.brush / max(self.display_scale, 1e-6))))
        draw = ImageDraw.Draw(self.base_mask_img)
        draw.line([(c0, r0), (c1, r1)], fill=value, width=2 * radius)
        for row, col in ((r0, c0), (r1, c1)):
            draw.ellipse([col - radius, row - radius, col + radius, row + radius], fill=value)

    def start_pan(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        self.panning = True
        self.pan_last = (event.x, event.y)
        self.last_pt = None
        self.stroke_pushed = False
        self.on_leave(None)
        self.canvas.configure(cursor="fleur")

    def pan_drag(self, event: tk.Event) -> None:
        if not self.panning or self.pan_last is None:
            return
        dx = event.x - self.pan_last[0]
        dy = event.y - self.pan_last[1]
        self.pan_last = (event.x, event.y)
        self.last_mouse = (event.x, event.y)
        if not dx and not dy:
            return

        canvas_w, canvas_h = self.canvas_size()
        disp_h, disp_w = self.disp_shape
        out_w = max(1, int(disp_w * self.display_scale))
        out_h = max(1, int(disp_h * self.display_scale))
        new_x = self.clamp_pan(self.pan_x + dx, canvas_w, out_w)
        new_y = self.clamp_pan(self.pan_y + dy, canvas_h, out_h)
        move_x = new_x - self.pan_x
        move_y = new_y - self.pan_y
        if not move_x and not move_y:
            return

        # Moving the existing canvas items is much cheaper than re-rendering
        # the full background and overlay on every drag event.
        self.pan_x, self.pan_y = new_x, new_y
        self.display_offset = (
            self.display_offset[0] + move_x,
            self.display_offset[1] + move_y,
        )
        if self.bg_id is not None:
            self.canvas.move(self.bg_id, move_x, move_y)
        if self.ov_id is not None:
            self.canvas.move(self.ov_id, move_x, move_y)

    def end_pan(self, _event: tk.Event | None = None) -> None:
        if not self.panning:
            return
        self.panning = False
        self.pan_last = None
        self.canvas.configure(cursor="crosshair" if self.tool == "fill" else "")
        if self.last_mouse is not None and self.tool != "fill":
            self.update_cursor(self.last_mouse[0], self.last_mouse[1], self.tool == "erase")

    def on_press(self, event: tk.Event, right: bool) -> None:
        self.canvas.focus_set()
        if event.state & CONTROL_MASK:
            self.start_pan(event)
            return
        if self.tool == "fill":
            self.bucket_fill(event, 0 if right else 255)
            return
        self.last_pt = None
        self.stroke_pushed = False
        self.continue_stroke(event, right or self.tool == "erase")

    def on_drag(self, event: tk.Event, right: bool) -> None:
        if self.panning:
            self.pan_drag(event)
            return
        if event.state & CONTROL_MASK:
            self.start_pan(event)
            return
        if self.tool != "fill":
            self.continue_stroke(event, right or self.tool == "erase")

    def continue_stroke(self, event: tk.Event, erase: bool) -> None:
        point = self.canvas_to_base(event.x, event.y)
        self.update_cursor(event.x, event.y, erase)
        if point is None:
            self.last_pt = None
            return
        if not self.stroke_pushed:
            self.push_undo("stroke")
            self.stroke_pushed = True
        value = 0 if erase else 255
        if self.last_pt is None:
            self.stamp(point[0], point[1], point[0], point[1], value)
        else:
            self.stamp(self.last_pt[0], self.last_pt[1], point[0], point[1], value)
        self.last_pt = point
        self.mask_dirty = True
        self.render_overlay()

    def end_stroke(self, _event: tk.Event | None = None) -> None:
        if self.panning:
            self.end_pan(_event)
            return
        self.last_pt = None
        if self.mask_dirty:
            self.mask_source = "edited"
            self.update_status()
            self.schedule_mask_save()

    def bucket_fill(self, event: tk.Event, value: int) -> None:
        """Set the 4-connected mask region under the cursor to ``value``."""
        point = self.canvas_to_base(event.x, event.y)
        if point is None or self.base_mask_img is None:
            return
        mask = np.array(self.base_mask_img)
        row, col = point
        if mask[row, col] == value:
            return
        labels, _ = ndimage.label(mask == mask[row, col])
        self.push_undo("fill")
        mask[labels == labels[row, col]] = value
        self.base_mask_img = Image.fromarray(mask)
        self.mask_source = "edited"
        self.mark_edited()

    def push_undo(self, action: str) -> None:
        if self.base_mask_img is None:
            return
        self.undo_stack.append(self.base_mask_img.copy())
        del self.undo_stack[:-UNDO_LIMIT]
        self.last_action = action

    def undo(self) -> None:
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return
        self.base_mask_img = self.undo_stack.pop()
        self.last_action = None
        self.mask_source = "edited"
        self.mark_edited()

    def mark_edited(self) -> None:
        self.mask_dirty = True
        self.render_overlay()
        self.update_status()
        self.schedule_mask_save()

    def clear_mask(self) -> None:
        if self.base_mask_img is None:
            return
        self.push_undo("clear")
        self.base_mask_img.paste(0, (0, 0, self.base_mask_img.width, self.base_mask_img.height))
        self.mask_source = "edited"
        self.mark_edited()

    def apply_morphology(self, grow: bool) -> None:
        """Dilate (grow) or erode (shrink) the mask by one pixel step."""
        if self.base_mask_img is None:
            return
        kernel = ImageFilter.MaxFilter(3) if grow else ImageFilter.MinFilter(3)
        self.push_undo("morphology")
        self.base_mask_img = self.base_mask_img.filter(kernel)
        self.mask_source = "edited"
        self.mark_edited()

    def erode_mask(self) -> None:
        self.apply_morphology(grow=False)

    def dilate_mask(self) -> None:
        self.apply_morphology(grow=True)

    # --- transforms --------------------------------------------------------- #
    def set_transform(self, flip: str, rotation: int) -> None:
        self.flip = flip
        self.rotation = normalize_rotation(rotation)
        self.records.at[self.index, "flip"] = self.flip
        self.records.at[self.index, "rotation"] = self.rotation
        self.render_all()
        self.schedule_csv_save()

    def rotate(self, delta: int) -> None:
        self.set_transform(self.flip, self.rotation + delta)

    def flip_h(self) -> None:
        lr_flip, up_flip = decode_flip(self.flip)
        self.set_transform(encode_flip(not lr_flip, up_flip), self.rotation)

    def flip_v(self) -> None:
        lr_flip, up_flip = decode_flip(self.flip)
        self.set_transform(encode_flip(lr_flip, not up_flip), self.rotation)

    # --- brush / mode ------------------------------------------------------- #
    def change_brush(self, delta: int) -> None:
        self.brush = int(min(MAX_BRUSH, max(MIN_BRUSH, self.brush + delta)))
        self.update_brush_label()

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self.update_mode_label()
        self.canvas.configure(cursor="crosshair" if tool == "fill" else "")
        if tool == "fill":
            self.on_leave(None)

    def cycle_tool(self) -> None:
        self.set_tool(TOOLS[(TOOLS.index(self.tool) + 1) % len(TOOLS)])

    def toggle_erase(self) -> None:
        self.set_tool("draw" if self.tool == "erase" else "erase")

    def toggle_fill(self) -> None:
        self.set_tool("draw" if self.tool == "fill" else "fill")

    def update_brush_label(self) -> None:
        self.brush_var.set(f"brush: {self.brush}px")

    def update_mode_label(self) -> None:
        self.mode_button.configure(text=f"Mode: {self.tool.capitalize()}")

    def update_cursor(self, x: int, y: int, erase: bool) -> None:
        radius = self.brush
        colour = "#ff4040" if erase else "#ffff40"
        if self.cursor_id is None:
            self.cursor_id = self.canvas.create_oval(0, 0, 0, 0, outline=colour, width=2)
        self.canvas.coords(self.cursor_id, x - radius, y - radius, x + radius, y + radius)
        self.canvas.itemconfigure(self.cursor_id, outline=colour)
        self.canvas.tag_raise(self.cursor_id)

    def on_motion(self, event: tk.Event) -> None:
        self.last_mouse = (event.x, event.y)
        if self.panning:
            return
        if self.tool != "fill":
            self.update_cursor(event.x, event.y, self.tool == "erase")

    def on_leave(self, _event: tk.Event | None) -> None:
        self.last_mouse = None
        if self.cursor_id is not None:
            self.canvas.delete(self.cursor_id)
            self.cursor_id = None

    def on_wheel(self, event: tk.Event) -> None:
        if event.state & SHIFT_MASK:
            self.change_brush(2 if event.delta > 0 else -2)
            return
        if event.delta == 0:
            return
        self.zoom_at(event.x, event.y, ZOOM_STEP if event.delta > 0 else 1.0 / ZOOM_STEP)

    # --- navigation / keys -------------------------------------------------- #
    def move(self, delta: int) -> None:
        self.flush_mask()
        self.pos = (self.pos + delta) % len(self.queue)
        self.index = self.queue[self.pos]
        self.load_current()

    def accept_and_next(self) -> None:
        """Save the current mask as-is (even if unedited) and advance."""
        self.mask_dirty = True
        self.flush_mask()
        self.status_var.set("Saved")
        if self.pos == len(self.queue) - 1:
            remaining = sum(not self.seg_path(i).exists() for i in self.queue)
            if remaining == 0:
                messagebox.showinfo(APP_TITLE, f"All {len(self.queue)} images in this list are segmented.")
                self.update_status()
                return
        self.move(1)

    def flush_mask(self) -> None:
        if self.mask_save_after_id is not None:
            self.after_cancel(self.mask_save_after_id)
            self.mask_save_after_id = None
        self.save_mask()

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

    def on_canvas_configure(self, _event: tk.Event) -> None:
        if self.base_img is None:
            return
        if self.resize_after_id is not None:
            self.after_cancel(self.resize_after_id)
        self.resize_after_id = self.after(120, self.render_all)

    # --- saving ------------------------------------------------------------- #
    def save_mask(self) -> None:
        if self.base_mask_img is None or not self.mask_dirty:
            return
        out_path = self.seg_path(self.index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        native_h, native_w = self.native_hw
        full = np.asarray(self.base_mask_img.resize((native_w, native_h), NEAREST))
        Image.fromarray((full > 0).astype(np.uint8)).save(out_path)
        self.mask_dirty = False
        self.update_status()

    def schedule_mask_save(self) -> None:
        if self.mask_save_after_id is not None:
            self.after_cancel(self.mask_save_after_id)
        self.mask_save_after_id = self.after(SAVE_DEBOUNCE_MS, self.run_mask_save)
        self.status_var.set("Saving...")

    def run_mask_save(self) -> None:
        self.mask_save_after_id = None
        self.save_mask()
        self.status_var.set("Saved")

    def schedule_csv_save(self) -> None:
        if self.pending_save_id is not None:
            self.after_cancel(self.pending_save_id)
        self.pending_save_id = self.after(SAVE_DEBOUNCE_MS, self.run_csv_save)
        self.status_var.set("Saving...")

    def run_csv_save(self) -> None:
        self.pending_save_id = None
        self.write_csv()
        self.status_var.set("Saved")

    def write_csv(self) -> None:
        output = self.records.copy()
        output["flip"] = output["flip"].fillna("").astype(str)
        output["rotation"] = output["rotation"].fillna(0).astype(int)
        output.to_csv(self.csv_path, index=False)

    def commit_current(self) -> None:
        self.flush_mask()
        if self.pending_save_id is not None:
            self.after_cancel(self.pending_save_id)
            self.pending_save_id = None
            self.write_csv()

    def save_now(self) -> None:
        self.mask_dirty = True
        self.commit_current()
        self.status_var.set("Saved")

    def on_close(self) -> None:
        self.commit_current()
        self.destroy()

    # --- status ------------------------------------------------------------- #
    def update_status(self) -> None:
        row = self.records.iloc[self.index]
        flip_display = self.flip if self.flip else "none"
        remaining = sum(not self.seg_path(i).exists() for i in self.queue)
        self.position_var.set(f"{self.pos + 1} / {len(self.queue)}  ({remaining} left)")
        self.file_var.set(f"{row.get('group', '')}   {row['base']}")
        self.mask_src_var.set(f"mask: {self.mask_source}")
        self.state_var.set(
            f"flip: {flip_display}    rotation: {self.rotation} x 90 ({self.rotation * 90:+d} deg)"
            f"    zoom: {self.zoom:.2f}x"
        )
        if self.seg_path(self.index).exists():
            self.done_var.set("\u25cf segmented")
            self.done_label.configure(fg="#2ca02c")
        else:
            self.done_var.set("\u25cb not saved")
            self.done_label.configure(fg="#c0392b")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
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
    parser.add_argument("--all", action="store_true", help="open every image, not only those without a segmentation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None
    app = SegmentGui(csv_path, args.data_root, args.seg_root, args.all)
    if app.aborted:
        app.destroy()
        return
    app.mainloop()


if __name__ == "__main__":
    main()
