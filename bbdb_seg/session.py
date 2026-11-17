"""GUI input/output, queue, saving and status methods."""

from __future__ import annotations

from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image

from .transforms import normalize_flip, normalize_rotation
from .datasets import write_records
from .image_ops import NEAREST, bright_background, load_gray_work, load_mask_work, otsu_threshold
from .ops_log import append_operation, operations_path
from .paths import resolve_path
from .roi import Roi
from .settings import APP_TITLE, MAX_CACHE_ITEMS, UNET_BADGE_COLOUR

class IOMixin:

    def seg_path(self, index: int) -> Path:
        return resolve_path(self.seg_root, self.records.at[index, "segmentation"])

    def init_mask_path(self, index: int) -> Path | None:
        """Predicted starting mask for a section, or None if there is none.

        A prediction is only a starting point: the caller never uses it when a
        mask has already been saved. The first existing candidate wins:
        the CSV layout (``<init-dir>/<group>/<base>.png``), a folder prepared
        like ``segmentations/`` (``<init-dir>/segmentations/<group>/<base>.png``),
        the flat name nnU-Net exports (``<init-dir>/<group>__<base>.png``), then
        ``<init-dir>/<base>.png``.
        """
        init_dir = getattr(self, "init_dir", None)
        if init_dir is None:
            return None
        base = self.records.at[index, "base"]
        group = self.records.at[index, "group"] if "group" in self.records.columns else ""
        candidates = []
        if group:
            candidates.append(init_dir / str(group) / f"{base}.png")
        candidates.append(resolve_path(init_dir, self.records.at[index, "segmentation"]))
        if group:
            candidates.append(init_dir / f"{group}__{base}.png")
        candidates.append(init_dir / f"{base}.png")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

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
        self._reset_section_rois()
        self._set_initial_threshold_band(base_img)
        self._load_section_mask(height, width)
        self._load_status()
        self._reset_section_view()
        self.on_section_loaded()

    def _reset_section_rois(self) -> None:
        self.cancel_slider_updates()
        self.rois.clear()
        self.active_roi_id = None
        self.next_roi_id = 1
        self.roi_dragging = False
        self.roi_drag_start = None
        self.roi_drag_moved = False
        self.lasso_dragging = False
        self.lasso_points = []
        self.lasso_moved = False
        self.stroke_before = None
        self.reset_brush_erase()
        self.reset_brush_select(render=False)
        self.clear_roi_items()

    def _set_initial_threshold_band(self, base_img: np.ndarray) -> None:
        self.otsu = otsu_threshold(base_img)
        self.otsu_var.set(f"Otsu: {self.otsu}")
        self.bright_border = bright_background(base_img, self.otsu)
        self.global_invert = False
        if self.bright_border:
            self.global_threshold_min, self.global_threshold_max = 0, self.otsu
        else:
            self.global_threshold_min, self.global_threshold_max = self.otsu, 255
        self.invert_var.set(False)
        self.set_threshold_range(self.global_threshold_min, self.global_threshold_max)

    def _load_section_mask(self, height: int, width: int) -> None:
        seg_path = self.seg_path(self.index)
        self.base_mask_img = None
        if seg_path.exists():
            try:
                self.base_mask_img = Image.fromarray(load_mask_work(seg_path, (height, width)))
                self.mask_source = "saved"
            except Exception:
                self.base_mask_img = None
        if self.base_mask_img is None:
            init_path = self.init_mask_path(self.index)
            if init_path is not None:
                try:
                    self.base_mask_img = Image.fromarray(load_mask_work(init_path, (height, width)))
                    self.mask_source = "prediction"
                except Exception:
                    self.base_mask_img = None
        if self.base_mask_img is None:
            self.base_mask_img = self.threshold_mask()
            self.mask_source = "threshold"

    def _reset_section_view(self) -> None:
        self.flip = normalize_flip(self.records.at[self.index, "flip"])
        self.rotation = normalize_rotation(self.records.at[self.index, "rotation"])
        self.last_pt = None
        self.mask_dirty = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.load_block_layer()
        self.commit_threshold_state()
        self.last_action = None
        self.disp_index = None
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.panning = False
        self.pan_last = None
        self.canvas.configure(cursor="crosshair" if self.tool == "fill" else "")
        self.render_all()

    # --- operation log ------------------------------------------------------ #

    def ops_path(self, index: int) -> Path:
        return operations_path(self.seg_path(index))

    def log_op(
        self,
        op: str,
        *,
        roi: Roi | None = None,
        threshold: dict | None = None,
        added: int = 0,
        removed: int = 0,
        bbox: list[int] | None = None,
        extra: dict | None = None,
    ) -> None:
        """Append one operation record to <mask>.ops.jsonl."""
        record = {
            "section": str(self.records.at[self.index, "base"]),
            "image": str(self.records.at[self.index, "image"]),
            "mask": str(self.records.at[self.index, "segmentation"]),
            "op": op,
            "segmented": bool(self.segmented_status),
            "stop_auto_confirm": bool(self.stop_auto_confirm),
            "roi_id": roi.roi_id if roi is not None else None,
            "roi": roi.bbox if roi is not None else None,
            "roi_shape": roi.shape if roi is not None else None,
            "roi_points": (
                [[row, col] for row, col in roi.points]
                if roi is not None and roi.points is not None
                else None
            ),
            "roi_threshold": (
                self._threshold_payload(
                    roi,
                    (self.threshold_min, self.threshold_max, bool(self.invert_var.get())),
                )
                if roi is not None
                else None
            ),
            "threshold": threshold,
            "added": int(added),
            "removed": int(removed),
            "bbox": bbox,
        }
        if extra:
            record.update(extra)
        try:
            append_operation(self.seg_path(self.index), record)
        except OSError as exc:
            self.status_var.set(f"Could not write ops log: {exc}")

    # --- threshold ---------------------------------------------------------- #

    def move(self, delta: int) -> None:
        self.flush_slider_updates()
        self.flush_mask()
        self.pos = (self.pos + delta) % len(self.queue)
        self.index = self.queue[self.pos]
        self.load_current()

    def accept_and_next(self) -> None:
        """Save the current mask as-is (even if unedited) and advance."""
        self.flush_slider_updates()
        self.mask_dirty = True
        self.flush_mask()
        self.status_var.set("Saved")
        if self.pos == len(self.queue) - 1:
            remaining = sum(not self.status_for_index(i)[0] for i in self.queue)
            if remaining == 0:
                messagebox.showinfo(APP_TITLE, f"All {len(self.queue)} images in this list are segmented.")
                self.update_status()
                return
        self.move(1)

    def flush_mask(self) -> None:
        self.save_mask()

    def save_mask(self) -> None:
        if self.base_mask_img is None or not self.mask_dirty:
            return
        out_path = self.seg_path(self.index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        native_h, native_w = self.native_hw
        full = np.asarray(self.base_mask_img.resize((native_w, native_h), NEAREST))
        Image.fromarray((full > 0).astype(np.uint8)).save(out_path)
        self.mask_dirty = False
        if not self.stop_auto_confirm and not self.segmented_status:
            self._set_segmented(True, auto=True)
        else:
            self.update_status()
        finalize = getattr(self, "finalize_prediction_after_save", None)
        if finalize is not None:
            finalize()

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
        write_records(self.csv_path, self.records)

    def commit_current(self) -> None:
        self.flush_slider_updates()
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
        shutdown = getattr(self, "shutdown_unet", None)
        if shutdown is not None:
            shutdown(render=False)
        self.destroy()

    # --- status ------------------------------------------------------------- #

    def update_status(self) -> None:
        row = self.records.iloc[self.index]
        flip_display = self.flip if self.flip else "none"
        remaining = sum(not self.status_for_index(i)[0] for i in self.queue)
        self.position_var.set(f"{self.pos + 1} / {len(self.queue)}  ({remaining} left)")
        self.file_var.set(f"{row.get('group', '')}   {row['base']}")
        roi = self.active_roi()
        if roi is None:
            self.mask_src_var.set(f"ROI: none ({len(self.rois)})" if self.rois else "ROI: none")
        else:
            index = next(
                (i + 1 for i, item in enumerate(self.rois) if item.roi_id == roi.roi_id),
                0,
            )
            self.mask_src_var.set(f"ROI: {index}/{len(self.rois)}")
        self.state_var.set(
            f"flip: {flip_display}    rotation: {self.rotation} x 90 ({self.rotation * 90:+d} deg)"
            f"    zoom: {self.zoom:.2f}x"
        )
        if self.segmented_status:
            self.done_var.set("\u25cf segmented")
            self.done_label.configure(fg="#2ca02c")
        elif getattr(self, "prediction_active", None) and self.prediction_active():
            # The prediction is the working mask but has not been edited or
            # confirmed yet: show it where "not segmented" normally sits.
            self.done_var.set("\u25cb predicted")
            self.done_label.configure(fg=UNET_BADGE_COLOUR)
        else:
            self.done_var.set("\u25cb not segmented")
            self.done_label.configure(fg="#c0392b")

