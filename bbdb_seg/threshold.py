"""Threshold range, invert and coalescing methods."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .image_ops import build_threshold_mask, mask_diff as diff_masks
from .roi import Roi
from .settings import SLIDER_UPDATE_MS

class ThresholdMixin:

    def cancel_slider_updates(self) -> None:
        if self.slider_after_id is not None:
            self.after_cancel(self.slider_after_id)
            self.slider_after_id = None
        self.threshold_pending = False
        self.alpha_pending = False
        self.contrast_pending = False
        self.overlay_pending = False

    def schedule_slider_update(self, kind: str) -> None:
        """Coalesce rapid slider/stroke events into one render every ~30 ms."""
        if not self.ui_ready:
            return
        if kind == "threshold":
            self.threshold_pending = True
        elif kind == "alpha":
            self.alpha_pending = True
        elif kind == "contrast":
            self.contrast_pending = True
        elif kind == "overlay":
            self.overlay_pending = True
        if self.slider_after_id is None:
            self.slider_after_id = self.after(SLIDER_UPDATE_MS, self.run_slider_updates)

    def run_slider_updates(self) -> None:
        self.slider_after_id = None
        self.apply_pending_slider_updates()

    def flush_slider_updates(self) -> None:
        if self.slider_after_id is not None:
            self.after_cancel(self.slider_after_id)
            self.slider_after_id = None
        self.apply_pending_slider_updates()

    def apply_pending_slider_updates(self) -> None:
        do_threshold = self.threshold_pending
        do_alpha = self.alpha_pending
        do_contrast = self.contrast_pending
        do_overlay = self.overlay_pending
        self.threshold_pending = False
        self.alpha_pending = False
        self.contrast_pending = False
        self.overlay_pending = False
        if do_threshold:
            self.apply_threshold()
        elif do_alpha or do_overlay:
            self.render_overlay()
        if do_contrast:
            self.render_background()

    def threshold_mask(self) -> Image.Image:
        assert self.base_img is not None
        return build_threshold_mask(
            self.base_img,
            self.threshold_min,
            self.threshold_max,
            bool(self.invert_var.get()),
        )

    def _sync_threshold_scope(self) -> None:
        roi = self.active_roi()
        if roi is not None:
            roi.threshold_min = self.threshold_min
            roi.threshold_max = self.threshold_max
            roi.invert = bool(self.invert_var.get())
        else:
            self.global_threshold_min = self.threshold_min
            self.global_threshold_max = self.threshold_max
            self.global_invert = bool(self.invert_var.get())

    def _load_threshold_scope(self) -> None:
        roi = self.active_roi()
        if roi is not None:
            low, high, invert = roi.threshold_min, roi.threshold_max, roi.invert
        else:
            low, high, invert = (
                self.global_threshold_min,
                self.global_threshold_max,
                self.global_invert,
            )
        self.threshold_min = int(low)
        self.threshold_max = int(high)
        self.threshold_min_scale.set(self.threshold_min)
        self.threshold_max_scale.set(self.threshold_max)
        self.invert_var.set(bool(invert))

    @staticmethod
    def _threshold_payload(roi: Roi | None, fallback: tuple[int, int, bool]) -> dict:
        if roi is not None:
            low, high, invert = roi.threshold_min, roi.threshold_max, roi.invert
        else:
            low, high, invert = fallback
        return {"min": int(low), "max": int(high), "invert": bool(invert)}

    def set_threshold_range(self, low: int, high: int) -> None:
        low = max(0, min(255, int(low)))
        high = max(0, min(255, int(high)))
        if low > high:
            low, high = high, low
        self.threshold_min = low
        self.threshold_max = high
        self.threshold_min_scale.set(low)
        self.threshold_max_scale.set(high)
        self._sync_threshold_scope()

    def apply_threshold(self) -> None:
        if self.base_img is None or self.base_mask_img is None:
            return
        # A slider drag fires many updates; keep one undo step for the whole drag.
        if self.last_action != "threshold":
            self.push_undo("threshold")
        before = np.asarray(self.base_mask_img).copy()
        thresholded = self.exclude_background(np.asarray(self.threshold_mask()))
        roi = self.active_roi()
        if roi is not None:
            after = before.copy()
            roi_mask = roi.mask_array(*before.shape)
            after[roi_mask] = thresholded[roi_mask]
        else:
            after = thresholded
        self.base_mask_img = Image.fromarray(after)
        added, removed, bbox = diff_masks(before, after)
        if bbox is not None:
            self.log_op(
                "threshold",
                roi=roi,
                threshold=self._threshold_payload(
                    roi, (self.threshold_min, self.threshold_max, bool(self.invert_var.get()))
                ),
                added=added,
                removed=removed,
                bbox=bbox,
            )
        self._sync_threshold_scope()
        self.mask_source = "threshold"
        self.mark_edited()

    def on_threshold_min_change(self, value: str) -> None:
        level = int(float(value))
        if level == self.threshold_min:
            return
        self.threshold_min = level
        if self.threshold_min > self.threshold_max:
            self.threshold_max = self.threshold_min
            self.threshold_max_scale.set(self.threshold_max)
        self._sync_threshold_scope()
        self.schedule_slider_update("threshold")

    def on_threshold_max_change(self, value: str) -> None:
        level = int(float(value))
        if level == self.threshold_max:
            return
        self.threshold_max = level
        if self.threshold_max < self.threshold_min:
            self.threshold_min = self.threshold_max
            self.threshold_min_scale.set(self.threshold_min)
        self._sync_threshold_scope()
        self.schedule_slider_update("threshold")

    def reset_threshold(self) -> None:
        self.flush_slider_updates()
        self.invert_var.set(False)
        if self.bright_border:
            low, high = 0, self.otsu
        else:
            low, high = self.otsu, 255
        self.set_threshold_range(low, high)
        self.last_action = None
        self.apply_threshold()

    def on_invert_change(self) -> None:
        self.flush_slider_updates()
        self._sync_threshold_scope()
        self.last_action = None
        self.apply_threshold()

    def toggle_invert(self) -> None:
        self.invert_var.set(not self.invert_var.get())
        self.on_invert_change()

    # --- selections --------------------------------------------------------- #

