"""GUI glue for background/legend exclusion and the Clean (k) button.

Hand edits win: pixels painted with the brush or fill bucket are never removed
by Clean (unless they are background or legend), and pixels erased by hand are
never added back. Clean only tidies boundaries produced by thresholding.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .clean import background_mask, clean_mask, legend_mask
from .image_ops import mask_diff as diff_masks


class CleanMixin:

    def _init_clean_state(self) -> None:
        self.excl_cache_key: int | None = None
        self.bg_soft: np.ndarray | None = None     # background + 1 px rim margin
        self.bg_strict: np.ndarray | None = None   # background only
        self.legend: np.ndarray | None = None      # slide text, numbers, scale bar
        self.hand_key: int | None = None
        self.hand_paint: np.ndarray | None = None
        self.hand_erase: np.ndarray | None = None

    # --- exclusion masks ---------------------------------------------------- #

    def _refresh_exclusions(self) -> None:
        key = id(self.base_img)
        if self.excl_cache_key != key or self.bg_soft is None:
            self.bg_strict = background_mask(self.base_img, margin=0)
            self.bg_soft = background_mask(self.base_img)
            self.legend = legend_mask(self.base_img, background=self.bg_strict)
            self.excl_cache_key = key

    def current_background(self, strict: bool = False) -> np.ndarray | None:
        if self.base_img is None:
            return None
        self._refresh_exclusions()
        return self.bg_strict if strict else self.bg_soft

    def current_excluded(self, strict: bool = False) -> np.ndarray | None:
        """Pixels that must never be cortex under the current checkboxes."""
        if self.base_img is None:
            return None
        self._refresh_exclusions()
        excluded = np.zeros(self.base_img.shape, bool)
        if bool(self.exclude_bg_var.get()):
            excluded |= self.bg_strict if strict else self.bg_soft
        if bool(self.exclude_legend_var.get()):
            excluded |= self.legend
        return excluded

    def exclude_background(self, mask_arr: np.ndarray, strict: bool = False) -> np.ndarray:
        """Zero excluded (background / legend) pixels in a 0/255 mask.

        ``strict`` (used for brush and fill) drops only true background, without
        the 1 px rim margin, so deliberate strokes along the edge are kept.
        """
        excluded = self.current_excluded(strict=strict)
        if excluded is None or excluded.shape != mask_arr.shape or not excluded.any():
            return mask_arr
        out = np.array(mask_arr, copy=True)
        out[excluded] = 0
        return out

    def _apply_exclusion_now(self, op: str) -> None:
        self.flush_slider_updates()
        if self.base_mask_img is None:
            return
        before = np.asarray(self.base_mask_img).copy()
        after = self.exclude_background(before)
        added, removed, bbox = diff_masks(before, after)
        if bbox is not None:
            self.push_undo(op)
            self.base_mask_img = Image.fromarray(after)
            self.log_op(op, removed=removed, bbox=bbox)
            self.mask_source = "edited"
            self.mark_edited()
        self.update_status()

    def on_exclude_bg_toggle(self) -> None:
        if bool(self.exclude_bg_var.get()):
            self._apply_exclusion_now("exclude_background")
        else:
            self.update_status()

    def on_exclude_legend_toggle(self) -> None:
        if bool(self.exclude_legend_var.get()):
            self._apply_exclusion_now("exclude_legend")
        else:
            self.update_status()

    # --- hand-edit memory --------------------------------------------------- #

    def _hand_layers(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        key = id(self.base_img)
        if self.hand_key != key or self.hand_paint is None or self.hand_paint.shape != shape:
            self.hand_paint = np.zeros(shape, bool)
            self.hand_erase = np.zeros(shape, bool)
            self.hand_key = key
        return self.hand_paint, self.hand_erase

    def record_hand_edit(self, before: np.ndarray, after: np.ndarray, op: str,
                         footprint: np.ndarray | None = None) -> None:
        """Remember which pixels the user painted or erased by hand.

        ``footprint`` is everything the brush touched (including pixels that were
        already in the mask); without it, the changed pixels are used.
        """
        paint, erase = self._hand_layers(before.shape)
        if footprint is None or footprint.shape != before.shape:
            footprint = (after > 0) != (before > 0)
        if op == "draw":
            touched = footprint & (after > 0)
            paint |= touched
            erase &= ~touched
        elif op == "erase":
            touched = footprint & (after == 0)
            erase |= touched
            paint &= ~touched

    def reset_hand_edits(self) -> None:
        self.hand_key = None
        self.hand_paint = None
        self.hand_erase = None

    def on_mask_restored(self) -> None:
        """After undo: pixels back in the mask no longer count as hand-erased."""
        if self.hand_erase is not None and self.base_mask_img is not None:
            current = np.asarray(self.base_mask_img) > 0
            if current.shape == self.hand_erase.shape:
                self.hand_erase &= ~current

    # --- Clean (k) ------------------------------------------------------------ #

    def clean_current(self) -> None:
        """Exclusions + edge snapping + small-island removal (undoable).

        Hand-painted pixels are kept, hand-erased pixels stay erased, and with
        an active ROI only pixels inside the ROI change.
        """
        self.flush_slider_updates()
        if self.base_img is None or self.base_mask_img is None:
            return
        before = np.asarray(self.base_mask_img).copy()
        current = before > 0
        excluded = self.current_excluded()
        cleaned = clean_mask(self.base_img, current, background=excluded)
        paint, erase = self._hand_layers(before.shape)
        strict = self.current_excluded(strict=True)
        cleaned |= current & paint & ~strict   # deliberate strokes survive
        cleaned &= ~erase                      # deliberate erasures stay erased
        after = cleaned.astype(np.uint8) * 255
        roi = self.active_roi()
        if roi is not None:
            inside = roi.mask_array(*before.shape)
            after = np.where(inside, after, before).astype(np.uint8)
        added, removed, bbox = diff_masks(before, after)
        if bbox is None:
            self.status_var.set("Clean: nothing to change")
            return
        self.push_undo("clean")
        self.base_mask_img = Image.fromarray(after)
        self.log_op("clean", roi=roi, added=added, removed=removed, bbox=bbox)
        self.status_var.set(f"Clean: -{removed} / +{added} px (Ctrl+Z to undo)")
        self.mask_source = "edited"
        self.mark_edited()
