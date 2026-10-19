"""Undo / redo for mask edits *and* selection (ROI) edits.

Every history entry is either a mask snapshot or a snapshot of the ROI list.
Mask snapshots also carry the threshold settings (sliders + Invert, globally
and per selection) that produced them, so undoing a slider drag also moves the
sliders back. Each slider drag (press to release) is one undo step.
Undo restores the latest entry and moves the current state of the same kind to
the redo stack; redo does the reverse. Any new edit clears the redo stack.
This mixin overrides ``push_undo``/``undo`` from PaintMixin, so it must come
before PaintMixin in SegmentGui's bases.
"""
from __future__ import annotations

import copy

import numpy as np

from .image_ops import mask_diff as diff_masks
from .settings import UNDO_LIMIT


class HistoryMixin:

    # --- recording ------------------------------------------------------------ #

    def _push_history(self, entry: tuple) -> None:
        self.undo_stack.append(entry)
        del self.undo_stack[:-UNDO_LIMIT]
        if not hasattr(self, "redo_stack"):
            self.redo_stack = []
        self.redo_stack.clear()

    def push_undo(self, action: str) -> None:
        if self.base_mask_img is None:
            return
        # the settings that produced the current mask (sliders may already
        # have moved on for the change about to be applied)
        thr = getattr(self, "_thr_committed", None) or self._threshold_snapshot()
        self._push_history(("mask", self.base_mask_img.copy(), thr))
        self.last_action = action

    # --- threshold settings -------------------------------------------------- #

    def _threshold_snapshot(self) -> dict | None:
        if not hasattr(self, "global_threshold_min"):
            return None
        return {
            "global": (int(self.global_threshold_min), int(self.global_threshold_max),
                       bool(self.global_invert)),
            "rois": {roi.roi_id: (int(roi.threshold_min), int(roi.threshold_max), bool(roi.invert))
                     for roi in self.rois},
        }

    def commit_threshold_state(self) -> None:
        """Record the settings that produced the current mask (after apply/restore/load)."""
        self._thr_committed = self._threshold_snapshot()

    def _restore_threshold(self, thr: dict | None) -> None:
        if not thr:
            return
        self.global_threshold_min, self.global_threshold_max, self.global_invert = thr["global"]
        for roi in self.rois:
            if roi.roi_id in thr["rois"]:
                roi.threshold_min, roi.threshold_max, roi.invert = thr["rois"][roi.roi_id]
        self._load_threshold_scope()   # sets the sliders; their callbacks see no change

    def end_threshold_drag(self, _event=None) -> None:
        """Slider released: the next slider move starts a new undo step."""
        self.flush_slider_updates()
        if self.last_action == "threshold":
            self.last_action = None

    def _roi_snapshot(self) -> tuple:
        return ("rois", copy.deepcopy(self.rois), self.active_roi_id, self.next_roi_id)

    def push_roi_undo(self) -> None:
        """Call before any change to the ROI list (create, delete, extend, erase)."""
        self._push_history(self._roi_snapshot())
        self.last_action = "roi"

    # --- restoring ------------------------------------------------------------ #

    def _current_of_kind(self, kind: str) -> tuple | None:
        if kind == "mask":
            if self.base_mask_img is None:
                return None
            return ("mask", self.base_mask_img.copy(), self._threshold_snapshot())
        return self._roi_snapshot()

    def _restore(self, entry: tuple, op: str) -> None:
        if entry[0] == "mask":
            before = np.asarray(self.base_mask_img).copy()
            self.base_mask_img = entry[1]
            self._restore_threshold(entry[2] if len(entry) > 2 else None)
            restored = getattr(self, "on_mask_restored", None)
            if restored is not None:
                restored()
            added, removed, bbox = diff_masks(before, np.asarray(self.base_mask_img))
            if bbox is not None:
                self.log_op(op, roi=self.active_roi(), added=added, removed=removed, bbox=bbox)
            self.mask_source = "edited"
            self.mark_edited()
        else:
            _kind, rois, active_id, next_id = entry
            self.rois[:] = copy.deepcopy(rois)
            self.active_roi_id = active_id if any(r.roi_id == active_id for r in self.rois) else None
            self.next_roi_id = max(next_id, self.next_roi_id)
            self._load_threshold_scope()
            self.log_op(op, extra={"target": "selection", "rois": len(self.rois)})
            self.render_rois()
            self.update_status()

    def _step(self, source: list, target: list, op: str, empty_msg: str) -> None:
        self.flush_slider_updates()
        if not source:
            self.status_var.set(empty_msg)
            return
        entry = source.pop()
        current = self._current_of_kind(entry[0])
        if current is not None:
            target.append(current)
            del target[:-UNDO_LIMIT]
        self.last_action = None
        self._restore(entry, op)
        self.commit_threshold_state()

    def undo(self) -> None:
        if not hasattr(self, "redo_stack"):
            self.redo_stack = []
        self._step(self.undo_stack, self.redo_stack, "undo", "Nothing to undo")

    def redo(self) -> None:
        if not hasattr(self, "redo_stack"):
            self.redo_stack = []
        self._step(self.redo_stack, self.undo_stack, "redo", "Nothing to redo")
