"""Undo / redo for mask edits *and* selection (ROI) edits.

Every history entry is either a mask snapshot or a snapshot of the ROI list.
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
        self._push_history(("mask", self.base_mask_img.copy()))
        self.last_action = action

    def _roi_snapshot(self) -> tuple:
        return ("rois", copy.deepcopy(self.rois), self.active_roi_id, self.next_roi_id)

    def push_roi_undo(self) -> None:
        """Call before any change to the ROI list (create, delete, extend, erase)."""
        self._push_history(self._roi_snapshot())
        self.last_action = "roi"

    # --- restoring ------------------------------------------------------------ #

    def _current_of_kind(self, kind: str) -> tuple | None:
        if kind == "mask":
            return ("mask", self.base_mask_img.copy()) if self.base_mask_img is not None else None
        return self._roi_snapshot()

    def _restore(self, entry: tuple, op: str) -> None:
        if entry[0] == "mask":
            before = np.asarray(self.base_mask_img).copy()
            self.base_mask_img = entry[1]
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

    def undo(self) -> None:
        if not hasattr(self, "redo_stack"):
            self.redo_stack = []
        self._step(self.undo_stack, self.redo_stack, "undo", "Nothing to undo")

    def redo(self) -> None:
        if not hasattr(self, "redo_stack"):
            self.redo_stack = []
        self._step(self.redo_stack, self.undo_stack, "redo", "Nothing to redo")
