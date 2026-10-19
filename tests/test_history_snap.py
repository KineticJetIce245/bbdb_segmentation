"""Tests for undo/redo of masks and selections, the edge-aware brush and Clean scope."""
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image
from scipy import ndimage

from bbdb_seg.clean import _disk, snap_stroke
from bbdb_seg.history import HistoryMixin
from bbdb_seg.roi import Roi
from tests.test_clean import _FakeGui, _Var, ring_image


def stroke(shape, start, end, radius, core_fraction=1 / 3):
    """Footprint and centre line of a straight brush stroke (row, col points)."""
    footprint = np.zeros(shape, bool)
    core = np.zeros(shape, bool)
    core_r = max(1, round(radius * core_fraction))
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
    for t in np.linspace(0.0, 1.0, steps):
        row = int(round(start[0] + (end[0] - start[0]) * t))
        col = int(round(start[1] + (end[1] - start[1]) * t))
        footprint[row - radius:row + radius + 1, col - radius:col + radius + 1] |= _disk(radius)
        core[row - core_r:row + core_r + 1, col - core_r:col + core_r + 1] |= _disk(core_r)
    return footprint, core


def edge_image() -> np.ndarray:
    """Dark 'cortex' left of column 50, light 'white matter' right of it."""
    rng = np.random.default_rng(0)
    img = np.full((100, 100), 200, np.float32)
    img[:, :50] = 120
    img = ndimage.gaussian_filter(img, 1.2) + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class SnapStrokeTests(unittest.TestCase):
    def test_flat_tissue_keeps_whole_stroke(self) -> None:
        rng = np.random.default_rng(1)
        img = np.clip(180 + rng.normal(0, 3, (100, 100)), 0, 255).astype(np.uint8)
        footprint, core = stroke(img.shape, (50, 20), (50, 80), 8)
        out = snap_stroke(img, np.zeros(img.shape, bool), footprint, core, True)
        self.assertEqual(out.sum(), footprint.sum())

    def test_draw_stops_at_strong_edge(self) -> None:
        img = edge_image()
        before = np.zeros(img.shape, bool)
        before[:, 30:44] = True                       # existing cortex mask
        footprint, core = stroke(img.shape, (20, 46), (80, 46), 8)   # spans cols 38..54
        out = snap_stroke(img, before, footprint, core, True)
        self.assertEqual(out[:, 52:].sum(), 0)        # nothing spills into white matter
        self.assertTrue(out[20:80, 44:50].all())      # gap up to the edge is filled

    def test_erase_stops_at_strong_edge(self) -> None:
        img = edge_image()
        before = np.zeros(img.shape, bool)
        before[:, 30:58] = True                       # mask spills 8 px past the edge
        footprint, core = stroke(img.shape, (20, 56), (80, 56), 8)   # spans cols 48..64
        out = snap_stroke(img, before, footprint, core, False)
        after = before & ~out
        self.assertTrue(after[25:75, 44:50].all())    # cortex survives the eraser
        self.assertFalse(after[25:75, 52:58].any())   # the spill is gone

    def test_excluded_pixels_never_drawn(self) -> None:
        img = edge_image()
        excluded = np.zeros(img.shape, bool)
        excluded[:, 60:] = True
        footprint, core = stroke(img.shape, (50, 40), (50, 70), 6)
        out = snap_stroke(img, np.zeros(img.shape, bool), footprint, core, True, excluded=excluded)
        self.assertFalse((out & excluded).any())


class _HistGui(HistoryMixin):
    """Minimal stand-in for SegmentGui's history-related state."""

    def __init__(self) -> None:
        self.base_mask_img = Image.new("L", (20, 20), 0)
        self.undo_stack, self.redo_stack, self.ops = [], [], []
        self.rois: list[Roi] = []
        self.active_roi_id = None
        self.next_roi_id = 1
        self.status_var = _Var("")
        self.status_var.set = lambda v: setattr(self.status_var, "value", v)
        self.restored = 0

    def flush_slider_updates(self): pass
    def active_roi(self): return next((r for r in self.rois if r.roi_id == self.active_roi_id), None)
    def log_op(self, op, **kw): self.ops.append(op)
    def mark_edited(self): pass
    def update_status(self): pass
    def render_rois(self): pass
    def _load_threshold_scope(self): pass
    def on_mask_restored(self): self.restored += 1

    def paint(self, value: int) -> None:
        self.push_undo("stroke")
        arr = np.asarray(self.base_mask_img).copy()
        arr[2:5, 2:5] = value
        self.base_mask_img = Image.fromarray(arr)

    def add_roi(self) -> None:
        self.push_roi_undo()
        self.rois.append(Roi(self.next_roi_id, 0, 0, 10, 10, 0, 255, False))
        self.active_roi_id = self.next_roi_id
        self.next_roi_id += 1


class HistoryTests(unittest.TestCase):
    def mask_sum(self, gui) -> int:
        return int((np.asarray(gui.base_mask_img) > 0).sum())

    def test_undo_redo_interleaved_mask_and_selection(self) -> None:
        gui = _HistGui()
        gui.paint(255)                      # mask: 9 px
        gui.add_roi()                       # selection
        gui.undo()                          # undo selection
        self.assertEqual(gui.rois, [])
        self.assertEqual(self.mask_sum(gui), 9)
        gui.undo()                          # undo paint
        self.assertEqual(self.mask_sum(gui), 0)
        self.assertGreaterEqual(gui.restored, 1)
        gui.redo()                          # redo paint
        self.assertEqual(self.mask_sum(gui), 9)
        gui.redo()                          # redo selection
        self.assertEqual(len(gui.rois), 1)
        self.assertEqual(gui.active_roi_id, 1)
        gui.redo()
        self.assertEqual(gui.status_var.value, "Nothing to redo")

    def test_new_edit_clears_redo(self) -> None:
        gui = _HistGui()
        gui.paint(255)
        gui.undo()
        self.assertEqual(len(gui.redo_stack), 1)
        gui.add_roi()
        self.assertEqual(gui.redo_stack, [])

    def test_undo_restores_deleted_selection_with_its_mask(self) -> None:
        gui = _HistGui()
        gui.add_roi()
        gui.rois[0].shape = "brush"
        gui.rois[0].mask = np.ones((20, 20), bool)
        gui.push_roi_undo()                 # e.g. brush-erase start
        gui.rois[0].mask[:10] = False       # erased in place
        gui.undo()
        self.assertTrue(gui.rois[0].mask.all())
        gui.redo()
        self.assertFalse(gui.rois[0].mask[:10].any())


class CleanScopeTests(unittest.TestCase):
    def test_clean_ignores_active_selection_and_keep_strokes_toggle(self) -> None:
        img, ring = ring_image()
        rough = ndimage.binary_dilation(ring, iterations=2)
        rough[5:8, 5:8] = True                       # speck far from any selection
        gui = _FakeGui(img, rough)

        class _Roi:
            def mask_array(self, h, w):
                m = np.zeros((h, w), bool); m[50:70, 50:70] = True
                return m
        gui.roi = _Roi()
        # a rough hand-painted tab into the tissue interior
        before = np.asarray(gui.base_mask_img).copy()
        painted = before.copy(); painted[58:61, 22:34] = 255
        tab = np.zeros(before.shape, bool); tab[58:61, 22:34] = True
        gui.record_hand_edit(before, painted, "draw", footprint=tab)
        gui.base_mask_img = Image.fromarray(painted)
        gui.keep_strokes_var = _Var(False)
        gui.clean_current()
        after = np.asarray(gui.base_mask_img) > 0
        self.assertFalse(after[5:8, 5:8].any())      # cleaned outside the selection
        beyond = ~ndimage.binary_dilation(ring, iterations=1)
        self.assertFalse((after & tab & beyond).any())     # rough stroke tidied


if __name__ == "__main__":
    unittest.main()
