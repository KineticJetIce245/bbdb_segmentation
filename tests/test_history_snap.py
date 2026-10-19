"""Tests for undo/redo of masks and selections, the edge-aware brush and Clean scope."""
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image
from scipy import ndimage

from bbdb_seg.clean import _disk, colour_region, snap_stroke
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


def nissl_like() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """White background, two white-matter blobs (205), cortex (150) with a thin
    pale outer layer (200) that has almost the white-matter gray level."""
    rng = np.random.default_rng(3)
    img = np.full((120, 160), 250.0)
    tissue = np.zeros(img.shape, bool); tissue[10:110, 10:150] = True
    img[tissue] = 150
    layer1 = tissue & ~ndimage.binary_erosion(tissue, iterations=3)   # 3 px pale rim
    img[layer1] = 200
    wm_a = np.zeros(img.shape, bool); wm_a[30:90, 25:65] = True
    wm_b = np.zeros(img.shape, bool); wm_b[30:90, 95:135] = True
    img[wm_a | wm_b] = 205
    img = np.clip(ndimage.gaussian_filter(img, 1.0) + rng.normal(0, 3, img.shape), 0, 255)
    return img.astype(np.uint8), wm_a, wm_b, layer1


class ColourPickTests(unittest.TestCase):
    def test_wand_takes_touched_region_and_spares_pale_cortex(self) -> None:
        img, wm_a, wm_b, layer1 = nissl_like()
        scribble = np.zeros(img.shape, bool); scribble[60, 30:60] = True
        scribble[60, 18:24] = True                     # strays into dark cortex
        region, low, high = colour_region(img, scribble, background=img > 240)
        inner_a = ndimage.binary_erosion(wm_a, iterations=2)
        self.assertGreater(region[inner_a].mean(), 0.95)   # white matter A taken
        self.assertFalse(region[wm_b].any())               # B not touched by scribble
        self.assertLess(region[layer1].mean(), 0.05)       # thin pale layer spared
        self.assertTrue(190 < low < 205 < high < 225)

    def test_whole_image_takes_every_matching_region(self) -> None:
        img, wm_a, wm_b, layer1 = nissl_like()
        scribble = np.zeros(img.shape, bool); scribble[60, 30:60] = True
        region, _low, _high = colour_region(img, scribble, background=img > 240, connected=False)
        self.assertGreater(region[ndimage.binary_erosion(wm_b, iterations=2)].mean(), 0.95)
        self.assertLess(region[layer1].mean(), 0.05)


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


class _ThrGui(_HistGui):
    """Adds threshold sliders: apply() mimics ThresholdMixin.apply_threshold."""

    def __init__(self) -> None:
        super().__init__()
        self.global_threshold_min, self.global_threshold_max, self.global_invert = 0, 100, False
        self.threshold_min, self.threshold_max = 0, 100
        self.last_action = None
        self.commit_threshold_state()

    def _load_threshold_scope(self):
        roi = self.active_roi()
        src = roi if roi is not None else None
        if src is not None:
            self.threshold_min, self.threshold_max = src.threshold_min, src.threshold_max
        else:
            self.threshold_min, self.threshold_max = self.global_threshold_min, self.global_threshold_max

    def slide(self, low: int, high: int) -> None:   # one motion event of a slider drag
        self.threshold_min, self.threshold_max = low, high
        self.global_threshold_min, self.global_threshold_max = low, high
        if self.last_action != "threshold":
            self.push_undo("threshold")
        arr = np.zeros((20, 20), np.uint8)
        arr[:, :high // 10] = 255
        self.base_mask_img = Image.fromarray(arr)
        self.commit_threshold_state()


class ThresholdUndoTests(unittest.TestCase):
    def test_undo_moves_sliders_back_and_each_drag_is_one_step(self) -> None:
        gui = _ThrGui()
        gui.slide(0, 90); gui.slide(0, 80)      # drag 1 (two motion events)
        gui.end_threshold_drag()
        gui.slide(0, 60); gui.slide(0, 50)      # drag 2
        gui.end_threshold_drag()
        self.assertEqual(len(gui.undo_stack), 2)
        gui.undo()                               # back to the end of drag 1
        self.assertEqual((gui.threshold_min, gui.threshold_max), (0, 80))
        self.assertEqual(int((np.asarray(gui.base_mask_img) > 0).sum()), 20 * 8)
        gui.undo()                               # back to before drag 1
        self.assertEqual(gui.threshold_max, 100)
        gui.redo()
        self.assertEqual(gui.threshold_max, 80)
        gui.redo()
        self.assertEqual(gui.threshold_max, 50)


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
