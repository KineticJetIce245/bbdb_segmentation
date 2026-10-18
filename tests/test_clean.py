"""Tests for background exclusion and edge snapping (bbdb_seg.clean)."""
from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

import bbdb_seg.clean as clean_module
from bbdb_seg.clean import background_mask, clean_mask, legend_mask, remove_small_islands, snap_to_edges
from bbdb_seg.cleaning import CleanMixin
from bbdb_seg.smoothing import StrokeSmoothMixin, chaikin


def ring_image(background: int = 255, tissue: int = 200, cortex: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Blurred section: background, a tissue disc, a darker cortex ring at its edge."""
    yy, xx = np.mgrid[:120, :120]
    r = np.hypot(yy - 60, xx - 60)
    img = np.full((120, 120), background, np.float32)
    img[r < 40] = tissue
    ring = (r >= 30) & (r < 40)
    img[ring] = cortex
    img = ndimage.gaussian_filter(img, 1.2)  # blurry edges, like the scans
    return np.clip(img, 0, 255).astype(np.uint8), ring


class BackgroundTests(unittest.TestCase):
    def test_white_background_detected_and_tissue_kept(self) -> None:
        img, ring = ring_image()
        bg = background_mask(img)
        self.assertTrue(bg[0, 0] and bg[-1, -1])
        self.assertFalse(bg[60, 60])                 # tissue centre
        core = ndimage.binary_erosion(ring, iterations=2)
        self.assertFalse(bg[core].any())             # cortex never background

    def test_dark_background_also_detected(self) -> None:
        img, ring = ring_image(background=15, tissue=180, cortex=230)
        bg = background_mask(img)
        self.assertTrue(bg[0, 0])
        self.assertFalse(bg[ndimage.binary_erosion(ring, iterations=2)].any())


class CleanTests(unittest.TestCase):
    def test_halo_island_removed_and_cortex_kept(self) -> None:
        img, ring = ring_image()
        rough = ndimage.binary_dilation(ring, iterations=2)  # threshold halo
        rough[5:8, 5:8] = True                                # speck on background
        out = clean_mask(img, rough)
        bg = background_mask(img)
        self.assertFalse((out & bg).any())
        self.assertFalse(out[5:8, 5:8].any())
        core = ndimage.binary_erosion(ring, iterations=1)
        self.assertGreater((out & core).sum() / core.sum(), 0.97)
        if clean_module._sk_watershed is not None:  # exact watershed available
            # everything left lies within 2 px of the true (blurred) ring boundary
            near = ndimage.binary_dilation(ring, iterations=2)
            self.assertLess((out & ~near).sum(), 0.02 * ring.sum())
        # and the outer halo (into the background) is gone entirely
        self.assertLess((out & ~ring).sum(), (rough & ~ring).sum() * 0.6)

    def test_thin_structure_survives_snapping(self) -> None:
        img = np.full((60, 60), 200, np.uint8)
        img[10:50, 29:31] = 100                       # 2-px-wide dark line
        mask = img < 150
        out = snap_to_edges(img, mask, band=3)
        self.assertGreater(out[10:50, 29:31].mean(), 0.9)

    def test_remove_small_islands_and_empty_input(self) -> None:
        mask = np.zeros((20, 20), bool)
        mask[1:3, 1:3] = True
        mask[8:18, 8:18] = True
        out = remove_small_islands(mask, 10)
        self.assertFalse(out[1:3, 1:3].any())
        self.assertTrue(out[8:18, 8:18].all())
        empty = np.zeros((20, 20), bool)
        self.assertFalse(clean_mask(np.full((20, 20), 255, np.uint8), empty).any())


class _Var:
    def __init__(self, value): self.value = value
    def get(self): return self.value


class _FakeGui(CleanMixin):
    """Just enough of SegmentGui to exercise CleanMixin without Tk."""
    def __init__(self, img, mask):
        self._init_clean_state()
        self.base_img = img
        from PIL import Image
        self.base_mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        self.exclude_bg_var = _Var(True)
        self.exclude_legend_var = _Var(True)
        self.status_var = _Var("")
        self.status_var.set = lambda v: setattr(self.status_var, "value", v)
        self.undo_stack, self.ops, self.roi = [], [], None
    def flush_slider_updates(self): pass
    def active_roi(self): return self.roi
    def push_undo(self, action): self.undo_stack.append(self.base_mask_img.copy())
    def log_op(self, op, **kw): self.ops.append((op, kw))
    def mark_edited(self): pass
    def update_status(self): pass


class CleanMixinTests(unittest.TestCase):
    def test_clean_button_is_undoable_and_logged(self) -> None:
        img, ring = ring_image()
        gui = _FakeGui(img, ndimage.binary_dilation(ring, iterations=2))
        gui.clean_current()
        self.assertEqual(len(gui.undo_stack), 1)
        self.assertEqual(gui.ops[0][0], "clean")
        after = np.asarray(gui.base_mask_img) > 0
        self.assertFalse((after & background_mask(img)).any())

    def test_exclude_background_toggle(self) -> None:
        img, _ring = ring_image()
        gui = _FakeGui(img, np.zeros(img.shape, bool))
        full = np.full(img.shape, 255, np.uint8)
        self.assertFalse(gui.exclude_background(full)[0, 0])   # corner is background
        self.assertTrue(gui.exclude_background(full)[60, 60])  # tissue kept
        gui.exclude_bg_var = _Var(False)
        self.assertTrue(gui.exclude_background(full)[0, 0])    # off = untouched


def slide_with_legend() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """White slide, a big tissue blob, dark 'labels' in the top/bottom margins."""
    img = np.full((400, 480), 255, np.uint8)   # ~ BBDB slide size / 2
    yy, xx = np.mgrid[:400, :480]
    tissue = np.hypot(yy - 200, xx - 240) < 110
    img[tissue] = 150
    legend = np.zeros(img.shape, bool)
    for r0, c0, h, w in ((10, 12, 14, 34), (10, 400, 14, 44), (360, 12, 3, 70), (372, 300, 6, 120)):
        legend[r0:r0 + h, c0:c0 + w] = True
    img[legend] = 20
    return img, tissue, legend


class LegendTests(unittest.TestCase):
    def test_labels_detected_tissue_untouched(self) -> None:
        img, tissue, legend = slide_with_legend()
        found = legend_mask(img)
        self.assertTrue(found[legend].all())
        self.assertFalse(found[tissue].any())

    def test_blob_outside_margins_is_not_legend(self) -> None:
        img, tissue, _legend = slide_with_legend()
        img[195:205, 5:12] = 20          # small fragment at mid-height: keep it
        self.assertFalse(legend_mask(img)[195:205, 5:12].any())


class HandEditTests(unittest.TestCase):
    def test_clean_keeps_brush_strokes_and_erasures(self) -> None:
        img, ring = ring_image()
        rough = ndimage.binary_dilation(ring, iterations=2)
        gui = _FakeGui(img, rough)
        before = np.asarray(gui.base_mask_img).copy()
        # hand-paint a 3 px tab into the tissue interior (not cortex-like): Clean
        # would snap it away, but a deliberate stroke must survive
        painted = before.copy()
        painted[58:61, 22:34] = 255
        stroke = np.zeros(before.shape, bool); stroke[58:61, 22:34] = True
        gui.record_hand_edit(before, painted, "draw", footprint=stroke)
        # hand-erase a chunk of the ring: Clean must not add it back
        erased = painted.copy()
        erased[55:65, 95:101] = 0
        gui.record_hand_edit(painted, erased, "erase")
        from PIL import Image
        gui.base_mask_img = Image.fromarray(erased)
        gui.clean_current()
        after = np.asarray(gui.base_mask_img) > 0
        self.assertTrue(after[58:61, 22:34].all())
        self.assertFalse(after[55:65, 95:101].any())

    def test_undo_clears_stale_erasures(self) -> None:
        img, ring = ring_image()
        gui = _FakeGui(img, ring)
        before = np.asarray(gui.base_mask_img).copy()
        erased = before.copy(); erased[55:65, 95:101] = 0
        gui.record_hand_edit(before, erased, "erase")
        gui.on_mask_restored()            # mask still has the ring (as after an undo)
        self.assertFalse(gui.hand_erase[55:65, 95:101].any())


class SmoothingTests(unittest.TestCase):
    def test_one_euro_smooths_jitter_but_not_fast_moves(self) -> None:
        class S(StrokeSmoothMixin):
            smooth_var = _Var(True)
        rng = np.random.default_rng(0)
        # slow, jittery hold around x=100: output varies much less than input
        s = S(); s.reset_smoothing()
        xs = [s.smooth_screen(int(100 + rng.integers(-3, 4)), 50, t)[0] for t in range(0, 800, 8)]
        self.assertLess(np.std(xs[20:]), 0.5 * 2.0)
        # fast sweep (2000 px/s): smoothed position stays within a few px of the pointer
        s = S(); s.reset_smoothing()
        lags = []
        for i, t in enumerate(range(0, 400, 8)):
            x = 16 * i
            lags.append(x - s.smooth_screen(x, 50, t)[0])
        self.assertLess(max(lags[5:]), 12)
        # disabled = raw pointer
        s = S(); s.smooth_var = _Var(False); s.reset_smoothing()
        self.assertEqual(s.smooth_screen(7, 9, 0), (7.0, 9.0))

    def test_chaikin_rounds_corners(self) -> None:
        square = [(0, 0), (0, 40), (40, 40), (40, 0)]
        out = chaikin(square, iterations=2)
        self.assertGreater(len(out), len(square))
        self.assertNotIn((0, 0), out)                 # sharp corner cut off
        rows = [p[0] for p in out]; cols = [p[1] for p in out]
        self.assertTrue(min(rows) >= 0 and max(rows) <= 40 and min(cols) >= 0 and max(cols) <= 40)


if __name__ == "__main__":
    unittest.main()
