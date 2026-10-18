"""Tests for background exclusion and edge snapping (bbdb_seg.clean)."""
from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

import bbdb_seg.clean as clean_module
from bbdb_seg.clean import background_mask, clean_mask, remove_small_islands, snap_to_edges
from bbdb_seg.cleaning import CleanMixin


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


if __name__ == "__main__":
    unittest.main()
