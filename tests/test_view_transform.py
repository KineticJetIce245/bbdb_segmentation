"""Tests for the segment_gui zoom/pan view transform.

Run from the repository root:

    python -m unittest discover -s tests -v

The GUI tests create a tiny synthetic section in a temporary directory.  They
skip themselves when Tk cannot connect to a display.
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import tkinter as tk
import unittest

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segment_gui import (  # noqa: E402
    CONTROL_MASK,
    MAX_VIEW_PIXELS,
    MIN_ZOOM,
    SHIFT_MASK,
    ZOOM_STEP,
    SegmentGui,
)


def make_event(
    x: int = 0,
    y: int = 0,
    state: int = 0,
    delta: int = 0,
    keysym: str = "",
    char: str = "",
) -> SimpleNamespace:
    """Small stand-in for a tkinter Event in tests."""
    return SimpleNamespace(x=x, y=y, state=state, delta=delta, keysym=keysym, char=char)


class SegmentGuiViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="bbdb-view-tests-")
        root = Path(cls._tmp.name)
        cls.data_root = root / "brain_biodiversity_bank"
        section_dir = cls.data_root / "v.1" / "sections" / "test_group"
        section_dir.mkdir(parents=True)

        # Two clear intensity classes so Otsu produces a mask with both 0 and 1.
        image = np.full((60, 80), 220, dtype=np.uint8)
        image[:, :40] = 30
        Image.fromarray(image, "L").save(section_dir / "one.png")

        cls.csv_path = root / "test_group.csv"
        with cls.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "group", "specimen", "species", "stain", "section",
                    "base", "image", "segmentation", "flip", "rotation",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "group": "test_group",
                "specimen": "test",
                "species": "Testus testus",
                "stain": "nissl",
                "section": "1",
                "base": "one",
                "image": "v.1/sections/test_group/one.png",
                "segmentation": "segmentations/test_group/one.png",
                "flip": "",
                "rotation": "0",
            })

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        # A fresh seg root per test prevents a saved mask in one test from
        # changing another test's queue.
        self.seg_root = Path(self._tmp.name) / f"seg-{self._testMethodName}"
        self.seg_root.mkdir()
        try:
            self.app = SegmentGui(
                self.csv_path,
                data_root_override=str(self.data_root),
                seg_root_override=str(self.seg_root),
            )
        except tk.TclError as exc:  # no display, e.g. headless CI
            self.skipTest(f"Tk display unavailable: {exc}")
        self.app.withdraw()
        self.app.update_idletasks()
        self.app.render_all()
        self.mask_before = np.asarray(self.app.base_mask_img).copy()

    def tearDown(self) -> None:
        app = getattr(self, "app", None)
        if app is None:
            return
        if app.mask_save_after_id is not None:
            app.after_cancel(app.mask_save_after_id)
            app.mask_save_after_id = None
        app.destroy()
        self.app = None

    def center(self) -> tuple[float, float]:
        offset_x, offset_y = self.app.display_offset
        disp_h, disp_w = self.app.disp_shape
        return (
            offset_x + disp_w * self.app.display_scale / 2.0,
            offset_y + disp_h * self.app.display_scale / 2.0,
        )

    def test_zoom_keeps_point_under_cursor(self) -> None:
        x, y = self.center()
        before = self.app.canvas_to_base(x, y)
        self.assertIsNotNone(before)

        self.app.zoom_at(x, y, ZOOM_STEP)

        self.assertGreater(self.app.zoom, 1.0)
        after = self.app.canvas_to_base(x, y)
        self.assertIsNotNone(after)
        # Integer display rounding may move the point by at most one base pixel.
        self.assertLessEqual(abs(before[0] - after[0]), 1)
        self.assertLessEqual(abs(before[1] - after[1]), 1)

    def test_zoom_is_clamped_to_pixel_budget(self) -> None:
        for _ in range(20):
            self.app.zoom_center(ZOOM_STEP)

        disp_h, disp_w = self.app.disp_shape
        out_w = max(1, int(disp_w * self.app.display_scale))
        out_h = max(1, int(disp_h * self.app.display_scale))
        self.assertLessEqual(self.app.zoom, self.app.max_zoom(disp_w, disp_h) + 1e-9)
        self.assertLessEqual(out_w * out_h, MAX_VIEW_PIXELS * 1.02)

        self.app.zoom = 0.01
        self.app.render_all()
        self.assertGreaterEqual(self.app.zoom, MIN_ZOOM)

    def test_pan_updates_offset_without_touching_mask(self) -> None:
        x, y = self.center()
        self.app.zoom_at(x, y, 2.0)
        pan_before = (self.app.pan_x, self.app.pan_y)
        offset_before = self.app.display_offset

        self.app.start_pan(make_event(x, y))
        self.app.pan_drag(make_event(x + 30, y - 20))
        self.app.end_pan(make_event(x + 30, y - 20))

        self.assertFalse(self.app.panning)
        self.assertNotEqual((self.app.pan_x, self.app.pan_y), pan_before)
        self.assertNotEqual(self.app.display_offset, offset_before)
        np.testing.assert_array_equal(self.mask_before, np.asarray(self.app.base_mask_img))
        self.assertFalse(self.app.mask_dirty)

    def test_ctrl_left_drag_pans_instead_of_painting(self) -> None:
        x, y = self.center()
        self.app.zoom_at(x, y, 2.0)
        pan_before = (self.app.pan_x, self.app.pan_y)

        self.app.on_press(make_event(x, y, state=CONTROL_MASK), right=False)
        self.assertTrue(self.app.panning)
        self.app.on_drag(make_event(x + 25, y + 15, state=CONTROL_MASK), right=False)
        self.app.end_stroke(make_event(x + 25, y + 15, state=CONTROL_MASK))

        self.assertFalse(self.app.panning)
        self.assertNotEqual((self.app.pan_x, self.app.pan_y), pan_before)
        np.testing.assert_array_equal(self.mask_before, np.asarray(self.app.base_mask_img))
        self.assertFalse(self.app.mask_dirty)

    def test_wheel_zooms_and_shift_wheel_changes_brush(self) -> None:
        x, y = self.center()
        zoom_before = self.app.zoom

        self.app.on_wheel(make_event(x, y, delta=120))
        self.assertGreater(self.app.zoom, zoom_before)

        zoom_after = self.app.zoom
        brush_before = self.app.brush
        self.app.on_wheel(make_event(x, y, state=SHIFT_MASK, delta=120))
        self.assertEqual(self.app.zoom, zoom_after)
        self.assertGreater(self.app.brush, brush_before)

    def test_keyboard_zoom_and_fit(self) -> None:
        self.app.on_key(make_event(keysym="plus", char="+"))
        self.assertGreater(self.app.zoom, 1.0)

        self.app.on_key(make_event(keysym="minus", char="-"))
        self.app.on_key(make_event(keysym="0", char="0"))

        self.assertEqual(self.app.zoom, 1.0)
        self.assertEqual((self.app.pan_x, self.app.pan_y), (0.0, 0.0))

    def test_paint_maps_through_zoom_and_pan(self) -> None:
        x, y = self.center()
        self.app.zoom_at(x, y, 2.0)
        self.app.start_pan(make_event(x, y))
        self.app.pan_drag(make_event(x + 40, y + 20))
        self.app.end_pan(make_event(x + 40, y + 20))

        canvas_w, canvas_h = self.app.canvas_size()
        mask = np.asarray(self.app.base_mask_img)
        base_w = self.app.base_img.shape[1]
        offset_x, offset_y = self.app.display_offset
        scale = self.app.display_scale
        screen = None
        base = None
        for flat in np.flatnonzero(mask.ravel() == 0):
            flat = int(flat)
            disp = np.argwhere(self.app.disp_index == flat)
            if not len(disp):
                continue
            disp_y, disp_x = disp[0]
            screen_x = int(offset_x + (disp_x + 0.5) * scale)
            screen_y = int(offset_y + (disp_y + 0.5) * scale)
            point = (flat // base_w, flat % base_w)
            if (0 <= screen_x < canvas_w and 0 <= screen_y < canvas_h
                    and self.app.canvas_to_base(screen_x, screen_y) == point):
                screen = (screen_x, screen_y)
                base = point
                break
        self.assertIsNotNone(screen, "no mask=0 pixel was visible")

        self.app.tool = "draw"
        self.app.on_press(make_event(*screen), right=False)
        self.app.on_drag(make_event(screen[0] + 1, screen[1] + 1), right=False)
        self.app.end_stroke(make_event(screen[0] + 1, screen[1] + 1))

        after = np.asarray(self.app.base_mask_img)
        self.assertEqual(int(after[base]), 255)
        self.assertTrue(self.app.mask_dirty)
        # This test is about coordinates, not writing a mask.
        if self.app.mask_save_after_id is not None:
            self.app.after_cancel(self.app.mask_save_after_id)
            self.app.mask_save_after_id = None
        self.app.mask_dirty = False

    def test_save_uses_native_size_after_zoom(self) -> None:
        x, y = self.center()
        self.app.zoom_at(x, y, 2.0)
        self.app.mask_dirty = True
        self.app.save_mask()

        out_path = self.seg_root / "segmentations" / "test_group" / "one.png"
        self.assertTrue(out_path.is_file())
        with Image.open(out_path) as saved:
            self.assertEqual(saved.size, (80, 60))
            values = set(np.asarray(saved.convert("L")).ravel().tolist())
        self.assertTrue(values.issubset({0, 1}))
        self.assertIn(0, values)
        self.assertIn(1, values)


class PanClampTests(unittest.TestCase):
    def test_centres_when_image_is_smaller_than_canvas(self) -> None:
        self.assertEqual(SegmentGui.clamp_pan(500.0, 100, 50), 0.0)

    def test_clamps_to_image_edges_when_image_is_larger(self) -> None:
        self.assertEqual(SegmentGui.clamp_pan(1000.0, 100, 300), 100.0)
        self.assertEqual(SegmentGui.clamp_pan(-1000.0, 100, 300), -100.0)
        self.assertEqual(SegmentGui.clamp_pan(25.0, 100, 300), 25.0)


if __name__ == "__main__":
    unittest.main()
