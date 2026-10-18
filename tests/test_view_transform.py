"""Tests for the segment_gui zoom/pan view transform.

Run from the repository root:

    python -m unittest discover -s tests -v

The GUI tests create a tiny synthetic section in a temporary directory.  They
skip themselves when Tk cannot connect to a display.
"""
from __future__ import annotations

import csv
import json
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

from bbdb_seg.app import SegmentGui  # noqa: E402
from bbdb_seg.settings import (  # noqa: E402
    CONTROL_MASK,
    MAX_BRUSH,
    MAX_VIEW_PIXELS,
    MIN_BRUSH,
    MIN_ZOOM,
    SHIFT_MASK,
    ZOOM_STEP,
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
        app.cancel_slider_updates()
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

    def test_overlay_is_clipped_to_visible_canvas_when_zoomed(self) -> None:
        x, y = self.center()
        self.app.zoom_at(x, y, 4.0)
        self.app.render_all()

        canvas_w, canvas_h = self.app.canvas_size()
        self.assertLessEqual(self.app.ov_photo.width(), canvas_w + 2)
        self.assertLessEqual(self.app.ov_photo.height(), canvas_h + 2)

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

    def test_direct_brush_entry_clamps_and_updates(self) -> None:
        self.app.brush_input_var.set("37")
        self.app.on_brush_entry()
        self.assertEqual(self.app.brush, 37)
        self.assertEqual(self.app.brush_input_var.get(), "37")

        self.app.brush_input_var.set("999")
        self.app.on_brush_entry()
        self.assertEqual(self.app.brush, MAX_BRUSH)
        self.assertEqual(self.app.brush_input_var.get(), str(MAX_BRUSH))

        self.app.brush_input_var.set("not a number")
        self.app.on_brush_entry()
        self.assertEqual(self.app.brush, MAX_BRUSH)
        self.assertEqual(self.app.brush_input_var.get(), str(MAX_BRUSH))

        self.app.brush_input_var.set("1")
        self.app.on_brush_entry()
        self.assertEqual(self.app.brush, MIN_BRUSH)

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

    def test_threshold_range_is_inclusive_and_invert_selects_outside(self) -> None:
        self.app.base_img = np.array([[99, 100, 150, 151]], dtype=np.uint8)
        self.app.set_threshold_range(100, 150)

        self.app.invert_var.set(False)
        inside = np.asarray(self.app.threshold_mask())
        np.testing.assert_array_equal(inside, np.array([[0, 255, 255, 0]], dtype=np.uint8))

        self.app.invert_var.set(True)
        outside = np.asarray(self.app.threshold_mask())
        np.testing.assert_array_equal(outside, np.array([[255, 0, 0, 255]], dtype=np.uint8))

    def test_threshold_range_crossing_clamps(self) -> None:
        self.app.set_threshold_range(100, 150)
        self.app.on_threshold_min_change("200")
        self.assertEqual((self.app.threshold_min, self.app.threshold_max), (200, 200))

        self.app.flush_slider_updates()
        self.app.set_threshold_range(100, 150)
        self.app.on_threshold_max_change("50")
        self.assertEqual((self.app.threshold_min, self.app.threshold_max), (50, 50))
        self.app.flush_slider_updates()

    def test_threshold_callbacks_are_coalesced(self) -> None:
        before = np.asarray(self.app.base_mask_img).copy()
        self.app.on_threshold_min_change("10")
        self.app.on_threshold_min_change("20")

        self.assertTrue(self.app.threshold_pending)
        self.assertIsNotNone(self.app.slider_after_id)
        np.testing.assert_array_equal(before, np.asarray(self.app.base_mask_img))

        self.app.flush_slider_updates()
        self.assertFalse(self.app.threshold_pending)
        self.assertIsNone(self.app.slider_after_id)
        self.assertEqual(self.app.threshold_min, 20)
        np.testing.assert_array_equal(
            np.asarray(self.app.threshold_mask()),
            np.asarray(self.app.base_mask_img),
        )

    def test_painting_flushes_pending_threshold_first(self) -> None:
        self.app.on_threshold_min_change("10")
        self.assertTrue(self.app.threshold_pending)

        x, y = self.center()
        self.app.on_press(make_event(x, y), right=False)

        self.assertFalse(self.app.threshold_pending)
        self.assertIsNone(self.app.slider_after_id)

    def test_painting_coalesces_overlay_updates(self) -> None:
        x, y = self.center()
        self.app.on_press(make_event(x, y), right=False)
        self.app.on_drag(make_event(x + 5, y + 5), right=False)

        self.assertTrue(self.app.overlay_pending)
        self.assertIsNotNone(self.app.slider_after_id)
        # The mask itself is updated immediately; only the overlay is deferred.
        point = self.app.canvas_to_base(x + 5, y + 5)
        self.assertEqual(int(np.asarray(self.app.base_mask_img)[point]), 255)

        self.app.end_stroke(make_event(x + 5, y + 5))
        self.assertFalse(self.app.overlay_pending)
        self.assertIsNone(self.app.slider_after_id)

    def test_alpha_and_contrast_handlers_coalesce(self) -> None:
        self.app.on_alpha_change("80")
        self.assertEqual(self.app.overlay_alpha, 80)
        self.assertTrue(self.app.alpha_pending)

        self.app.min_scale.set(20)
        self.app.max_scale.set(240)
        self.app.on_contrast_change()
        self.assertEqual((self.app.contrast_min, self.app.contrast_max), (20, 240))
        self.assertTrue(self.app.contrast_pending)

        self.app.flush_slider_updates()
        self.assertFalse(self.app.alpha_pending)
        self.assertFalse(self.app.contrast_pending)

    def test_otsu_reset_turns_invert_off_and_restores_initial_band(self) -> None:
        self.app.set_threshold_range(100, 150)
        self.app.invert_var.set(True)
        self.app.reset_threshold()

        expected = (0, self.app.otsu) if self.app.bright_border else (self.app.otsu, 255)
        self.assertEqual((self.app.threshold_min, self.app.threshold_max), expected)
        self.assertFalse(self.app.invert_var.get())

    def test_initial_band_selects_the_non_background_side(self) -> None:
        expected = (0, self.app.otsu) if self.app.bright_border else (self.app.otsu, 255)
        self.assertEqual((self.app.threshold_min, self.app.threshold_max), expected)
        self.assertFalse(self.app.invert_var.get())

    def test_auto_contrast_applies_immediately(self) -> None:
        self.app.auto_contrast()
        expected = (int(self.app.base_img.min()), int(self.app.base_img.max()))
        self.assertEqual((self.app.contrast_min, self.app.contrast_max), expected)
        self.assertIsNone(self.app.slider_after_id)
        self.assertFalse(self.app.contrast_pending)

    def test_roi_create_activates_without_changing_mask(self) -> None:
        before = np.asarray(self.app.base_mask_img).copy()
        self.app.create_roi(10, 10, 30, 30)

        self.assertEqual(len(self.app.rois), 1)
        self.assertEqual(self.app.active_roi_id, 1)
        np.testing.assert_array_equal(before, np.asarray(self.app.base_mask_img))

        path = self.app.ops_path(self.app.index)
        self.assertTrue(path.is_file())
        record = json.loads(path.read_text().splitlines()[-1])
        self.assertEqual(record["op"], "roi_create")
        self.assertEqual(record["roi"], [10, 10, 30, 30])

    def test_threshold_updates_only_active_roi(self) -> None:
        base = np.zeros((60, 80), dtype=np.uint8)
        base[0:10, 0:10] = 255
        self.app.base_mask_img = Image.fromarray(base)
        self.app.create_roi(0, 40, 60, 80)

        self.app.set_threshold_range(0, 255)
        self.app.invert_var.set(False)
        self.app.apply_threshold()

        after = np.asarray(self.app.base_mask_img)
        self.assertTrue(np.all(after[:, 40:] == 255))
        np.testing.assert_array_equal(
            after[0:10, 0:10],
            np.full((10, 10), 255, dtype=np.uint8),
        )
        self.assertTrue(np.all(after[10:, :40] == 0))

    def test_roi_activation_loads_its_threshold(self) -> None:
        self.app.create_roi(5, 5, 20, 20)
        self.app.set_threshold_range(10, 100)
        self.app.create_roi(30, 30, 50, 60)
        self.app.set_threshold_range(120, 200)

        self.app.activate_roi(1)
        self.assertEqual((self.app.threshold_min, self.app.threshold_max), (10, 100))
        self.app.activate_roi(2)
        self.assertEqual((self.app.threshold_min, self.app.threshold_max), (120, 200))

    def test_delete_active_roi_restores_global_threshold(self) -> None:
        global_range = (self.app.threshold_min, self.app.threshold_max)
        self.app.create_roi(5, 5, 20, 20)
        self.app.set_threshold_range(10, 100)
        before = np.asarray(self.app.base_mask_img).copy()

        self.app.delete_active_roi()

        self.assertIsNone(self.app.active_roi_id)
        self.assertEqual(self.app.rois, [])
        self.assertEqual(
            (self.app.threshold_min, self.app.threshold_max),
            global_range,
        )
        np.testing.assert_array_equal(before, np.asarray(self.app.base_mask_img))

    def test_backspace_deletes_active_roi(self) -> None:
        self.app.create_roi(5, 5, 20, 20)
        self.app.on_key(make_event(keysym="BackSpace"))
        self.assertEqual(self.app.rois, [])
        self.assertIsNone(self.app.active_roi_id)

    def test_otsu_reset_updates_active_roi_not_global(self) -> None:
        global_range = (self.app.threshold_min, self.app.threshold_max)
        self.app.create_roi(5, 5, 20, 20)
        self.app.set_threshold_range(100, 150)
        self.app.invert_var.set(True)

        self.app.reset_threshold()

        roi = self.app.rois[0]
        expected = (0, self.app.otsu) if self.app.bright_border else (self.app.otsu, 255)
        self.assertEqual((roi.threshold_min, roi.threshold_max), expected)
        self.assertFalse(roi.invert)
        self.assertEqual(
            (self.app.global_threshold_min, self.app.global_threshold_max),
            global_range,
        )

    def test_ops_log_records_selection_and_threshold(self) -> None:
        self.app.create_roi(0, 40, 60, 80)
        self.app.set_threshold_range(0, 255)
        self.app.apply_threshold()

        path = self.app.ops_path(self.app.index)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        ops = [record["op"] for record in records]
        self.assertIn("roi_create", ops)
        self.assertIn("threshold", ops)

        threshold = next(record for record in records if record["op"] == "threshold")
        self.assertEqual(threshold["roi_id"], 1)
        self.assertEqual(threshold["threshold"], {"min": 0, "max": 255, "invert": False})
        self.assertGreater(threshold["added"], 0)
        self.assertIsNotNone(threshold["bbox"])

    def test_ops_autosave_without_png(self) -> None:
        self.app.base_mask_img = Image.fromarray(
            np.zeros((60, 80), dtype=np.uint8)
        )
        x, y = self.center()
        self.app.on_press(make_event(x, y), right=False)
        self.app.end_stroke(make_event(x, y))

        self.assertTrue(self.app.mask_dirty)
        self.assertFalse(self.app.seg_path(self.app.index).exists())
        path = self.app.ops_path(self.app.index)
        self.assertTrue(path.is_file())
        record = json.loads(path.read_text().splitlines()[-1])
        self.assertEqual(record["op"], "draw")
        self.assertGreater(record["added"], 0)
        self.assertIsNotNone(record["bbox"])

        self.app.mask_dirty = True
        self.app.save_mask()
        self.assertTrue(self.app.seg_path(self.app.index).exists())

    def test_shift_drag_creates_roi_and_shift_click_reactivates(self) -> None:
        offset_x, offset_y = self.app.display_offset
        scale = self.app.display_scale
        start = (int(offset_x + 20 * scale), int(offset_y + 20 * scale))
        end = (start[0] + 40, start[1] + 30)

        self.app.start_roi_select(make_event(*start))
        self.app.drag_roi_select(make_event(*end))
        self.app.end_roi_select(make_event(*end))

        self.assertEqual(len(self.app.rois), 1)
        self.assertEqual(self.app.active_roi_id, 1)
        roi = self.app.rois[0]
        rect = self.app._roi_screen_rect(roi)
        centre = (int((rect[0] + rect[2]) / 2), int((rect[1] + rect[3]) / 2))

        self.app.deactivate_roi()
        self.assertIsNone(self.app.active_roi_id)

        self.app.start_roi_select(make_event(*centre))
        self.app.end_roi_select(make_event(*centre))
        self.assertEqual(self.app.active_roi_id, roi.roi_id)

    def test_shift_right_generic_callbacks_start_lasso_not_erase(self) -> None:
        x, y = self.center()
        self.app.on_press(make_event(x, y, state=SHIFT_MASK), right=True)
        self.assertTrue(self.app.lasso_dragging)
        self.assertIsNone(self.app.stroke_before)
        self.assertFalse(self.app.roi_dragging)

        self.app.on_drag(make_event(x + 10, y + 10, state=SHIFT_MASK), right=True)
        self.app.end_stroke(make_event(x + 10, y + 10, state=SHIFT_MASK))
        self.assertFalse(self.app.lasso_dragging)
        self.assertIsNone(self.app.stroke_before)

    def test_shift_right_lasso_creates_polygon_roi(self) -> None:
        offset_x, offset_y = self.app.display_offset
        scale = self.app.display_scale

        def screen(row: int, col: int) -> tuple[int, int]:
            disp_row, disp_col = self.app._base_to_display(row, col)
            return (
                int(offset_x + disp_col * scale),
                int(offset_y + disp_row * scale),
            )

        points = [(10, 45), (10, 65), (30, 65), (30, 45)]
        screen_points = [screen(*point) for point in points]
        self.app.start_lasso_select(make_event(*screen_points[0]))
        for point in screen_points[1:]:
            self.app.drag_lasso_select(make_event(*point))
        self.app.end_lasso_select(make_event(*screen_points[-1]))

        self.assertEqual(len(self.app.rois), 1)
        self.assertEqual(self.app.active_roi_id, 1)
        roi = self.app.rois[0]
        self.assertEqual(roi.shape, "polygon")
        self.assertGreaterEqual(len(roi.points or []), 3)
        self.assertTrue(roi.contains(20, 55))
        self.assertFalse(roi.contains(20, 5))

        # Threshold must touch only the polygon interior.
        before = np.asarray(self.app.base_mask_img).copy()
        self.app.set_threshold_range(0, 255)
        self.app.invert_var.set(False)
        self.app.apply_threshold()
        after = np.asarray(self.app.base_mask_img)
        mask = roi.mask_array(*before.shape)
        self.assertTrue(np.all(after[mask] == 255))
        np.testing.assert_array_equal(after[~mask], before[~mask])

        record = json.loads(self.app.ops_path(self.app.index).read_text().splitlines()[0])
        self.assertEqual(record["op"], "roi_create")
        self.assertEqual(record["roi_shape"], "polygon")
        self.assertGreaterEqual(len(record["roi_points"]), 3)

        # Shift+right click inside the polygon re-activates it.
        self.app.deactivate_roi()
        centre = screen(20, 55)
        self.app.start_lasso_select(make_event(*centre))
        self.app.end_lasso_select(make_event(*centre))
        self.assertEqual(self.app.active_roi_id, roi.roi_id)

    def test_shift_right_binding_is_registered(self) -> None:
        for sequence, handler in (
            ("<Shift-ButtonPress-3>", "start_lasso_select"),
            ("<Shift-B3-Motion>", "drag_lasso_select"),
            ("<Shift-ButtonRelease-3>", "end_lasso_select"),
        ):
            self.assertIn(handler, self.app.canvas.bind(sequence))

    def test_brush_select_unions_multiple_strokes_into_one_roi(self) -> None:
        self.app.base_mask_img = Image.fromarray(
            np.zeros((60, 80), dtype=np.uint8)
        )
        before = np.asarray(self.app.base_mask_img).copy()
        self.app.set_tool("select")
        x, y = self.center()

        self.app.start_brush_select(make_event(x, y))
        self.assertTrue(self.app.brush_select_dragging)
        self.assertTrue(self.app.brush_select_items)
        self.app.drag_brush_select(make_event(x + 20, y))
        self.app.end_brush_select(make_event(x + 20, y))
        self.assertEqual(len(self.app.rois), 1)
        self.assertEqual(self.app.rois[0].shape, "brush")
        self.assertIsNotNone(self.app.brush_preview_item)
        self.app._finish_brush_preview()
        self.assertIsNone(self.app.brush_preview_item)
        self.assertTrue(self.app.roi_photos)

        first_points = len(self.app.rois[0].points or [])
        self.app.start_brush_select(make_event(x, y + 30))
        self.app.drag_brush_select(make_event(x + 20, y + 30))
        self.app.end_brush_select(make_event(x + 20, y + 30))
        self.assertEqual(len(self.app.rois), 1)
        roi = self.app.rois[0]
        self.assertGreater(len(roi.points or []), first_points)
        np.testing.assert_array_equal(before, np.asarray(self.app.base_mask_img))

        records = [
            json.loads(line)
            for line in self.app.ops_path(self.app.index).read_text().splitlines()
        ]
        self.assertEqual(records[0]["op"], "roi_create")
        self.assertIn("roi_extend", [record["op"] for record in records])
        self.assertEqual(records[0]["roi_shape"], "brush")

    def test_brush_select_right_drag_erases_selection_not_mask(self) -> None:
        self.app.base_mask_img = Image.fromarray(
            np.zeros((60, 80), dtype=np.uint8)
        )
        before_mask = np.asarray(self.app.base_mask_img).copy()
        self.app.set_tool("select")
        self.app.set_brush(20)
        offset_x, offset_y = self.app.display_offset
        scale = self.app.display_scale

        def screen(row: int, col: int) -> tuple[int, int]:
            disp_row, disp_col = self.app._base_to_display(row, col)
            return (
                int(offset_x + disp_col * scale),
                int(offset_y + disp_row * scale),
            )

        start = screen(30, 20)
        end = screen(30, 60)
        self.app.start_brush_select(make_event(*start))
        self.app.drag_brush_select(make_event(*end))
        self.app.end_brush_select(make_event(*end))
        self.app._finish_brush_preview()
        roi = self.app.rois[0]
        area_before = int(roi.mask.sum()) if roi.mask is not None else 0

        erase_start = screen(30, 20)
        erase_end = screen(30, 35)
        self.app.on_press(make_event(*erase_start), right=True)
        self.assertTrue(self.app.brush_erase_dragging)
        self.app.on_drag(make_event(*erase_end), right=True)
        self.app.end_stroke(make_event(*erase_end))

        self.assertEqual(len(self.app.rois), 1)
        self.assertLess(int(roi.mask.sum()), area_before)
        np.testing.assert_array_equal(before_mask, np.asarray(self.app.base_mask_img))
        records = [
            json.loads(line)
            for line in self.app.ops_path(self.app.index).read_text().splitlines()
        ]
        self.assertIn("roi_erase", [record["op"] for record in records])

    def test_brush_select_right_drag_can_remove_whole_roi(self) -> None:
        self.app.base_mask_img = Image.fromarray(
            np.zeros((60, 80), dtype=np.uint8)
        )
        before_mask = np.asarray(self.app.base_mask_img).copy()
        self.app.set_tool("select")
        self.app.set_brush(20)
        offset_x, offset_y = self.app.display_offset
        scale = self.app.display_scale
        disp_row, disp_col = self.app._base_to_display(30, 40)
        point = (
            int(offset_x + disp_col * scale),
            int(offset_y + disp_row * scale),
        )

        self.app.start_brush_select(make_event(*point))
        self.app.end_brush_select(make_event(*point))
        self.app._finish_brush_preview()
        self.assertEqual(len(self.app.rois), 1)

        self.app.on_press(make_event(*point), right=True)
        self.app.end_stroke(make_event(*point))

        self.assertEqual(self.app.rois, [])
        self.assertIsNone(self.app.active_roi_id)
        np.testing.assert_array_equal(before_mask, np.asarray(self.app.base_mask_img))
        records = [
            json.loads(line)
            for line in self.app.ops_path(self.app.index).read_text().splitlines()
        ]
        self.assertIn("roi_delete", [record["op"] for record in records])

    def test_threshold_applies_inside_brush_roi_only(self) -> None:
        self.app.base_mask_img = Image.fromarray(
            np.zeros((60, 80), dtype=np.uint8)
        )
        self.app.set_tool("select")
        x, y = self.center()
        self.app.start_brush_select(make_event(x, y))
        self.app.end_brush_select(make_event(x, y))
        self.app._finish_brush_preview()

        roi = self.app.rois[0]
        self.app.set_threshold_range(0, 255)
        self.app.apply_threshold()

        after = np.asarray(self.app.base_mask_img)
        mask = roi.mask_array(*after.shape)
        self.assertTrue(np.all(after[mask] == 255))
        self.assertTrue(np.all(after[~mask] == 0))

    def test_l_key_toggles_brush_select(self) -> None:
        self.app.on_key(make_event(keysym="l", char="l"))
        self.assertEqual(self.app.tool, "select")
        self.app.on_key(make_event(keysym="l", char="l"))
        self.assertEqual(self.app.tool, "draw")

    def test_toolbar_is_split_into_two_rows(self) -> None:
        toolbar = self.app.grid_slaves(row=0, column=0)[0]
        rows = toolbar.winfo_children()
        self.assertEqual(len(rows), 2)
        self.assertLessEqual(toolbar.winfo_reqwidth(), 1450)

    def test_review_finished_checkbox_rebuilds_queue(self) -> None:
        # Add a second pending section to the in-memory records.
        extra = self.app.records.iloc[0].copy()
        extra["base"] = "two"
        extra["section"] = "2"
        extra["image"] = "v.1/sections/test_group/two.png"
        extra["segmentation"] = "segmentations/test_group/two.png"
        extra["flip"] = ""
        extra["rotation"] = 0
        self.app.records.loc[len(self.app.records)] = extra
        Image.fromarray(np.full((60, 80), 100, dtype=np.uint8), "L").save(
            self.data_root / "v.1" / "sections" / "test_group" / "two.png"
        )

        self.app.mask_dirty = True
        self.app.save_mask()
        self.assertEqual(self.app.queue, [0])

        self.app.review_finished_var.set(True)
        self.app.on_review_finished_change()
        self.assertTrue(self.app.review_all)
        self.assertEqual(self.app.queue, [0, 1])
        self.assertEqual(self.app.index, 0)

        self.app.review_finished_var.set(False)
        self.app.on_review_finished_change()
        self.assertFalse(self.app.review_all)
        self.assertEqual(self.app.queue, [1])
        self.assertEqual(self.app.index, 1)

    def test_roi_display_mapping_matches_disp_index(self) -> None:
        self.app.create_roi(5, 6, 15, 20)
        base_w = self.app.base_img.shape[1]
        for flip, rotation in (
            ("", 0), ("LR", 0), ("UP", 0), ("LR+UP", 1), ("", 2), ("", 3),
        ):
            self.app.flip = flip
            self.app.rotation = rotation
            self.app.render_all()
            for row, col in ((5, 6), (14, 19), (5, 19), (14, 6)):
                disp_row, disp_col = self.app._base_to_display(row, col)
                self.assertEqual(
                    int(self.app.disp_index[disp_row, disp_col]),
                    row * base_w + col,
                )


class PanClampTests(unittest.TestCase):
    def test_centres_when_image_is_smaller_than_canvas(self) -> None:
        self.assertEqual(SegmentGui.clamp_pan(500.0, 100, 50), 0.0)

    def test_clamps_to_image_edges_when_image_is_larger(self) -> None:
        self.assertEqual(SegmentGui.clamp_pan(1000.0, 100, 300), 100.0)
        self.assertEqual(SegmentGui.clamp_pan(-1000.0, 100, 300), -100.0)
        self.assertEqual(SegmentGui.clamp_pan(25.0, 100, 300), 25.0)


if __name__ == "__main__":
    unittest.main()
