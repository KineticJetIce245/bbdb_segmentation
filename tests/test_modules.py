"""Unit tests for the pure modules split out of segment_gui.py."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from bbdb_seg.transforms import apply_transform
from bbdb_seg.datasets import pending_indices, read_records, write_records
from bbdb_seg.image_ops import (
    apply_contrast,
    bright_background,
    build_threshold_mask,
    mask_diff,
    otsu_threshold,
)
from bbdb_seg.ops_log import append_operation, operations_path
from bbdb_seg.paths import find_data_root
from bbdb_seg.roi import Roi, base_to_display, roi_at_point, roi_screen_points, roi_screen_rect
from bbdb_seg.view_math import clamp_pan, fit_scale, max_zoom


class ImageOpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arr = np.full((60, 80), 220, dtype=np.uint8)
        self.arr[:, :40] = 30

    def test_otsu_and_bright_background(self) -> None:
        threshold = otsu_threshold(self.arr)
        self.assertEqual(threshold, 31)
        self.assertTrue(bright_background(self.arr, threshold))

    def test_threshold_mask_is_inclusive_and_invertible(self) -> None:
        sample = np.array([[99, 100, 150, 151]], dtype=np.uint8)
        inside = np.asarray(build_threshold_mask(sample, 100, 150, False))
        outside = np.asarray(build_threshold_mask(sample, 100, 150, True))
        np.testing.assert_array_equal(inside, np.array([[0, 255, 255, 0]], dtype=np.uint8))
        np.testing.assert_array_equal(outside, np.array([[255, 0, 0, 255]], dtype=np.uint8))

    def test_contrast_and_mask_diff(self) -> None:
        contrast = apply_contrast(np.array([[0, 128, 255]], dtype=np.uint8), 64, 192)
        np.testing.assert_array_equal(contrast, np.array([[0, 127, 255]], dtype=np.uint8))
        before = np.array([[0, 0, 255]], dtype=np.uint8)
        after = np.array([[0, 255, 255]], dtype=np.uint8)
        added, removed, bbox = mask_diff(before, after)
        self.assertEqual((added, removed), (1, 0))
        self.assertEqual(bbox, [0, 1, 1, 2])


class RoiTests(unittest.TestCase):
    def test_base_to_display_matches_flip_and_rotation(self) -> None:
        self.assertEqual(base_to_display(1, 2, 4, 5, "", 0), (1, 2))
        self.assertEqual(base_to_display(1, 2, 4, 5, "LR", 0), (1, 2))
        self.assertEqual(base_to_display(1, 2, 4, 5, "", 1), (2, 1))
        self.assertEqual(base_to_display(1, 2, 4, 5, "", 2), (2, 2))

    def test_roi_screen_rect_and_hit_test(self) -> None:
        roi = Roi(1, 10, 20, 30, 40, 0, 255, False)
        self.assertIs(roi_at_point([roi], 15, 25), roi)
        self.assertIsNone(roi_at_point([roi], 5, 5))
        rect = roi_screen_rect(roi, 100, 100, "", 0, 2.0, (10.0, 20.0))
        self.assertEqual(rect, (50.0, 40.0, 90.0, 80.0))

    def test_roi_bbox(self) -> None:
        roi = Roi(1, 1, 2, 3, 4, 0, 255, False)
        self.assertEqual(roi.bbox, [1, 2, 3, 4])

    def test_polygon_roi_contains_and_masks(self) -> None:
        points = [(10, 10), (10, 29), (29, 29), (29, 10)]
        roi = Roi(
            2, 10, 10, 30, 30, 0, 255, False,
            shape="polygon", points=points,
        )
        self.assertTrue(roi.contains(15, 15))
        self.assertFalse(roi.contains(5, 5))
        mask = roi.mask_array(40, 40)
        self.assertTrue(mask[15, 15])
        self.assertFalse(mask[5, 5])
        self.assertEqual(int(mask.sum()), 400)
        screen = roi_screen_points(roi, 40, 40, "", 0, 2.0, (10.0, 20.0))
        self.assertEqual(screen[0], (30.0, 40.0))
        self.assertEqual(len(screen), 4)


class ViewMathTests(unittest.TestCase):
    def test_fit_and_clamp(self) -> None:
        self.assertAlmostEqual(fit_scale(200, 100, 100, 100), 1.0)
        self.assertEqual(clamp_pan(500.0, 100, 50), 0.0)
        self.assertEqual(clamp_pan(1000.0, 100, 300), 100.0)
        self.assertEqual(clamp_pan(-1000.0, 100, 300), -100.0)

    def test_max_zoom_has_a_floor(self) -> None:
        self.assertGreaterEqual(max_zoom(100, 100, 80, 60), 1.0)


class PathAndLogTests(unittest.TestCase):
    def test_find_data_root_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(find_data_root(tmp), Path(tmp).resolve())

    def test_ops_log_appends_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seg_path = Path(tmp) / "masks" / "one.png"
            append_operation(seg_path, {"op": "draw", "added": 2})
            path = operations_path(seg_path)
            self.assertTrue(path.is_file())
            record = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(record["op"], "draw")
            self.assertEqual(record["added"], 2)
            self.assertIn("time", record)


class DatasetTests(unittest.TestCase):
    def test_read_queue_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "group.csv"
            csv_path.write_text(
                "group,specimen,species,stain,section,base,image,segmentation,flip,rotation\n"
                "g,s,Species,nissl,1,one,v.1/sections/g/one.png,segmentations/g/one.png,LR,90\n"
            )
            records = read_records(csv_path)
            self.assertEqual(records.at[0, "flip"], "LR")
            self.assertEqual(records.at[0, "rotation"], 1)
            self.assertEqual(pending_indices(records, root), [0])

            mask_path = root / "segmentations" / "g" / "one.png"
            mask_path.parent.mkdir(parents=True)
            mask_path.write_bytes(b"")
            self.assertEqual(pending_indices(records, root), [])

            out_path = root / "copy.csv"
            write_records(out_path, records)
            self.assertTrue(out_path.is_file())

    def test_apply_transform(self) -> None:
        arr = np.arange(6).reshape(2, 3)
        self.assertEqual(apply_transform(arr, "LR", 1).shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
