"""Tests for the optional UNet prediction channel in the GUI.

The model itself is not started here: these tests cover the state machine
(cache, auto-apply, stale results, explicit overwrite) with a fake prediction
folder.  The worker protocol is exercised by the manual smoke test.
"""
from __future__ import annotations

import csv
import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from bbdb_seg.app import SegmentGui
from bbdb_seg.settings import UNET_BADGE_COLOUR

FIELDS = ["group", "specimen", "species", "stain", "section",
          "base", "image", "segmentation", "flip", "rotation"]


class UnetGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="bbdb-unet-tests-")
        cls.root = Path(cls._tmp.name)
        cls.data_root = cls.root / "brain_biodiversity_bank"
        section_dir = cls.data_root / "v.1" / "sections" / "test_group"
        section_dir.mkdir(parents=True)
        for name in ("one", "two"):
            image = np.full((60, 80), 200, dtype=np.uint8)
            image[:, :40] = 30
            Image.fromarray(image, "L").save(section_dir / f"{name}.png")
        cls.csv_path = cls.root / "test_group.csv"
        with cls.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for section, name in ((1, "one"), (2, "two")):
                writer.writerow({
                    "group": "test_group", "specimen": "test",
                    "species": "Testus testus", "stain": "nissl",
                    "section": str(section), "base": name,
                    "image": f"v.1/sections/test_group/{name}.png",
                    "segmentation": f"segmentations/test_group/{name}.png",
                    "flip": "", "rotation": "0",
                })

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.seg_root = Path(self._tmp.name) / f"seg-{self._testMethodName}"
        self.seg_root.mkdir()
        self.app: SegmentGui | None = None

    def tearDown(self) -> None:
        if self.app is not None:
            self.app.cancel_slider_updates()
            self.app.shutdown_unet(render=False)
            self.app.destroy()
            self.app = None

    def start(self) -> SegmentGui:
        try:
            app = SegmentGui(
                self.csv_path,
                data_root_override=str(self.data_root),
                seg_root_override=str(self.seg_root),
            )
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.app = app
        return app

    def write_mask(self, path: Path, box: tuple[slice, slice]) -> int:
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[box] = 1
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, "L").save(path)
        return int(mask.sum())

    def cache_path(self, name: str = "one") -> Path:
        return (self.seg_root / "segmentations" / "UNet_output"
                / "test_group" / f"{name}.png")

    # --- storage ------------------------------------------------------------- #

    def test_output_path_lives_under_segmentations_unet_output(self) -> None:
        app = self.start()
        expected = (self.seg_root / "segmentations" / "UNet_output"
                    / "test_group" / "one.png").resolve()
        self.assertEqual(app.unet_output_path(0), expected)

    # --- automatic predictions ------------------------------------------------ #

    def test_cached_prediction_is_used_as_yellow_start(self) -> None:
        expected = self.write_mask(self.cache_path(), (slice(10, 20), slice(10, 20)))
        app = self.start()
        app.unet_enabled = True
        app.on_section_loaded()

        self.assertEqual(app.mask_source, "unet")
        self.assertTrue(app.prediction_applied)
        self.assertTrue(app.prediction_yellow)
        self.assertTrue(app.prediction_active())
        self.assertEqual(int((np.asarray(app.base_mask_img) > 0).sum()), expected)
        self.assertEqual(app.done_var.get(), "\u25cb predicted")
        self.assertEqual(str(app.done_label.cget("fg")), UNET_BADGE_COLOUR)
        self.assertNotIn("disabled", app.use_prediction_button.state())
        self.assertEqual(str(app.use_prediction_button.cget("text")), "Save prediction")

    def test_saved_mask_wins_over_cached_prediction(self) -> None:
        self.write_mask(self.seg_root / "segmentations" / "test_group" / "one.png",
                        (slice(40, 50), slice(60, 70)))
        self.write_mask(self.cache_path(), (slice(10, 20), slice(10, 20)))
        app = self.start()
        app.unet_enabled = True
        app.on_section_loaded()

        self.assertEqual(app.mask_source, "saved")
        self.assertIsNone(app.prediction_img)
        self.assertFalse(app.prediction_active())
        self.assertNotIn("predicted", app.done_var.get())

    def test_missing_cache_triggers_one_request(self) -> None:
        app = self.start()
        app.unet_enabled = True
        calls: list[tuple] = []
        app.request_unet = lambda index=None, force=False, apply_if_empty=False: calls.append(
            (index, force, apply_if_empty))
        app.on_section_loaded()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], app.index)
        self.assertTrue(calls[0][2])
        self.assertFalse(calls[0][1])

    # --- explicit overwrite --------------------------------------------------- #

    def test_use_prediction_replaces_mask_and_undo_restores(self) -> None:
        app = self.start()
        before = np.asarray(app.base_mask_img).copy()
        prediction = np.zeros_like(before)
        prediction[5:25, 5:25] = 255
        app.prediction_img = Image.fromarray(prediction)
        app.prediction_index = app.index
        app.prediction_applied = False

        app.use_prediction()
        self.assertEqual(app.mask_source, "unet")
        self.assertFalse(app.prediction_yellow)   # accepted: no longer a draft
        self.assertEqual(app.done_var.get(), "\u25cb predicted")
        self.assertTrue(app.mask_dirty)
        self.assertEqual(int((np.asarray(app.base_mask_img) > 0).sum()),
                         int((prediction > 0).sum()))
        self.assertNotIn("disabled", app.use_prediction_button.state())
        self.assertEqual(str(app.use_prediction_button.cget("text")), "Save prediction")

        # A second press accepts the prediction: the mask is written, the
        # draft state is dropped and the section becomes segmented.
        app.use_prediction()
        self.assertTrue(app.seg_path(app.index).exists())
        self.assertIsNone(app.prediction_img)
        self.assertIn("disabled", app.use_prediction_button.state())
        self.assertEqual(app.done_var.get(), "\u25cf segmented")

        app.undo()
        self.assertEqual(app.mask_source, "edited")
        self.assertTrue(np.array_equal(np.asarray(app.base_mask_img), before))

    def test_use_prediction_is_disabled_until_a_prediction_exists(self) -> None:
        app = self.start()
        self.assertIn("disabled", app.use_prediction_button.state())
        app.prediction_img = Image.fromarray(np.zeros((60, 80), np.uint8))
        app.prediction_index = app.index
        app.prediction_applied = False
        app._update_prediction_controls()
        self.assertNotIn("disabled", app.use_prediction_button.state())

    # --- cached prediction buttons -------------------------------------------- #

    def test_load_prediction_uses_cache_without_running_the_model(self) -> None:
        expected = self.write_mask(self.cache_path(), (slice(10, 20), slice(10, 20)))
        app = self.start()
        self.assertFalse(app.unet_enabled)
        self.assertNotIn("disabled", app.load_prediction_button.state())

        app.load_cached_prediction()
        self.assertEqual(app.mask_source, "unet")
        self.assertTrue(app.prediction_applied)
        self.assertTrue(app.prediction_yellow)
        self.assertEqual(app.done_var.get(), "\u25cb predicted")
        self.assertEqual(int((np.asarray(app.base_mask_img) > 0).sum()), expected)

    def test_load_prediction_on_saved_mask_is_only_a_preview(self) -> None:
        saved = self.write_mask(
            self.seg_root / "segmentations" / "test_group" / "one.png",
            (slice(40, 50), slice(60, 70)))
        self.write_mask(self.cache_path(), (slice(10, 20), slice(10, 20)))
        app = self.start()

        app.load_cached_prediction()
        self.assertEqual(app.mask_source, "saved")
        self.assertFalse(app.prediction_applied)
        self.assertTrue(app.prediction_separate())
        self.assertEqual(int((np.asarray(app.base_mask_img) > 0).sum()), saved)

    def test_clear_prediction_deletes_only_the_cached_file(self) -> None:
        self.write_mask(self.cache_path(), (slice(10, 20), slice(10, 20)))
        app = self.start()
        app.load_cached_prediction()
        before = int((np.asarray(app.base_mask_img) > 0).sum())

        app.clear_cached_prediction()
        self.assertFalse(self.cache_path().exists())
        self.assertIsNone(app.prediction_img)
        self.assertEqual(int((np.asarray(app.base_mask_img) > 0).sum()), before)
        self.assertNotIn(app.done_var.get(), ("\u25cb predicted",))
        self.assertIn("disabled", app.clear_prediction_button.state())

    def test_load_prediction_without_a_cache_reports_it(self) -> None:
        app = self.start()
        app.load_cached_prediction()
        self.assertIsNone(app.prediction_img)
        self.assertIn("No cached prediction", app.status_var.get())

    # --- colours -------------------------------------------------------------- #

    def test_overlay_paints_manual_mask_red_and_prediction_yellow(self) -> None:
        app = self.start()
        app.bg_visible = None
        app.bg_place = None
        manual = Image.new("L", (10, 10), 0)
        manual.putpixel((1, 1), 255)
        predicted = Image.new("L", (10, 10), 0)
        predicted.putpixel((5, 5), 255)

        app._set_overlay_photo(manual, 0, 0, None, predicted)
        self.assertEqual(app.ov_image.getpixel((1, 1))[:3], (255, 40, 40))
        self.assertEqual(app.ov_image.getpixel((5, 5))[:3], (255, 205, 0))

    def test_applied_prediction_uses_yellow_for_the_base_mask(self) -> None:
        app = self.start()
        app.bg_visible = None
        app.bg_place = None
        app.mask_source = "unet"
        app.prediction_yellow = True
        stencil = Image.new("L", (10, 10), 0)
        stencil.putpixel((2, 2), 255)

        app._set_overlay_photo(stencil, 0, 0, None, None)
        self.assertEqual(app.ov_image.getpixel((2, 2))[:3], (255, 205, 0))

    def test_accepted_prediction_is_red_but_keeps_predicted_status(self) -> None:
        app = self.start()
        app.bg_visible = None
        app.bg_place = None
        app.mask_source = "unet"
        app.prediction_yellow = False
        stencil = Image.new("L", (10, 10), 0)
        stencil.putpixel((2, 2), 255)

        app._set_overlay_photo(stencil, 0, 0, None, None)
        self.assertEqual(app.ov_image.getpixel((2, 2))[:3], (255, 40, 40))

    # --- stale results and edits ---------------------------------------------- #

    def test_stale_worker_result_does_not_touch_the_current_section(self) -> None:
        app = self.start()
        stale = self.write_mask(self.cache_path("two"), (slice(0, 5), slice(0, 5)))
        request_id = 7
        output = app.unet_output_path(1)
        app.unet_pending[request_id] = (1, output, True)
        app._handle_unet_line(json.dumps(
            {"status": "ok", "id": request_id, "mask": str(output)}))

        self.assertIsNone(app.prediction_img)
        self.assertEqual(app.mask_source, "threshold")
        self.assertNotEqual(int((np.asarray(app.base_mask_img) > 0).sum()), stale)

    def test_first_edit_restores_not_segmented_status(self) -> None:
        self.write_mask(self.cache_path(), (slice(10, 20), slice(10, 20)))
        app = self.start()
        app.unet_enabled = True
        app.on_section_loaded()
        self.assertEqual(app.done_var.get(), "\u25cb predicted")

        app.mask_source = "edited"
        app.mark_edited()
        self.assertFalse(app.prediction_active())
        self.assertEqual(app.done_var.get(), "\u25cb not segmented")
        self.assertEqual(str(app.done_label.cget("fg")), "#c0392b")

    def test_plain_save_also_finalizes_the_draft(self) -> None:
        self.write_mask(self.cache_path(), (slice(10, 20), slice(10, 20)))
        app = self.start()
        app.unet_enabled = True
        app.on_section_loaded()
        self.assertTrue(app.prediction_yellow)

        app.mask_dirty = True
        app.flush_mask()
        self.assertIsNone(app.prediction_img)
        self.assertEqual(app.done_var.get(), "\u25cf segmented")

    def test_worker_error_is_reported_without_replacing_the_mask(self) -> None:
        app = self.start()
        before = np.asarray(app.base_mask_img).copy()
        app._handle_unet_line(json.dumps(
            {"status": "error", "id": 3, "error": "boom"}))
        self.assertTrue(app.unet_status.startswith("error"))
        self.assertTrue(np.array_equal(np.asarray(app.base_mask_img), before))


if __name__ == "__main__":
    unittest.main()
