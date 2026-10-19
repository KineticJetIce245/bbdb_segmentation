"""Tests for ``--init-dir``: starting a section from a predicted mask.

The GUI must use a prediction only where nothing has been saved, and it must
accept the layouts people actually produce: the GUI's own
``segmentations/<group>/<base>.png`` tree, a plain ``<group>/<base>.png``
export, or the flat ``<group>__<base>.png`` that ``nnUNetv2_predict`` writes.
"""
from __future__ import annotations

import csv
import tempfile
import tkinter as tk
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bbdb_seg.app import SegmentGui

FIELDS = ["group", "specimen", "species", "stain", "section",
          "base", "image", "segmentation", "flip", "rotation"]


class InitDirTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="bbdb-init-tests-")
        root = Path(cls._tmp.name)
        cls.data_root = root / "brain_biodiversity_bank"
        section_dir = cls.data_root / "v.1" / "sections" / "test_group"
        section_dir.mkdir(parents=True)
        Image.fromarray(np.full((60, 80), 200, np.uint8), "L").save(
            section_dir / "one.png")
        cls.csv_path = root / "test_group.csv"
        with cls.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow({
                "group": "test_group", "specimen": "test",
                "species": "Testus testus", "stain": "nissl", "section": "1",
                "base": "one", "image": "v.1/sections/test_group/one.png",
                "segmentation": "segmentations/test_group/one.png",
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
            self.app.destroy()
            self.app = None

    def start(self, init_dir: Path) -> SegmentGui:
        try:
            app = SegmentGui(
                self.csv_path,
                data_root_override=str(self.data_root),
                seg_root_override=str(self.seg_root),
                init_dir_override=str(init_dir),
            )
        except tk.TclError as exc:  # no display, e.g. headless CI
            self.skipTest(f"Tk display unavailable: {exc}")
        self.app = app
        return app

    @staticmethod
    def write_mask(path: Path, box: tuple[slice, slice]) -> int:
        mask = np.zeros((60, 80), np.uint8)
        mask[box] = 1
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, "L").save(path)
        return int(mask.sum())

    def test_flat_case_name_is_used_when_no_mask_is_saved(self) -> None:
        init_dir = Path(self._tmp.name) / "init-flat"
        expected = self.write_mask(init_dir / "test_group__one.png",
                                   (slice(10, 20), slice(10, 20)))

        app = self.start(init_dir)
        self.assertEqual(app.mask_source, "prediction")
        self.assertEqual(int((np.asarray(app.base_mask_img) > 0).sum()), expected)

    def test_group_folder_layout_is_used(self) -> None:
        init_dir = Path(self._tmp.name) / "init-group"
        expected = self.write_mask(init_dir / "test_group" / "one.png",
                                   (slice(0, 5), slice(0, 5)))

        app = self.start(init_dir)
        self.assertEqual(app.mask_source, "prediction")
        self.assertEqual(int((np.asarray(app.base_mask_img) > 0).sum()), expected)

    def test_csv_layout_is_used(self) -> None:
        init_dir = Path(self._tmp.name) / "init-csv"
        expected = self.write_mask(
            init_dir / "segmentations" / "test_group" / "one.png",
            (slice(0, 4), slice(0, 5)))

        app = self.start(init_dir)
        self.assertEqual(app.mask_source, "prediction")
        self.assertEqual(int((np.asarray(app.base_mask_img) > 0).sum()), expected)

    def test_saved_mask_wins_over_the_prediction(self) -> None:
        init_dir = Path(self._tmp.name) / "init-saved"
        self.write_mask(init_dir / "test_group" / "one.png",
                        (slice(0, 5), slice(0, 5)))
        saved = self.write_mask(
            self.seg_root / "segmentations" / "test_group" / "one.png",
            (slice(50, 60), slice(70, 80)))

        app = self.start(init_dir)
        self.assertEqual(app.mask_source, "saved")
        self.assertEqual(int((np.asarray(app.base_mask_img) > 0).sum()), saved)

    def test_without_init_dir_the_threshold_is_used(self) -> None:
        app = self.start(Path(self._tmp.name) / "does-not-exist")
        self.assertEqual(app.mask_source, "threshold")


if __name__ == "__main__":
    unittest.main()
