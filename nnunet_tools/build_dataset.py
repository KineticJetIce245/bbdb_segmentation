"""Turn the GUI's masks into an nnU-Net v2 2D dataset.

The GUI stores one mask per section as ``segmentations/<group>/<base>.png``
(values 0/1) and the source scan as ``brain_biodiversity_bank/<image>`` (RGB
JPEG).  nnU-Net v2 reads plain PNG for 2D data, so this script writes:

    <out>/nnUNet_raw/Dataset<ID>_BBDBCortex/
        imagesTr/<case>_0000.png     grayscale scan (channel 0)
        labelsTr/<case>.png          0/1 cortex mask, same size
        dataset.json

Every CSV row whose mask exists becomes a training case; every row without a
mask becomes a prediction case under ``<out>/pending_pngs/<case>_0000.png``.
Predictions must not be written into ``segmentations/``: the GUI treats any
file there as a finished section.

Usage
-----
    python nnunet_tools/build_dataset.py
    python nnunet_tools/build_dataset.py --group owl_monkey_nissl+myelin
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "brain_biodiversity_bank"
DEFAULT_CSV_DIR = REPO_ROOT / "csv"

CASE_ID_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def case_id(group: str, base: str) -> str:
    """Stable, filename-safe case identifier for one section."""
    return CASE_ID_SAFE.sub("-", f"{group}__{base}").strip("-")


def find_source(data_root: Path, image_rel: str) -> Path | None:
    """Resolve a CSV ``image`` entry, tolerating the .JPG/.jpg case mix."""
    direct = data_root / image_rel
    if direct.is_file():
        return direct
    folder = (data_root / image_rel).parent
    if not folder.is_dir():
        return None
    wanted = Path(image_rel).name.lower()
    for candidate in glob.glob(str(folder / "*")):
        if Path(candidate).name.lower() == wanted:
            return Path(candidate)
    return None


def load_mask(path: Path) -> Image.Image:
    """Open a 0/1 mask and refuse anything that is not strictly binary."""
    with Image.open(path) as raw:
        mask = raw.convert("L")
    colors = mask.getcolors(maxcolors=256)   # None when >256 distinct values
    if colors is None or not {value for _, value in colors} <= {0, 1}:
        raise ValueError(f"{path}: mask is not a 0/1 image")
    return mask


def build(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out).resolve()
    dataset_dir = out_root / "nnUNet_raw" / f"Dataset{args.dataset_id}_BBDBCortex"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    pending_dir = out_root / "pending_pngs"
    for folder in (images_dir, labels_dir, pending_dir):
        folder.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(DEFAULT_CSV_DIR.glob("*.csv"))
    if args.group:
        csv_files = [p for p in csv_files if p.stem == args.group]
    if not csv_files:
        print(f"no CSV matched in {DEFAULT_CSV_DIR}", file=sys.stderr)
        return 2

    written = skipped = pending = 0
    problems: list[str] = []
    absent_groups: list[str] = []
    for csv_path in csv_files:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and not (data_root / Path(rows[0]["image"]).parent).is_dir():
            # The scan folder for this specimen/stain was not downloaded.
            absent_groups.append(csv_path.stem)
            continue
        for row in rows:
            base = row["base"]
            source = find_source(data_root, row["image"])
            if source is None:
                problems.append(f"{base}: source image missing ({row['image']})")
                continue
            mask_path = REPO_ROOT / row["segmentation"]
            cid = case_id(csv_path.stem, base)
            try:
                with Image.open(source) as raw:
                    image = raw.convert("L")
                if mask_path.is_file():
                    mask = load_mask(mask_path)
                    if mask.size != image.size:
                        problems.append(
                            f"{base}: mask {mask.size} != image {image.size}")
                        continue
                    image.save(images_dir / f"{cid}_0000.png")
                    mask.save(labels_dir / f"{cid}.png")
                    written += 1
                else:
                    image.save(pending_dir / f"{cid}_0000.png")
                    pending += 1
            except (OSError, ValueError) as exc:
                problems.append(f"{base}: {exc}")
            else:
                continue
            skipped += 1

    dataset_json = {
        "channel_names": {"0": "gray"},
        "labels": {"background": 0, "cortex": 1},
        "numTraining": written,
        "file_ending": ".png",
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")

    print(f"dataset      {dataset_dir}")
    print(f"train cases  {written}")
    print(f"pending      {pending}  -> {pending_dir}")
    print(f"groups without scans  {len(absent_groups)}")
    if absent_groups:
        print("  " + ", ".join(absent_groups[:5])
              + (" ..." if len(absent_groups) > 5 else ""))
    print(f"problems     {len(problems)}")
    for line in problems:
        print(f"  ! {line}")
    if written == 0:
        print("no training cases: nothing for nnU-Net to learn from", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO_ROOT / "nnUNet"),
                        help="where nnUNet_raw/ and pending_pngs/ are written")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="folder holding v.1/sections/... source scans")
    parser.add_argument("--dataset-id", default="501",
                        help="nnU-Net dataset number (default 501)")
    parser.add_argument("--group", default=None,
                        help="only this CSV stem, e.g. owl_monkey_nissl+myelin")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(build(parse_args()))
