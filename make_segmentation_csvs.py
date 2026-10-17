#!/usr/bin/env python3
"""Build one segmentation CSV per specimen/stain from the BBDB sections.

Rows come from ``sect_info.csv`` (specimen, species, stain, section metadata).
Images found on disk in a listed ``v.1/sections/<folder>/`` but missing from
``sect_info.csv`` (e.g. upper-case ``.JPG`` files) are added too, with the
section parsed from the leading digits of the filename. Existing ``_manual``
annotation files are excluded.

Writes ``csv/<specimen>_<stain>.csv`` with an ``image`` path (relative to the
data root) and a ``segmentation`` path (relative to this folder) where the GUI
saves the mask. Re-running keeps any flip/rotation already recorded.

Usage
  python make_segmentation_csvs.py --data-root /path/to/brain_biodiversity_bank
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from common import normalize_flip, normalize_rotation

TOOL_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
COLUMNS = [
    "group", "specimen", "species", "stain", "section",
    "base", "image", "segmentation", "flip", "rotation",
]


def find_data_root(override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    for parent in [TOOL_DIR, *TOOL_DIR.parents]:
        for candidate in [
            parent,
            *sorted(parent.glob("brain_biodiversity_bank*")),
            *sorted(parent.glob("brain_biodiversity_bank*/brain_biodiversity_bank")),
        ]:
            if (candidate / "v.1" / "sections").is_dir():
                return candidate
    raise SystemExit("Could not find the data root; pass --data-root /path/to/brain_biodiversity_bank")


def is_section_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "manual" not in path.stem.lower()
        and not path.name.startswith(".")
    )


def collect_rows(info: pd.DataFrame, version_dir: Path) -> pd.DataFrame:
    """sect_info rows that exist on disk, plus unlisted images in the same folders."""
    info = info.copy()
    info["image_path"] = [version_dir / f for f in info["filename"]]
    exists = info["image_path"].apply(Path.is_file)
    for missing in info.loc[~exists, "filename"]:
        print(f"warning: listed in sect_info.csv but not on disk: {missing}")
    info = info[exists & info["image_path"].apply(is_section_image)]

    listed = set(info["image_path"])
    extras = []
    for folder, group in info.groupby(info["image_path"].apply(lambda p: p.parent)):
        meta = group.iloc[0]
        for path in sorted(folder.iterdir()):
            if path in listed or not is_section_image(path):
                continue
            match = re.match(r"\d+", path.name)
            extras.append({
                "specimen": meta["specimen"],
                "species": meta["species"],
                "stain": meta["stain"],
                "filename": path.relative_to(version_dir).as_posix(),
                "section": match.group(0) if match else "",
                "image_path": path,
            })
            print(f"added (not in sect_info.csv): {extras[-1]['filename']}")
    return pd.concat([info, pd.DataFrame(extras)], ignore_index=True)


def build_group(rows: pd.DataFrame, group: str, data_root: Path, seg_dir: Path) -> pd.DataFrame:
    rows = rows.sort_values("filename", key=lambda s: s.str.lower()).reset_index(drop=True)
    stems = rows["image_path"].apply(lambda p: p.stem)
    records = []
    for (_, row), stem in zip(rows.iterrows(), stems):
        path: Path = row["image_path"]
        # Disambiguate e.g. "x.jpg" and "x.JPG" in the same group.
        base = stem if (stems == stem).sum() == 1 else path.name.replace(".", "_")
        records.append({
            "group": group,
            "specimen": row["specimen"],
            "species": row["species"],
            "stain": row["stain"],
            "section": row["section"],
            "base": base,
            "image": path.relative_to(data_root).as_posix(),
            "segmentation": (seg_dir.relative_to(TOOL_DIR) / group / f"{base}.png").as_posix(),
            "flip": "",
            "rotation": 0,
        })
    return pd.DataFrame(records, columns=COLUMNS)


def merge_existing(records: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Keep flip/rotation from a previous run, matched by image path."""
    if not csv_path.exists():
        return records
    existing = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    if "image" not in existing.columns:
        return records
    existing = existing.drop_duplicates("image", keep="last").set_index("image")
    for row_index, image in records["image"].items():
        if image in existing.index:
            if "flip" in existing.columns:
                records.at[row_index, "flip"] = normalize_flip(existing.at[image, "flip"])
            if "rotation" in existing.columns:
                records.at[row_index, "rotation"] = normalize_rotation(existing.at[image, "rotation"])
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None, help="folder containing v.1/sections (auto-detected by default)")
    parser.add_argument("--sect-info", default=str(TOOL_DIR / "sect_info.csv"), help="section metadata CSV")
    parser.add_argument("--csv-dir", default=str(TOOL_DIR / "csv"), help="output folder for the per-group CSVs")
    parser.add_argument("--seg-dir", default=str(TOOL_DIR / "segmentations"), help="where masks will be saved (must be inside this folder)")
    args = parser.parse_args()

    data_root = find_data_root(args.data_root)
    version_dir = data_root / "v.1"
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    seg_dir = Path(args.seg_dir).expanduser().resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    info = pd.read_csv(args.sect_info, dtype=str, keep_default_na=False)
    rows = collect_rows(info, version_dir)

    for (specimen, stain), group_rows in rows.groupby(["specimen", "stain"], sort=True):
        group = f"{specimen}_{stain}"
        csv_path = csv_dir / f"{group}.csv"
        records = merge_existing(build_group(group_rows, group, data_root, seg_dir), csv_path)
        records.to_csv(csv_path, index=False)
        print(f"{csv_path.relative_to(TOOL_DIR)}: {len(records)} images")


if __name__ == "__main__":
    main()
