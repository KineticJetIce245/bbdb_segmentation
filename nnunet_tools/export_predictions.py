"""Lay nnU-Net predictions out the way the GUI's ``--init-dir`` expects.

``nnUNetv2_predict`` names its output ``<case>.png``, where the case id is the
sanitised ``<group>__<base>``. The GUI looks for ``<init-dir>/<group>/<base>.png``
first, so these need renaming; doing it here keeps the CSV's group/base naming
authoritative instead of guessing from a file name.

Usage
-----
    python nnunet_tools/export_predictions.py --predictions nnUNet/pred_pending
    python nnunet_tools/export_predictions.py --predictions DIR --out nnUNet/init_masks
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ID_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def case_id(group: str, base: str) -> str:
    return CASE_ID_SAFE.sub("-", f"{group}__{base}").strip("-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", required=True,
                        help="folder written by nnUNetv2_predict")
    parser.add_argument("--out", default=str(REPO_ROOT / "nnUNet" / "init_masks"),
                        help="folder to pass to the GUI's --init-dir")
    parser.add_argument("--csv-dir", default=str(REPO_ROOT / "csv"))
    args = parser.parse_args(argv)

    pred_dir = Path(args.predictions)
    out_dir = Path(args.out)
    if not pred_dir.is_dir():
        print(f"no such folder: {pred_dir}", file=sys.stderr)
        return 2

    written = missing = 0
    for csv_path in sorted(Path(args.csv_dir).glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            cid = case_id(csv_path.stem, row["base"])
            source = pred_dir / f"{cid}.png"
            if not source.is_file():
                missing += 1
                continue
            target = out_dir / csv_path.stem / f"{row['base']}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            written += 1

    print(f"copied   {written}  -> {out_dir}")
    print(f"not found {missing} (sections without a prediction)")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
