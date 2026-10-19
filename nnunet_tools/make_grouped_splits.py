"""Write nnU-Net splits that never validate on a synthetic sibling.

Dataset 502 mixes real sections with 100 synthetic variants of each of them.
A plain random split would put ``caseA__syn003`` in training and
``caseA__syn057`` in validation, so validation would measure memorisation of the
same mask rather than generalisation.

This script groups every case by its base section (the part before ``__syn``)
and assigns whole groups to folds, so a validation case is always a section the
model has never seen -- synthetic or real.

Usage
-----
    python nnunet_tools/make_grouped_splits.py --dataset-dir nnUNet/nnUNet_preprocessed/Dataset502_BBDBCortexSynth
    python nnunet_tools/make_grouped_splits.py --dataset-dir ... --write
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def base_group(case: str) -> str:
    return case.split("__syn")[0]


def build(cases: list[str], folds: int, seed: int) -> list[dict]:
    groups = sorted({base_group(case) for case in cases})
    random.Random(seed).shuffle(groups)
    buckets: list[set[str]] = [set() for _ in range(folds)]
    for index, group in enumerate(groups):
        buckets[index % folds].add(group)
    splits = []
    for bucket in buckets:
        splits.append({
            "train": sorted(case for case in cases if base_group(case) not in bucket),
            "val": sorted(case for case in cases if base_group(case) in bucket),
        })
    return splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", required=True,
                        help="nnUNet_preprocessed/DatasetXXX_... folder")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", action="store_true",
                        help="actually write splits_final.json")
    args = parser.parse_args(argv)

    folder = Path(args.dataset_dir)
    # nnU-Net stores preprocessed cases under nnUNetPlans_<config>/ as .b2nd
    plan_dirs = sorted(folder.glob("nnUNetPlans_*"))
    if not plan_dirs:
        print(f"no nnUNetPlans_* folder in {folder}", file=sys.stderr)
        return 2
    cases = sorted(p.name[: -len(".b2nd")] for p in plan_dirs[0].glob("*.b2nd")
                   if not p.name.endswith("_seg.b2nd"))
    if not cases:
        print(f"no preprocessed cases in {plan_dirs[0]}", file=sys.stderr)
        return 2
    splits = build(cases, args.folds, args.seed)
    for index, split in enumerate(splits):
        groups = sorted({base_group(c) for c in split["val"]})
        print(f"fold {index}: train {len(split['train']):4d}  "
              f"val {len(split['val']):4d}  val sections {groups}")
    if args.write:
        path = folder / "splits_final.json"
        path.write_text(json.dumps(splits, indent=1) + "\n", encoding="utf-8")
        print(f"written {path}")
    else:
        print("(dry run; pass --write to replace splits_final.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
