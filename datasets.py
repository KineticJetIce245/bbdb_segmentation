"""CSV loading and queue helpers for the segmentation GUI."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import normalize_flip, normalize_rotation
from paths import resolve_path


def read_records(csv_path: Path) -> pd.DataFrame:
    """Read and normalize a group CSV; raises ValueError on malformed input."""
    records = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    missing = {"base", "image", "segmentation"} - set(records.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    if records.empty:
        raise ValueError("no rows")
    for column, default in (("flip", ""), ("rotation", "0")):
        if column not in records.columns:
            records[column] = default
    records["flip"] = records["flip"].apply(normalize_flip)
    records["rotation"] = records["rotation"].apply(normalize_rotation).astype(int)
    return records


def pending_indices(records: pd.DataFrame, seg_root: Path) -> list[int]:
    return [
        index for index in range(len(records))
        if not resolve_path(seg_root, records.at[index, "segmentation"]).exists()
    ]


def write_records(csv_path: Path, records: pd.DataFrame) -> None:
    output = records.copy()
    output["flip"] = output["flip"].fillna("").astype(str)
    output["rotation"] = output["rotation"].fillna(0).astype(int)
    output.to_csv(csv_path, index=False)
