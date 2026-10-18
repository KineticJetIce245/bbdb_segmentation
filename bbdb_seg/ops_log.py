"""Append-only JSONL operation log next to a mask."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def operations_path(seg_path: Path) -> Path:
    return seg_path.with_name(seg_path.stem + ".ops.jsonl")


def append_operation(seg_path: Path, record: dict) -> None:
    """Append one record, adding an ISO timestamp if the caller did not."""
    record.setdefault(
        "time",
        datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    path = operations_path(seg_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
