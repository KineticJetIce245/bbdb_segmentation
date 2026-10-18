"""Append-only JSONL operation log next to a mask."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .settings import MAX_LOG_RECORDS


def operations_path(seg_path: Path) -> Path:
    return seg_path.with_name(seg_path.stem + ".ops.jsonl")


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _write_records(path: Path, records: list[dict]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _merge_records(first: dict, second: dict) -> dict:
    tags: list[str] = []
    for record in (first, second):
        tag = record.get("tag")
        if tag:
            tags.append(str(tag))
        for item in record.get("tags", []):
            if item not in tags:
                tags.append(str(item))

    merged = {
        "time": second.get("time", first.get("time")),
        "first_time": first.get("first_time", first.get("time")),
        "last_time": second.get("last_time", second.get("time")),
        "section": second.get("section", first.get("section")),
        "image": second.get("image", first.get("image")),
        "mask": second.get("mask", first.get("mask")),
        "op": "merged",
        "count": int(first.get("count", 1)) + int(second.get("count", 1)),
        "added": int(first.get("added", 0)) + int(second.get("added", 0)),
        "removed": int(first.get("removed", 0)) + int(second.get("removed", 0)),
        "segmented": bool(
            second.get("segmented", first.get("segmented", False))
        ),
        "stop_auto_confirm": bool(
            second.get(
                "stop_auto_confirm",
                first.get("stop_auto_confirm", False),
            )
        ),
        "tags": tags,
    }
    return merged


def append_operation(seg_path: Path, record: dict) -> None:
    """Append one record, compacting the oldest pair once over the limit."""
    record.setdefault(
        "time",
        datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    path = operations_path(seg_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _read_records(path)
    records.append(record)
    while len(records) > MAX_LOG_RECORDS:
        records[0:2] = [_merge_records(records[0], records[1])]
    _write_records(path, records)


def read_last_operation(seg_path: Path) -> dict | None:
    records = _read_records(operations_path(seg_path))
    return records[-1] if records else None


def read_status(seg_path: Path) -> tuple[bool, bool]:
    """Return (segmented, stop_auto_confirm), falling back to PNG existence."""
    for record in reversed(_read_records(operations_path(seg_path))):
        if "segmented" in record:
            return (
                bool(record["segmented"]),
                bool(record.get("stop_auto_confirm", False)),
            )
    return seg_path.exists(), False
