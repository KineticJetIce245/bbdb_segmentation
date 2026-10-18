"""Filesystem helpers shared by the GUI, tests and the CSV builder."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(root_dir: Path, path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return root_dir / path


def rel_path(root_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def find_data_root(override: str | None = None) -> Path | None:
    if override:
        return Path(override).expanduser().resolve()
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        for candidate in [
            parent,
            *sorted(parent.glob("brain_biodiversity_bank*")),
            *sorted(parent.glob("brain_biodiversity_bank*/brain_biodiversity_bank")),
        ]:
            if (candidate / "v.1" / "sections").is_dir():
                return candidate
    return None
