"""CSV selection and work-queue refresh for the segmentation GUI."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tkinter import filedialog, messagebox

from .datasets import queue_indices, read_records
from .paths import REPO_ROOT, find_data_root
from .settings import APP_TITLE


class QueueMixin:
    def resolve_inputs(
        self,
        csv_path: Path | None,
        data_root_override: str | None,
    ) -> tuple[Path, Path, pd.DataFrame, list[int]] | None:
        """Pick a CSV and data root (dialogs if needed) and build the work queue."""
        csv_path = self._pick_csv(csv_path)
        if csv_path is None:
            return None
        data_root = self._pick_data_root(data_root_override)
        if data_root is None:
            return None
        try:
            records = read_records(csv_path)
        except Exception as exc:  # malformed/unreadable CSV -- report instead of crashing
            messagebox.showerror(APP_TITLE, f"Could not read CSV:\n{csv_path}\n\n{exc}", parent=self)
            return None
        queue = self._choose_queue(records, data_root)
        if queue is None:
            return None
        return csv_path, data_root, records, queue

    def _pick_csv(self, csv_path: Path | None) -> Path | None:
        if csv_path is None:
            chosen = filedialog.askopenfilename(
                parent=self,
                title="Select a segmentation CSV",
                initialdir=str(REPO_ROOT / "csv"),
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not chosen:
                return None
            csv_path = Path(chosen).expanduser().resolve()
        if not csv_path.exists():
            messagebox.showerror(APP_TITLE, f"CSV not found:\n{csv_path}", parent=self)
            return None
        return csv_path

    def _pick_data_root(self, data_root_override: str | None) -> Path | None:
        data_root = find_data_root(data_root_override)
        if data_root is not None:
            return data_root
        chosen = filedialog.askdirectory(
            parent=self,
            title="Select the brain_biodiversity_bank data folder (contains v.1/)",
        )
        if not chosen:
            return None
        return Path(chosen).expanduser().resolve()

    def _choose_queue(
        self,
        records: pd.DataFrame,
        data_root: Path,
    ) -> list[int] | None:
        queue = list(range(len(records)))
        if not queue:
            messagebox.showerror(APP_TITLE, "No rows in the CSV.", parent=self)
            return None
        if not (data_root / str(records.at[queue[0], "image"])).exists():
            messagebox.showerror(
                APP_TITLE,
                f"Images not found under data folder:\n{data_root}\n\n"
                "Pass --data-root /path/to/brain_biodiversity_bank",
                parent=self,
            )
            return None
        return queue

    def on_queue_filter_change(self, _event=None) -> None:
        """Rebuild the work queue from not segmented / segmented / all."""
        mode = self.queue_filter_var.get()
        new_queue = queue_indices(self.records, self.seg_root, mode)
        if not new_queue:
            messagebox.showinfo(
                APP_TITLE,
                f"No images match the queue filter: {mode}.",
                parent=self,
            )
            return
        current = self.index
        self.queue = new_queue
        if current in new_queue:
            self.pos = new_queue.index(current)
            self.update_status()
        else:
            self.pos = 0
            self.index = new_queue[0]
            self.load_current()
