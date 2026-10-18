"""CSV selection and work-queue refresh for the segmentation GUI."""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox

import pandas as pd

from .datasets import pending_indices, read_records
from .paths import REPO_ROOT, find_data_root
from .settings import APP_TITLE


class QueueMixin:

    def resolve_inputs(
        self,
        csv_path: Path | None,
        data_root_override: str | None,
        review_all: bool,
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
        queue = self._choose_queue(records, csv_path, data_root, review_all)
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
        csv_path: Path,
        data_root: Path,
        review_all: bool,
    ) -> list[int] | None:
        pending = pending_indices(records, self.seg_root)
        if review_all:
            queue = list(range(len(records)))
            self.review_all = True
        elif pending:
            queue = pending
            self.review_all = False
        elif messagebox.askyesno(
            APP_TITLE,
            f"All {len(records)} images in {csv_path.name} are already segmented.\n\nReview them anyway?",
            parent=self,
        ):
            queue = list(range(len(records)))
            self.review_all = True
        else:
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

    def on_review_finished_change(self) -> None:
        """Rebuild the work queue when the Review finished checkbox changes."""
        include = bool(self.review_finished_var.get())
        if include == self.review_all:
            return

        self.flush_slider_updates()
        current = self.index
        if include:
            new_queue = list(range(len(self.records)))
        else:
            new_queue = [
                i for i in range(len(self.records))
                if not self.seg_path(i).exists()
            ]
            if current not in new_queue and new_queue:
                # Leaving a finished section; preserve its edits before moving.
                self.flush_mask()
        if not new_queue:
            self.review_finished_var.set(True)
            self.review_all = True
            self.update_status()
            messagebox.showinfo(
                APP_TITLE,
                "There are no unfinished sections to show.",
                parent=self,
            )
            return

        self.review_all = include
        self.queue = new_queue
        if current in new_queue:
            self.pos = new_queue.index(current)
            self.update_status()
        else:
            self.pos = 0
            self.index = new_queue[0]
            self.load_current()

    # --- UI construction ---------------------------------------------------- #
