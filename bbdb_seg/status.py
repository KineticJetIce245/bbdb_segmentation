"""Per-image segmented status and ops-log tags."""
from __future__ import annotations

from .ops_log import read_status


class StatusMixin:
    def status_for_index(self, index: int) -> tuple[bool, bool]:
        if index not in self.status_cache:
            self.status_cache[index] = read_status(self.seg_path(index))
        return self.status_cache[index]

    def _load_status(self) -> None:
        self.segmented_status, self.stop_auto_confirm = self.status_for_index(self.index)
        self.segmented_var.set(self.segmented_status)
        self.stop_auto_confirm_var.set(self.stop_auto_confirm)

    def _set_segmented(self, value: bool, *, auto: bool = False) -> None:
        value = bool(value)
        changed = value != self.segmented_status
        self.segmented_status = value
        self.segmented_var.set(value)
        self.status_cache[self.index] = (value, self.stop_auto_confirm)
        if changed:
            self.log_op(
                "status",
                extra={
                    "tag": "segmented" if value else "not segmented",
                    "auto": auto,
                },
            )
        self.update_status()

    def on_segmented_toggle(self) -> None:
        self._set_segmented(bool(self.segmented_var.get()), auto=False)

    def on_stop_auto_confirm_toggle(self) -> None:
        self.stop_auto_confirm = bool(self.stop_auto_confirm_var.get())
        self.status_cache[self.index] = (
            self.segmented_status,
            self.stop_auto_confirm,
        )
        self.log_op(
            "status",
            extra={
                "tag": "stop autolog" if self.stop_auto_confirm else "resume autolog",
            },
        )
        self.update_status()

