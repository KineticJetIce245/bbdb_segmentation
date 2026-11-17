"""UNet model integration: a persistent worker, yellow predictions and caching.

The GUI runs in a small environment (no torch).  Predictions therefore come
from ``nnunet_tools/predict_worker.py``, launched with the separate
``.venv-nnunet`` interpreter.  The worker loads the 504 checkpoint once and
answers one JSON request per section; this mixin owns its lifecycle and keeps
Tk responsive by talking to it from a background reader thread.
"""
from __future__ import annotations

import collections
import json
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image

from .image_ops import load_mask_work
from .paths import REPO_ROOT, resolve_path
from .settings import (
    APP_TITLE,
    UNET_CHECKPOINT,
    UNET_DEVICE,
    UNET_MODEL_DIR,
    UNET_OUTPUT_DIR,
    UNET_POLL_MS,
    UNET_PYTHON,
)


class UnetMixin:

    # --- lifecycle ----------------------------------------------------------- #

    def _init_unet_state(self) -> None:
        self.unet_enabled = False
        self.unet_proc: subprocess.Popen | None = None
        self.unet_ready = False
        self.unet_poll_id: str | None = None
        self.unet_messages: queue.Queue = queue.Queue()
        self.unet_stderr: collections.deque[str] = collections.deque(maxlen=40)
        self.unet_pending: dict[int, tuple[int, Path, bool]] = {}
        self.unet_request_seq = 0
        self.unet_waiting: tuple[int, bool, bool] | None = None
        self.unet_status = "off"

        self.prediction_img: Image.Image | None = None
        self.prediction_index: int | None = None
        self.prediction_applied = False
        # Yellow means "draft the model proposed". Accepting it with Use
        # prediction turns the overlay red while the status stays predicted.
        self.prediction_yellow = False

        model_override = getattr(self, "unet_model_override", None)
        python_override = getattr(self, "unet_python_override", None)
        self.unet_model = self._resolve_under_repo(model_override or UNET_MODEL_DIR)
        self.unet_python = self._resolve_under_repo(python_override or self._default_unet_python())
        self.unet_worker = REPO_ROOT / "nnunet_tools" / "predict_worker.py"
        self.unet_device = getattr(self, "unet_device_override", None) or UNET_DEVICE
        self._set_unet_status("off")

    @staticmethod
    def _resolve_under_repo(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    @staticmethod
    def _default_unet_python() -> Path:
        windows = REPO_ROOT / ".venv-nnunet" / "Scripts" / "python.exe"
        return windows if windows.is_file() else REPO_ROOT / UNET_PYTHON

    def unet_output_dir(self) -> Path:
        """``segmentations/UNet_output`` under the folder the CSV paths use."""
        return self.seg_root / "segmentations" / UNET_OUTPUT_DIR

    def unet_output_path(self, index: int) -> Path:
        row = self.records.iloc[index]
        group = str(row.get("group", "") or "unknown")
        return self.unet_output_dir() / group / f"{row['base']}.png"

    def _missing_unet_pieces(self) -> list[str]:
        missing = []
        if not Path(self.unet_python).is_file():
            missing.append(f"interpreter: {self.unet_python}")
        if not Path(self.unet_worker).is_file():
            missing.append(f"worker: {self.unet_worker}")
        if not Path(self.unet_model).is_dir():
            missing.append(f"model: {self.unet_model}")
        return missing

    def start_unet(self) -> None:
        if self.unet_proc is not None and self.unet_proc.poll() is None:
            self.unet_enabled = True
            return
        missing = self._missing_unet_pieces()
        if missing:
            self._set_unet_status("error: missing " + ", ".join(missing))
            self.unet_var.set(False)
            self.unet_enabled = False
            return
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bbdb-mpl"))
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            str(self.unet_python), str(self.unet_worker),
            "--model-dir", str(self.unet_model),
            "--checkpoint", UNET_CHECKPOINT,
            "--device", self.unet_device,
        ]
        try:
            self.unet_proc = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            self._set_unet_status(f"error: {exc}")
            self.unet_var.set(False)
            self.unet_enabled = False
            return
        self.unet_enabled = True
        self.unet_ready = False
        self.unet_messages = queue.Queue()
        self.unet_stderr.clear()
        self.unet_pending.clear()
        self.unet_waiting = None
        self._set_unet_status("loading")
        threading.Thread(target=self._unet_read_stdout, args=(self.unet_proc,),
                         daemon=True).start()
        threading.Thread(target=self._unet_read_stderr, args=(self.unet_proc,),
                         daemon=True).start()
        if self.unet_poll_id is not None:
            self.after_cancel(self.unet_poll_id)
        self.unet_poll_id = self.after(UNET_POLL_MS, self._poll_unet)

    def _unet_read_stdout(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:
                self.unet_messages.put(line)
        finally:
            self.unet_messages.put(None)

    def _unet_read_stderr(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stderr:
                self.unet_stderr.append(line.rstrip())
        except (ValueError, OSError):
            pass

    def _poll_unet(self) -> None:
        self.unet_poll_id = None
        proc = self.unet_proc
        if proc is None:
            return
        while True:
            try:
                line = self.unet_messages.get_nowait()
            except queue.Empty:
                break
            if line is None:
                break
            self._handle_unet_line(line)
        if proc.poll() is not None:
            if self.unet_enabled:
                tail = " | ".join(list(self.unet_stderr)[-3:])
                self._set_unet_status(f"error: worker exited ({proc.returncode}) {tail}".strip())
                self.unet_enabled = False
                self.unet_ready = False
                self.unet_proc = None
                self.unet_var.set(False)
                self.unet_pending.clear()
                self.unet_waiting = None
            return
        if self.unet_enabled:
            self.unet_poll_id = self.after(UNET_POLL_MS, self._poll_unet)

    def _handle_unet_line(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return
        status = message.get("status")
        if status == "ready":
            self.unet_ready = True
            self._set_unet_status("ready")
            waiting = self.unet_waiting
            self.unet_waiting = None
            if waiting is not None:
                self.request_unet(*waiting)
        elif status == "ok":
            request = self.unet_pending.pop(message.get("id"), None)
            self._set_unet_status("ready")
            if request is None:
                return
            index, output, apply_if_empty = request
            mask_path = Path(str(message.get("mask", output)))
            if not mask_path.is_file():
                self._set_unet_status(f"error: missing prediction {mask_path.name}")
                return
            if index == self.index:
                self._load_prediction(index, mask_path, apply_if_empty=apply_if_empty)
        elif status == "error":
            self.unet_pending.pop(message.get("id"), None)
            self._set_unet_status("error: " + str(message.get("error", "unknown"))[:160])

    def stop_unet(self, *, render: bool = True) -> None:
        self.unet_enabled = False
        self.unet_ready = False
        self.unet_waiting = None
        self.unet_pending.clear()
        if self.unet_poll_id is not None:
            self.after_cancel(self.unet_poll_id)
            self.unet_poll_id = None
        proc, self.unet_proc = self.unet_proc, None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write('{"cmd": "quit"}\n')
                    proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if self.prediction_applied and self.mask_source == "unet":
            # Without the model there is no prediction badge to explain the
            # yellow overlay, so the mask becomes an ordinary edited one.
            self.mask_source = "edited"
        self.prediction_img = None
        self.prediction_index = None
        self.prediction_applied = False
        self.prediction_yellow = False
        self._update_prediction_controls()
        self._set_unet_status("off")
        if render:
            self.render_all()

    def shutdown_unet(self, *, render: bool = True) -> None:
        self.stop_unet(render=render)

    def _set_unet_status(self, state: str) -> None:
        self.unet_status = state
        if hasattr(self, "unet_status_var"):
            self.unet_status_var.set(f"UNet: {state}")

    # --- requests ------------------------------------------------------------ #

    def request_unet(self, index: int | None = None, force: bool = False,
                     apply_if_empty: bool = False) -> None:
        if not self.unet_enabled:
            return
        index = self.index if index is None else index
        output = self.unet_output_path(index)
        if not force and output.is_file():
            self._load_prediction(index, output, apply_if_empty=apply_if_empty)
            return
        if not self.unet_ready:
            self.unet_waiting = (index, force, apply_if_empty)
            self._set_unet_status("loading")
            return
        if any(pending[0] == index for pending in self.unet_pending.values()):
            return
        image = resolve_path(self.data_root, self.records.at[index, "image"])
        if not image.is_file():
            self._set_unet_status(f"error: missing image {image.name}")
            return
        request_id = self.unet_request_seq
        self.unet_request_seq += 1
        self.unet_pending[request_id] = (index, output, apply_if_empty)
        payload = {
            "id": request_id,
            "cmd": "predict",
            "image": str(image),
            "out": str(output),
        }
        if not self._send_unet(payload):
            self.unet_pending.pop(request_id, None)
            return
        self._set_unet_status("predicting")

    def _send_unet(self, payload: dict) -> bool:
        proc = self.unet_proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            self._set_unet_status("error: worker is not running")
            return False
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._set_unet_status(f"error: {exc}")
            return False

    # --- prediction state ---------------------------------------------------- #

    def _load_prediction(self, index: int, path: Path, apply_if_empty: bool) -> None:
        if index != self.index or self.base_img is None:
            return
        try:
            work = load_mask_work(path, self.base_img.shape)
        except Exception as exc:
            self._set_unet_status(f"error: {exc}")
            return
        self.prediction_img = Image.fromarray(work)
        self.prediction_index = index
        self.prediction_applied = False
        self.prediction_yellow = False
        if apply_if_empty and self._can_auto_apply():
            self.base_mask_img = Image.fromarray(work)
            self.mask_source = "unet"
            self.prediction_applied = True
            self.prediction_yellow = True
            self.mask_dirty = False
        self._update_prediction_controls()
        self.render_all()

    def _update_prediction_controls(self) -> None:
        button = getattr(self, "use_prediction_button", None)
        if button is None:
            return
        usable = self.prediction_img is not None and self.prediction_index == self.index
        button.state(["!disabled"] if usable else ["disabled"])
        working = usable and self.prediction_applied and self.mask_source == "unet"
        button.configure(text="Save prediction" if working else "Use prediction")
        cached = self.unet_output_path(self.index).is_file()
        for name in ("load_prediction_button", "clear_prediction_button"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.state(["!disabled"] if cached else ["disabled"])

    def _can_auto_apply(self) -> bool:
        if self.seg_path(self.index).exists() or self.mask_dirty:
            return False
        return self.mask_source not in ("saved", "edited")

    def on_section_loaded(self) -> None:
        self.prediction_img = None
        self.prediction_index = None
        self.prediction_applied = False
        self.prediction_yellow = False
        self.unet_waiting = None
        self._update_prediction_controls()
        if not self.unet_enabled:
            return
        if self.seg_path(self.index).exists():
            return
        output = self.unet_output_path(self.index)
        if output.is_file():
            self._load_prediction(self.index, output, apply_if_empty=True)
            return
        self.request_unet(self.index, apply_if_empty=True)

    def on_unet_toggle(self) -> None:
        if self.unet_var.get():
            self.start_unet()
            if self.prediction_img is None and not self.seg_path(self.index).exists():
                self.on_section_loaded()
        else:
            self.stop_unet()

    def run_unet_current(self) -> None:
        if not self.unet_enabled:
            self.unet_var.set(True)
            self.start_unet()
        self.request_unet(self.index, force=True, apply_if_empty=False)

    def load_cached_prediction(self) -> None:
        """Show the prediction already stored in UNet_output for this section."""
        path = self.unet_output_path(self.index)
        if not path.is_file():
            self.status_var.set("No cached prediction for this section")
            self._update_prediction_controls()
            return
        self._load_prediction(
            self.index, path,
            apply_if_empty=not self.seg_path(self.index).exists(),
        )
        self.status_var.set("Loaded cached prediction")

    def clear_cached_prediction(self) -> None:
        """Delete this section's prediction PNG (never the manual mask)."""
        path = self.unet_output_path(self.index).resolve()
        root = self.unet_output_dir().resolve()
        try:
            path.relative_to(root)
        except ValueError:
            self.status_var.set("Refusing to clear a path outside UNet_output")
            return
        if not path.is_file():
            self.status_var.set("No cached prediction to clear")
            self._update_prediction_controls()
            return
        try:
            path.unlink()
        except OSError as exc:
            self.status_var.set(f"Could not clear prediction: {exc}")
            return
        if self.prediction_applied and self.prediction_index == self.index:
            # The working mask stays, but it is an ordinary edited mask now.
            self.mask_source = "edited"
        self.prediction_yellow = False
        self.prediction_img = None
        self.prediction_index = None
        self.prediction_applied = False
        self._update_prediction_controls()
        self.render_all()
        self.status_var.set("Cleared cached prediction")

    def use_prediction(self) -> None:
        if self.prediction_img is None or self.prediction_index != self.index:
            self.status_var.set("No UNet prediction to use")
            return
        if self.prediction_applied and self.mask_source == "unet":
            # The prediction is already the working mask (automatic start, or
            # an earlier Use prediction). Saving it drops the draft styling.
            self.mask_dirty = True
            self.commit_current()
            self.status_var.set("Saved")
            return
        if self.seg_path(self.index).exists():
            if not messagebox.askyesno(
                APP_TITLE,
                "Replace the saved mask with the UNet prediction?\n"
                "The file is overwritten the next time this section is saved.",
                parent=self,
            ):
                return
        from .image_ops import mask_diff
        before = np.asarray(self.base_mask_img).copy()
        self.push_undo("unet")
        self.base_mask_img = self.prediction_img.copy()
        self.prediction_applied = True
        self.prediction_yellow = False   # accepted predictions are a normal mask
        self.mask_source = "unet"        # ...but still count as model-derived
        self.mask_dirty = True
        added, removed, bbox = mask_diff(before, np.asarray(self.base_mask_img))
        if bbox is not None:
            self.log_op("unet_apply", added=added, removed=removed, bbox=bbox,
                        extra={"source": "UNet"})
        self._update_prediction_controls()
        self.render_all()
        self.update_status()

    def prediction_separate(self) -> bool:
        return (self.prediction_img is not None
                and self.prediction_index == self.index
                and not self.prediction_applied)

    def finalize_prediction_after_save(self) -> None:
        """A saved mask is no longer a draft: red overlay, no predicted status."""
        if self.mask_source != "unet":
            return
        self.mask_source = "edited"
        self.prediction_yellow = False
        self.prediction_img = None
        self.prediction_index = None
        self.prediction_applied = False
        self._update_prediction_controls()
        self.render_all()

    def prediction_active(self) -> bool:
        """True while the working mask is an unedited UNet prediction.

        This is what the status line shows instead of "not segmented"; a
        prediction kept as a separate overlay does not change the mask status.
        """
        return (self.prediction_img is not None
                and self.prediction_index == self.index
                and self.prediction_applied
                and self.mask_source == "unet")
