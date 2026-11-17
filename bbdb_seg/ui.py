"""Tk widget construction for the segmentation GUI."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .settings import MAX_BRUSH, MIN_BRUSH, ZOOM_STEP


class UiMixin:
    def build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.grid(row=0, column=0, sticky="ew")
        self._build_toolbar(toolbar)

        controls = ttk.Frame(self, padding=(8, 2))
        controls.grid(row=1, column=0, sticky="ew")
        self._build_controls(controls)

        self._build_canvas()
        self._build_statusbar()
        self._build_help()
        self.update_brush_label()
        self.update_mode_label()

    def _build_toolbar(self, toolbar: ttk.Frame) -> None:
        top_row = ttk.Frame(toolbar)
        top_row.pack(side="top", fill="x")
        bottom_row = ttk.Frame(toolbar)
        bottom_row.pack(side="top", fill="x", pady=(4, 0))
        options_row = ttk.Frame(toolbar)
        options_row.pack(side="top", fill="x", pady=(4, 0))
        unet_row = ttk.Frame(toolbar)
        unet_row.pack(side="top", fill="x", pady=(4, 0))
        self._build_toolbar_actions(top_row)
        self._build_toolbar_brush(bottom_row)
        self._build_toolbar_options(options_row)
        self._build_toolbar_unet(unet_row)

    def _build_toolbar_actions(self, row: ttk.Frame) -> None:
        ttk.Button(row, text="< Prev", command=lambda: self.move(-1)).pack(side="left")
        ttk.Button(row, text="Next >", command=lambda: self.move(1)).pack(side="left", padx=(2, 10))

        ttk.Button(row, text="Rotate -90", command=lambda: self.rotate(-1)).pack(side="left")
        ttk.Button(row, text="Rotate +90", command=lambda: self.rotate(1)).pack(side="left", padx=(2, 10))

        ttk.Button(row, text="Flip H", command=self.flip_h).pack(side="left")
        ttk.Button(row, text="Flip V", command=self.flip_v).pack(side="left", padx=(2, 10))

        self.mode_button = ttk.Button(row, text="Mode: Draw", width=12, command=self.cycle_tool)
        self.mode_button.pack(side="left")
        ttk.Button(row, text="Fill (f)", command=lambda: self.set_tool("fill")).pack(side="left", padx=(2, 0))
        ttk.Button(row, text="Brush select (L)", command=lambda: self.set_tool("select")).pack(side="left", padx=(2, 0))
        ttk.Button(row, text="Colour pick (p)", command=lambda: self.set_tool("pick")).pack(side="left", padx=(2, 0))
        ttk.Button(row, text="Clear blue", command=self.clear_block_layer).pack(side="left", padx=(2, 0))
        ttk.Button(row, text="Undo (Ctrl+Z)", command=self.undo).pack(side="left", padx=(2, 0))
        ttk.Button(row, text="Redo (Ctrl+Y)", command=self.redo).pack(side="left", padx=(2, 0))
        ttk.Button(row, text="Clear mask", command=self.clear_mask).pack(side="left", padx=(2, 10))

    def _build_toolbar_options(self, row: ttk.Frame) -> None:
        """Third toolbar row: tool options (checkboxes)."""
        ttk.Label(row, text="Options:").pack(side="left", padx=(0, 4))
        options = (
            ("Exclude background", self.exclude_bg_var, self.on_exclude_bg_toggle),
            ("Exclude legend", self.exclude_legend_var, self.on_exclude_legend_toggle),
            ("Smooth strokes", self.smooth_var, None),
            ("Snap brush to edges", self.snap_brush_var, None),
            ("Keep brush strokes (Clean)", self.keep_strokes_var, None),
        )
        for text, var, command in options:
            kwargs = {"command": command} if command is not None else {}
            ttk.Checkbutton(row, text=text, variable=var, takefocus=False, **kwargs).pack(
                side="left", padx=(2, 8))

    def _build_toolbar_unet(self, row: ttk.Frame) -> None:
        """Fourth toolbar row: optional UNet predictions (yellow)."""
        ttk.Label(row, text="UNet:").pack(side="left", padx=(0, 4))
        ttk.Checkbutton(
            row, text="Use model", variable=self.unet_var,
            command=self.on_unet_toggle, takefocus=False,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Run UNet", command=self.run_unet_current).pack(
            side="left", padx=(0, 4))
        self.load_prediction_button = ttk.Button(
            row, text="Load prediction", command=self.load_cached_prediction)
        self.load_prediction_button.pack(side="left", padx=(0, 4))
        self.load_prediction_button.state(["disabled"])
        self.clear_prediction_button = ttk.Button(
            row, text="Clear prediction", command=self.clear_cached_prediction)
        self.clear_prediction_button.pack(side="left", padx=(0, 4))
        self.clear_prediction_button.state(["disabled"])
        self.use_prediction_button = ttk.Button(
            row, text="Use prediction", command=self.use_prediction)
        self.use_prediction_button.pack(side="left", padx=(0, 8))
        self.use_prediction_button.state(["disabled"])
        ttk.Label(row, textvariable=self.unet_status_var).pack(side="left", padx=(8, 0))

    def _build_toolbar_brush(self, row: ttk.Frame) -> None:
        ttk.Button(row, text="Erode (E)", command=self.erode_mask).pack(side="left")
        ttk.Button(row, text="Dilate (D)", command=self.dilate_mask).pack(side="left", padx=(2, 2))
        ttk.Button(row, text="Clean (k)", command=self.clean_current).pack(side="left", padx=(2, 2))
        ttk.Frame(row, width=12).pack(side="left")

        ttk.Button(row, text="Brush -", command=lambda: self.change_brush(-2)).pack(side="left")
        ttk.Label(row, text="Brush:").pack(side="left", padx=(6, 2))
        self.brush_spin = ttk.Spinbox(
            row, from_=MIN_BRUSH, to=MAX_BRUSH, increment=1, width=4,
            textvariable=self.brush_input_var, command=self.on_brush_entry,
        )
        self.brush_spin.pack(side="left")
        self.brush_spin.bind("<Return>", self.on_brush_entry)
        self.brush_spin.bind("<KP_Enter>", self.on_brush_entry)
        self.brush_spin.bind("<FocusOut>", self.on_brush_entry)
        ttk.Label(row, text="px").pack(side="left", padx=(2, 0))
        ttk.Button(row, text="Brush +", command=lambda: self.change_brush(2)).pack(side="left", padx=(4, 14))

        ttk.Button(row, text="Accept & next (Enter)", command=self.accept_and_next).pack(side="left")
        self.done_label = tk.Label(row, textvariable=self.done_var, font=("TkDefaultFont", 10, "bold"))
        self.done_label.pack(side="left", padx=(8, 8))
        ttk.Checkbutton(
            row, text="Segmented", variable=self.segmented_var,
            command=self.on_segmented_toggle, takefocus=False,
        ).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(
            row, text="Stop auto confirm", variable=self.stop_auto_confirm_var,
            command=self.on_stop_auto_confirm_toggle, takefocus=False,
        ).pack(side="left", padx=(0, 8))
        ttk.Label(row, textvariable=self.status_var).pack(side="right")

    def _build_controls(self, controls: ttk.Frame) -> None:
        threshold_row = ttk.Frame(controls)
        threshold_row.pack(side="top", fill="x")
        self._build_threshold_controls(threshold_row)

        display_row = ttk.Frame(controls)
        display_row.pack(side="top", fill="x", pady=(4, 0))
        self._build_display_controls(display_row)

    def _build_threshold_controls(self, row: ttk.Frame) -> None:
        self.threshold_min_scale = tk.Scale(
            row, from_=0, to=255, orient="horizontal", length=170,
            label="Threshold min", command=self.on_threshold_min_change,
            takefocus=False, highlightthickness=0,
        )
        self.threshold_min_scale.pack(side="left", padx=(0, 4))
        self.threshold_max_scale = tk.Scale(
            row, from_=0, to=255, orient="horizontal", length=170,
            label="Threshold max", command=self.on_threshold_max_change,
            takefocus=False, highlightthickness=0,
        )
        self.threshold_max_scale.pack(side="left", padx=(0, 8))
        for scale in (self.threshold_min_scale, self.threshold_max_scale):
            scale.bind("<ButtonRelease-1>", self.end_threshold_drag, add="+")
        ttk.Button(row, text="Otsu (t)", command=self.reset_threshold).pack(side="left")
        ttk.Checkbutton(
            row, text="Invert (i)", variable=self.invert_var, command=self.on_invert_change,
            takefocus=False,
        ).pack(side="left", padx=(6, 4))
        ttk.Label(row, textvariable=self.otsu_var, width=10).pack(side="left", padx=(0, 18))

    def _build_display_controls(self, row: ttk.Frame) -> None:
        self.alpha_scale = tk.Scale(
            row, from_=0, to=255, orient="horizontal", length=160,
            label="Overlay opacity", command=self.on_alpha_change,
            takefocus=False, highlightthickness=0,
        )
        self.alpha_scale.pack(side="left", padx=(0, 14))
        self.min_scale = tk.Scale(
            row, from_=0, to=255, orient="horizontal", length=180,
            label="Contrast min", command=self.on_contrast_change,
            takefocus=False, highlightthickness=0,
        )
        self.min_scale.pack(side="left", padx=(0, 8))
        self.max_scale = tk.Scale(
            row, from_=0, to=255, orient="horizontal", length=180,
            label="Contrast max", command=self.on_contrast_change,
            takefocus=False, highlightthickness=0,
        )
        self.max_scale.pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Auto", command=self.auto_contrast).pack(side="left", padx=(4, 4))
        ttk.Button(row, text="Reset", command=self.reset_contrast).pack(side="left")
        ttk.Label(row, text="Queue:").pack(side="left", padx=(10, 2))
        self.queue_menu = ttk.Combobox(
            row, textvariable=self.queue_filter_var,
            values=("not segmented", "segmented", "all"),
            state="readonly", width=16,
        )
        self.queue_menu.pack(side="left")
        self.queue_menu.bind("<<ComboboxSelected>>", self.on_queue_filter_change)
        self.alpha_scale.set(self.overlay_alpha)
        self.min_scale.set(self.contrast_min)
        self.max_scale.set(self.contrast_max)

    def _build_canvas(self) -> None:
        self.canvas = tk.Canvas(self, background="#101010", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", lambda e: self.on_press(e, right=False))
        self.canvas.bind("<B1-Motion>", lambda e: self.on_drag(e, right=False))
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.canvas.bind("<Shift-ButtonPress-1>", self.start_roi_select)
        self.canvas.bind("<Shift-B1-Motion>", self.drag_roi_select)
        self.canvas.bind("<Shift-ButtonRelease-1>", self.end_roi_select)
        self.canvas.bind("<Shift-ButtonPress-3>", self.start_lasso_select)
        self.canvas.bind("<Shift-B3-Motion>", self.drag_lasso_select)
        self.canvas.bind("<Shift-ButtonRelease-3>", self.end_lasso_select)
        self.canvas.bind("<ButtonPress-3>", lambda e: self.on_press(e, right=True))
        self.canvas.bind("<B3-Motion>", lambda e: self.on_drag(e, right=True))
        self.canvas.bind("<ButtonRelease-3>", self.end_stroke)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        self.canvas.bind("<Control-ButtonPress-1>", self.start_pan)
        self.canvas.bind("<Control-B1-Motion>", self.pan_drag)
        self.canvas.bind("<Control-ButtonRelease-1>", self.end_pan)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, ZOOM_STEP))
        self.canvas.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 1.0 / ZOOM_STEP))
        self.canvas.bind("<Shift-Button-4>", lambda e: self.change_brush(2))
        self.canvas.bind("<Shift-Button-5>", lambda e: self.change_brush(-2))

    def _build_statusbar(self) -> None:
        statusbar = ttk.Frame(self, padding=(8, 4))
        statusbar.grid(row=3, column=0, sticky="ew")
        statusbar.columnconfigure(1, weight=1)
        ttk.Label(statusbar, textvariable=self.position_var, width=24).grid(row=0, column=0, sticky="w")
        ttk.Label(statusbar, textvariable=self.file_var).grid(row=0, column=1, sticky="w")
        ttk.Label(statusbar, textvariable=self.mask_src_var).grid(row=0, column=2, sticky="e", padx=(0, 12))
        ttk.Label(statusbar, textvariable=self.state_var).grid(row=0, column=3, sticky="e")

    def _build_help(self) -> None:
        help_text = (
            "Enter: accept & next    Left/Right: section    t: Otsu    i: invert    "
            "r: rotate 90    h/v: flip    Left-drag: draw    Right-drag: erase    e: draw/erase\n"
            "f: fill    L: brush selection    Right-drag in Brush select: erase selection    Ctrl+Z/u: undo    Ctrl+Y/y: redo    p: colour pick (left-drag include, right-drag exclude)    c: clear    E/D: erode/dilate    k: clean whole mask    [ ]: brush    brush box: type px    s: save    "
            "Wheel: zoom    Shift+Wheel: brush    +/-: zoom    0: fit    Middle/Ctrl-drag: pan    Esc/q: quit\n"
            "Shift+left-drag: rectangle ROI    Shift+right-drag: lasso ROI    Shift-click: activate/clear ROI    Backspace/Delete: delete ROI\n"
            "(moving the threshold sliders replaces hand edits)"
        )
        ttk.Label(self, text=help_text, padding=(8, 2)).grid(row=4, column=0, sticky="w")
