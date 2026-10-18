"""Brush size, tool mode and cursor helpers."""
from __future__ import annotations

import tkinter as tk

from .settings import MAX_BRUSH, MIN_BRUSH, SHIFT_MASK, TOOLS, ZOOM_STEP


class ToolsMixin:

    def change_brush(self, delta: int) -> None:
        self.brush = int(min(MAX_BRUSH, max(MIN_BRUSH, self.brush + delta)))
        self.update_brush_label()

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self.update_mode_label()
        self.canvas.configure(cursor="crosshair" if tool == "fill" else "")
        if tool == "fill":
            self.on_leave(None)

    def cycle_tool(self) -> None:
        self.set_tool(TOOLS[(TOOLS.index(self.tool) + 1) % len(TOOLS)])

    def toggle_erase(self) -> None:
        self.set_tool("draw" if self.tool == "erase" else "erase")

    def toggle_fill(self) -> None:
        self.set_tool("draw" if self.tool == "fill" else "fill")

    def update_brush_label(self) -> None:
        self.brush_var.set(f"brush: {self.brush}px")

    def update_mode_label(self) -> None:
        self.mode_button.configure(text=f"Mode: {self.tool.capitalize()}")

    def update_cursor(self, x: int, y: int, erase: bool) -> None:
        radius = self.brush
        colour = "#ff4040" if erase else "#ffff40"
        if self.cursor_id is None:
            self.cursor_id = self.canvas.create_oval(0, 0, 0, 0, outline=colour, width=2)
        self.canvas.coords(self.cursor_id, x - radius, y - radius, x + radius, y + radius)
        self.canvas.itemconfigure(self.cursor_id, outline=colour)
        self.canvas.tag_raise(self.cursor_id)

    def on_motion(self, event: tk.Event) -> None:
        self.last_mouse = (event.x, event.y)
        if self.panning or self.roi_dragging:
            return
        if self.tool != "fill":
            self.update_cursor(event.x, event.y, self.tool == "erase")

    def on_leave(self, _event: tk.Event | None) -> None:
        self.last_mouse = None
        if self.cursor_id is not None:
            self.canvas.delete(self.cursor_id)
            self.cursor_id = None

    def on_wheel(self, event: tk.Event) -> None:
        if event.state & SHIFT_MASK:
            self.change_brush(2 if event.delta > 0 else -2)
            return
        if event.delta == 0:
            return
        self.zoom_at(event.x, event.y, ZOOM_STEP if event.delta > 0 else 1.0 / ZOOM_STEP)
