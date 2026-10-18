"""Shared defaults for the BBDB segmentation GUI."""
from __future__ import annotations

APP_TITLE = "BBDB Segmentation GUI"
WORK_MAX = 1600          # longest side of the in-memory working image
OVERLAY_ALPHA = 110      # 0..255 opacity of the red overlay
MAX_CACHE_ITEMS = 4
SAVE_DEBOUNCE_MS = 400
MAX_LOG_RECORDS = 50
DEFAULT_BRUSH = 14       # brush radius in on-screen pixels
MIN_BRUSH = 2
MAX_BRUSH = 120
UNDO_LIMIT = 30
TOOLS = ("draw", "erase", "fill", "select")
ZOOM_STEP = 1.25         # one wheel notch / key press
MIN_ZOOM = 0.25          # relative to the fit-to-window scale
MAX_ZOOM = 8.0
MAX_VIEW_PIXELS = 12_000_000
SHIFT_MASK = 0x0001
CONTROL_MASK = 0x0004
SLIDER_UPDATE_MS = 30    # coalesce slider/stroke render events
