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
BG_TOLERANCE = 12        # gray levels from the border median that count as background
BG_MARGIN = 1            # px grown around background to absorb the blurry tissue rim
CLEAN_BAND = 3           # px band around the mask boundary that edge snapping may move
CLEAN_MIN_ISLAND = 30    # px; smaller connected pieces are removed by Clean
LEGEND_BAND = 0.2        # top/bottom fraction of the slide where labels and scale bars live
LEGEND_MAX_AREA = 0.004  # legend blobs are smaller than this fraction of the image
LEGEND_MARGIN = 2        # px grown around detected text to cover soft letter edges
SMOOTH_STROKES = True    # default for the "Smooth strokes" checkbox
ONE_EURO_MIN_CUTOFF = 1.0   # Hz; smoothing strength for slow strokes (lower = smoother)
ONE_EURO_BETA = 0.05        # speed coefficient; higher = less lag on fast strokes
ONE_EURO_D_CUTOFF = 1.0     # Hz; cutoff for the speed estimate
LASSO_SMOOTH_ITERATIONS = 2  # Chaikin corner-cutting passes applied to lasso outlines
