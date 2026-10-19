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
TOOLS = ("draw", "erase", "fill", "select", "pick")
ZOOM_STEP = 1.25         # one wheel notch / key press
MIN_ZOOM = 0.25          # relative to the fit-to-window scale
MAX_ZOOM = 8.0
MAX_VIEW_PIXELS = 12_000_000
SHIFT_MASK = 0x0001
CONTROL_MASK = 0x0004
SLIDER_UPDATE_MS = 30    # coalesce slider/stroke render events
BG_TOLERANCE = 12        # gray levels from the border median that count as background
BG_MAX_TOLERANCE = 48    # cap for the automatic widening on a noisy border
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
KEEP_BRUSH_STROKES = True  # default for "Keep brush strokes": Clean never removes hand-painted pixels
SNAP_BRUSH = True        # default for "Snap brush to edges": brush strokes settle on the nearest strong edge
SNAP_RING = 3            # px ring just outside a stroke used as "not this label" watershed markers
SNAP_CORE_FRACTION = 1 / 3  # stroke centre line (this fraction of the brush radius) is always kept
SNAP_EDGE_MIN = 10.0     # gray levels/px; weaker edges than this do not trim a stroke
PICK_SIGMA = 1.5         # px; the image is smoothed before colour picking (ignores grain/noise)
PICK_SPREAD = 2.5        # picked range = median +/- this many robust SDs of the scribbled pixels
PICK_MIN_SPREAD = 8.0    # gray levels; minimum half-width of the picked range
PICK_MAX_SPREAD = 18.0   # gray levels; maximum half-width of the picked range
PICK_OPEN = 2            # px; opening radius that cuts thin bridges (e.g. pale cortical layers)
PICK_MIN_REGION = 50     # px; picked regions smaller than this are ignored
PICK_MAX_HOLE = 400      # px; holes up to this size inside a picked block are filled (vessels, streaks)
BLOCK_COLOUR = (15, 35, 170)  # transparent dark blue: regions rejected with the colour picker
BLOCK_ALPHA_GAIN = 1.4        # blue is drawn a bit more opaque than the red mask so it reads on pale tissue
BG_CHROMA_TOLERANCE = 4  # colour units (blue vs red/green, red vs green); tinted pixels are tissue, not background

# --- UNet predictions ------------------------------------------------------- #
UNET_OUTPUT_DIR = "UNet_output"   # inside segmentations/, one <group>/<base>.png per prediction
UNET_MODEL_DIR = "bbdb_model_504"
UNET_PYTHON = ".venv-nnunet/bin/python"
UNET_CHECKPOINT = "checkpoint_final.pth"
UNET_DEVICE = "auto"
UNET_POLL_MS = 100
UNET_PREDICTION_COLOUR = (255, 205, 0)   # yellow: model output, distinct from the red manual mask
UNET_BADGE_COLOUR = "#ffd600"             # same yellow for the status-line "predicted"
