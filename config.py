# =============================================================================
# config.py — Central configuration for Tetris Advisor
# =============================================================================

# Board region corners (pixels in the raw screenshot)
BOARD_CORNERS = [
    (26,  18),
    (471, 18),
    (26,  635),
    (471, 635),
]
BOARD_X_MIN = 26
BOARD_Y_MIN = 18
BOARD_X_MAX = 471
BOARD_Y_MAX = 635

# Next-piece preview region (UPDATED)
NEXT_PIECE_CORNERS = [
    (470, 440),
    (640, 440),
    (470, 630),
    (640, 630),
]
NEXT_X_MIN = 470
NEXT_Y_MIN = 440
NEXT_X_MAX = 640
NEXT_Y_MAX = 630

# Grid dimensions
GRID_ROWS = 20
GRID_COLS = 10

BOARD_WIDTH_PX  = BOARD_X_MAX - BOARD_X_MIN   # 445
BOARD_HEIGHT_PX = BOARD_Y_MAX - BOARD_Y_MIN   # 617
CELL_W = BOARD_WIDTH_PX  / GRID_COLS
CELL_H = BOARD_HEIGHT_PX / GRID_ROWS

# Preprocessing
GAUSSIAN_BLUR_KERNEL = (5, 5)
BINARY_THRESHOLD     = 30

# -------------------------------------------------------------------
# Pixel classification thresholds
# A pixel counts as "colored block" if EITHER:
#   (a) Saturated color:  S >= MIN_SATURATION and V >= MIN_VALUE
#   (b) White/gray block: S <= MAX_SAT_GRAY   and V >= MIN_VAL_GRAY
# This handles both vivid colored pieces AND gray/white pieces.
# -------------------------------------------------------------------
MIN_SATURATION = 60     # (a) minimum saturation for colored blocks
MIN_VALUE      = 60     # (a) minimum brightness for colored blocks
MAX_SAT_GRAY   = 30     # (b) maximum saturation for gray/white blocks
MIN_VAL_GRAY   = 150    # (b) minimum brightness for gray/white blocks

# Output / display
BEST_MOVE_COLOR   = (0, 255, 0)
GHOST_PIECE_COLOR = (0, 255, 128)
TEXT_COLOR        = (255, 255, 255)
HEATMAP_ALPHA     = 0.4

# Paths
INPUT_DIR   = "input/screenshots"
OUTPUT_DIR  = "output"
DEBUG_DIR   = "output/debug"
HEATMAP_DIR = "output/heatmap"
VIDEO_DIR   = "output/video"

# Video mode
FRAME_DIFF_THRESHOLD = 15

# Runtime controls
DEBUG   = False   # Set to True to save all intermediate DIP images
VERBOSE = False   # Set to True to print detailed per-module logs