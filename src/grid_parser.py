# =============================================================================
# src/grid_parser.py — Task 5: Grid Cell Parser
#
# Fix: Detects BOTH colored blocks (high saturation) AND
#      white/gray blocks (low saturation, high brightness).
#      Cell is filled only if >=50% of its pixels qualify.
# =============================================================================

import cv2
import numpy as np
import os, sys
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

FILL_THRESHOLD = 0.50


def get_block_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask of ALL block pixels — colored OR white/gray.

    Colored blocks : S >= MIN_SATURATION  AND V >= MIN_VALUE
    Gray/white blocks: S <= MAX_SAT_GRAY  AND V >= MIN_VAL_GRAY
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # (a) Saturated colored blocks
    mask_color = cv2.inRange(
        hsv,
        (0,   config.MIN_SATURATION, config.MIN_VALUE),
        (180, 255,                   255)
    )

    # (b) Gray / white blocks (low saturation, high brightness)
    mask_gray = cv2.inRange(
        hsv,
        (0,   0,                   config.MIN_VAL_GRAY),
        (180, config.MAX_SAT_GRAY, 255)
    )

    return cv2.bitwise_or(mask_color, mask_gray)


def parse_grid(board_img: np.ndarray,
               save_debug: bool = True,
               debug_prefix: str = "frame") -> np.ndarray:
    """
    Parse board into a 20x10 binary occupancy matrix.
    Cell is FILLED (1) if >=50% of its inner pixels are block pixels.
    """
    h, w = board_img.shape[:2]
    cw = w / config.GRID_COLS
    ch = h / config.GRID_ROWS

    block_mask  = get_block_mask(board_img)
    grid        = np.zeros((config.GRID_ROWS, config.GRID_COLS), dtype=np.uint8)
    fill_ratios = np.zeros((config.GRID_ROWS, config.GRID_COLS), dtype=np.float32)

    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            x1 = int(col       * cw) + 4
            x2 = int((col + 1) * cw) - 4
            y1 = int(row       * ch) + 4
            y2 = int((row + 1) * ch) - 4

            cell   = block_mask[y1:y2, x1:x2]
            total  = cell.size
            filled = int(np.count_nonzero(cell))
            ratio  = filled / total if total > 0 else 0.0

            fill_ratios[row, col] = ratio
            grid[row, col]        = 1 if ratio >= FILL_THRESHOLD else 0

    if save_debug:
        os.makedirs(config.DEBUG_DIR, exist_ok=True)
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_mask_combined.jpg", block_mask)
        debug_img = _draw_grid_debug(board_img.copy(), grid, fill_ratios)
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_7_grid_parsed.jpg", debug_img)

    return grid


def get_cell_color(board_img: np.ndarray, row: int, col: int) -> tuple:
    h, w = board_img.shape[:2]
    x1 = int(col       * w / config.GRID_COLS) + 5
    x2 = int((col + 1) * w / config.GRID_COLS) - 5
    y1 = int(row       * h / config.GRID_ROWS) + 5
    y2 = int((row + 1) * h / config.GRID_ROWS) - 5
    mean = cv2.mean(board_img[y1:y2, x1:x2])[:3]
    return tuple(int(c) for c in mean)


def _draw_grid_debug(board_img, grid, fill_ratios):
    h, w = board_img.shape[:2]
    cw = w / config.GRID_COLS
    ch = h / config.GRID_ROWS
    overlay = board_img.copy()
    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            x1,x2 = int(col*cw), int((col+1)*cw)
            y1,y2 = int(row*ch), int((row+1)*ch)
            if grid[row, col] == 1:
                cv2.rectangle(overlay,(x1+1,y1+1),(x2-1,y2-1),(0,220,0),-1)
            cv2.rectangle(board_img,(x1,y1),(x2,y2),(50,50,50),1)
            pct = f"{int(fill_ratios[row,col]*100)}"
            cv2.putText(board_img, pct, (x1+3, y2-5),
                        cv2.FONT_HERSHEY_PLAIN, 0.6,
                        (200,200,200) if grid[row,col]==0 else (0,0,0), 1)
    return cv2.addWeighted(overlay, 0.25, board_img, 0.75, 0)


def print_grid(grid: np.ndarray):
    print("\n  Board state (X=filled):")
    print("  +" + "--+" * config.GRID_COLS)
    for row in range(config.GRID_ROWS):
        print("  |" + "".join(" X|" if grid[row,c]==1 else "  |"
                               for c in range(config.GRID_COLS)))
    print("  +" + "--+" * config.GRID_COLS)


def remove_active_piece_cells(grid: np.ndarray, cells: set) -> np.ndarray:
    """
    Remove active piece cells from the grid, returning only settled blocks.

    Args:
        grid: 20x10 binary occupancy grid (may include active piece cells).
        cells: Set of (row, col) tuples representing active piece cells.

    Returns:
        Clean 20x10 grid with active piece cells set to 0.
    """
    clean = grid.copy()
    for r, c in cells:
        if 0 <= r < config.GRID_ROWS and 0 <= c < config.GRID_COLS:
            clean[r, c] = 0
    return clean


def validate_grid(grid: np.ndarray, verbose: bool = False) -> tuple:
    """
    Validate that the grid represents a physically possible Tetris board state
    using Tetris game logic.

    In standard Tetris, blocks are placed from above and settle onto existing
    blocks or the floor. After line clears, all rows above shift down.
    This means every filled cell must be supported — either by the bottom row,
    by a filled cell directly below, or through a connected path of filled
    cells that ultimately reaches the bottom.

    Checks:
        1. Correct dimensions (20x10).
        2. No floating blocks: every filled cell must be connected to the
           bottom row via a 4-directional path of adjacent filled cells.

    Returns:
        (is_valid: bool, reason: str)
    """
    if grid.shape != (config.GRID_ROWS, config.GRID_COLS):
        return False, f"Invalid dimensions: {grid.shape}"

    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    q = deque()

    # Seed BFS from bottom-row filled cells
    for c in range(cols):
        if grid[rows - 1, c] == 1:
            visited[rows - 1, c] = True
            q.append((rows - 1, c))

    # Also seed every filled cell that has a filled cell directly below
    # (these are supported from beneath)
    for r in range(rows - 1):
        for c in range(cols):
            if grid[r, c] == 1 and grid[r + 1, c] == 1 and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))

    # BFS through all adjacent filled cells
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1 and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))

    # Any unvisited filled cell is floating — impossible in real Tetris
    unvisited = np.where((grid == 1) & (~visited))
    if len(unvisited[0]) > 0:
        floating = list(zip(unvisited[0].tolist(), unvisited[1].tolist()))
        if verbose:
            for r, c in floating[:10]:
                print(f"  [validate_grid] Floating cell at ({r}, {c})")
        return False, f"Grid has {len(floating)} floating block(s) — physically impossible in Tetris"

    return True, "Valid Tetris board state"