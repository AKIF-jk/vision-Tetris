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