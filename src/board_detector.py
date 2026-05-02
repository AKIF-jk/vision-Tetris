# =============================================================================
# src/board_detector.py — Task 4: Board Region Detection
#
# Input:  Raw screenshot (BGR numpy array)
# Output: Cropped board image, bounding box, perspective transform matrix
#
# Since coordinates are fixed and known from config.py, we use them directly
# instead of running contour detection every frame (faster + more reliable).
# Perspective warp is still applied in case the game window is slightly skewed.
# =============================================================================

import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_transform_matrix() -> np.ndarray:
    """
    Compute the perspective transform matrix from the 4 known board corners
    to a clean rectangle of the same width/height.

    Returns:
        M : 3x3 perspective transform matrix (float32)
    """
    src_pts = np.float32([
        config.BOARD_CORNERS[0],   # top-left
        config.BOARD_CORNERS[1],   # top-right
        config.BOARD_CORNERS[2],   # bottom-left
        config.BOARD_CORNERS[3],   # bottom-right
    ])

    # Destination: a clean upright rectangle
    w = config.BOARD_WIDTH_PX
    h = config.BOARD_HEIGHT_PX
    dst_pts = np.float32([
        [0, 0],       # top-left
        [w, 0],       # top-right
        [0, h],       # bottom-left
        [w, h],       # bottom-right
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M


def detect_board(screenshot: np.ndarray,
                 save_debug: bool = True,
                 debug_prefix: str = "frame") -> tuple:
    """
    Crop and warp the board region from the full screenshot.

    Args:
        screenshot   : Full BGR screenshot
        save_debug   : Save cropped board to debug folder
        debug_prefix : Prefix for debug filenames

    Returns:
        board_img    : Warped BGR image of just the board (BOARD_HEIGHT_PX x BOARD_WIDTH_PX)
        bbox         : (x_min, y_min, x_max, y_max) bounding box in original image
        M            : Perspective transform matrix (use cv2.perspectiveTransform to map points back)
    """
    M    = get_transform_matrix()
    w    = int(config.BOARD_WIDTH_PX)
    h    = int(config.BOARD_HEIGHT_PX)

    # Apply perspective warp
    board_img = cv2.warpPerspective(screenshot, M, (w, h))

    bbox = (config.BOARD_X_MIN, config.BOARD_Y_MIN,
            config.BOARD_X_MAX, config.BOARD_Y_MAX)

    if save_debug:
        os.makedirs(config.DEBUG_DIR, exist_ok=True)
        # Save board with grid overlay
        debug_img = _draw_grid_overlay(board_img.copy())
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_4_board.jpg",      board_img)
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_5_board_grid.jpg", debug_img)

        # Also draw bounding box on original screenshot
        annotated = screenshot.copy()
        cv2.rectangle(annotated,
                      (config.BOARD_X_MIN, config.BOARD_Y_MIN),
                      (config.BOARD_X_MAX, config.BOARD_Y_MAX),
                      (0, 255, 0), 2)
        cv2.rectangle(annotated,
                      (config.NEXT_X_MIN, config.NEXT_Y_MIN),
                      (config.NEXT_X_MAX, config.NEXT_Y_MAX),
                      (255, 165, 0), 2)
        cv2.putText(annotated, "BOARD", (config.BOARD_X_MIN + 5, config.BOARD_Y_MIN + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, "NEXT", (config.NEXT_X_MIN + 5, config.NEXT_Y_MIN + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_4_regions.jpg", annotated)

    return board_img, bbox, M


def detect_next_piece_region(screenshot: np.ndarray,
                              save_debug: bool = True,
                              debug_prefix: str = "frame") -> np.ndarray:
    """
    Crop and warp the next-piece preview box.

    Args:
        screenshot   : Full BGR screenshot

    Returns:
        next_img     : Warped BGR image of the next-piece box
    """
    src_pts = np.float32([
        config.NEXT_PIECE_CORNERS[0],   # top-left
        config.NEXT_PIECE_CORNERS[1],   # top-right
        config.NEXT_PIECE_CORNERS[2],   # bottom-left
        config.NEXT_PIECE_CORNERS[3],   # bottom-right
    ])

    w = config.NEXT_X_MAX - config.NEXT_X_MIN
    h = config.NEXT_Y_MAX - config.NEXT_Y_MIN

    dst_pts = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h],
    ])

    M_next   = cv2.getPerspectiveTransform(src_pts, dst_pts)
    next_img = cv2.warpPerspective(screenshot, M_next, (w, h))

    if save_debug:
        os.makedirs(config.DEBUG_DIR, exist_ok=True)
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_6_next_piece_region.jpg", next_img)

    return next_img


def _draw_grid_overlay(board_img: np.ndarray) -> np.ndarray:
    """Draw the 20x10 grid lines on top of the board image for debugging."""
    h, w = board_img.shape[:2]
    cell_w = w / config.GRID_COLS
    cell_h = h / config.GRID_ROWS

    # Vertical lines
    for col in range(config.GRID_COLS + 1):
        x = int(col * cell_w)
        cv2.line(board_img, (x, 0), (x, h), (50, 50, 50), 1)

    # Horizontal lines
    for row in range(config.GRID_ROWS + 1):
        y = int(row * cell_h)
        cv2.line(board_img, (0, y), (w, y), (50, 50, 50), 1)

    return board_img


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    import glob
    screenshots = glob.glob(f"{config.INPUT_DIR}/*.jpg") + glob.glob(f"{config.INPUT_DIR}/*.png")
    if not screenshots:
        print("No screenshots found in", config.INPUT_DIR)
    else:
        img = cv2.imread(screenshots[0])
        board, bbox, M = detect_board(img, save_debug=True, debug_prefix="test")
        next_reg       = detect_next_piece_region(img, save_debug=True, debug_prefix="test")
        print(f"  [board_detector] Board: {board.shape} | BBox: {bbox}")
        print(f"  [board_detector] Next piece region: {next_reg.shape}")
        print("  [board_detector] Done. Check output/debug/ for results.")
