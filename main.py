# =============================================================================
# main.py — Entry point for Tetris Advisor (Phase 1 + 2)
#
# Usage:
#   python main.py --input input/screenshots/test_01.jpg
#   python main.py --input input/screenshots/test_01.jpg --no-debug
# =============================================================================

import argparse
import time
import cv2
import os
import sys
import numpy as np

import config
from src.preprocess      import load_image, preprocess
from src.board_detector  import detect_board, detect_next_piece_region
from src.grid_parser     import parse_grid, print_grid
from src.piece_detector  import detect_piece, detect_next_piece
from src.engine          import best_move, print_scores_matrix
from src.simulator       import simulate_drop, _get_column_offsets, _get_row_offsets


def run_pipeline(input_path: str, debug: bool = None, verbose: bool = None):
    # Use config defaults if not specified
    if debug is None:   debug = config.DEBUG
    if verbose is None: verbose = config.VERBOSE

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Tetris Advisor — Processing: {input_path}")
        print(f"{'='*60}")

    prefix = os.path.splitext(os.path.basename(input_path))[0]
    t0 = time.time()

    # ── Step 1: Load & preprocess ──────────────────────────────────────────
    t = time.time()
    img    = load_image(input_path, verbose=verbose)
    binary = preprocess(img, save_debug=debug, debug_prefix=prefix)
    if verbose: print(f"  ✓ Preprocess        {(time.time()-t)*1000:.0f}ms")

    # ── Step 2: Detect board region ────────────────────────────────────────
    t = time.time()
    board, bbox, M = detect_board(img, save_debug=debug, debug_prefix=prefix)
    if verbose: print(f"  ✓ Board detection   {(time.time()-t)*1000:.0f}ms")

    # ── Step 3: Parse grid ─────────────────────────────────────────────────
    t = time.time()
    grid = parse_grid(board, save_debug=debug, debug_prefix=prefix)
    if verbose: print(f"  ✓ Grid parsing      {(time.time()-t)*1000:.0f}ms")

    # ── Step 4: Detect active piece ────────────────────────────────────────
    t = time.time()
    piece_type, shape, pos, rot = detect_piece(
        board, grid, save_debug=debug, debug_prefix=prefix)
    if verbose: print(f"  ✓ Active piece      {(time.time()-t)*1000:.0f}ms")

    # ── Step 5: Detect next piece ──────────────────────────────────────────
    t = time.time()
    next_region = detect_next_piece_region(img, save_debug=debug, debug_prefix=prefix)
    next_type, next_shape, _ = detect_next_piece(
        next_region, save_debug=debug, debug_prefix=prefix)
    if verbose: print(f"  ✓ Next piece        {(time.time()-t)*1000:.0f}ms")

    # ── Step 6: Two-piece lookahead decision engine ─────────────────────────
    t = time.time()
    best_rotation, best_col, scores_matrix = best_move(
        grid, shape, debug=False)
    if verbose: print(f"  ✓ Engine lookahead  {(time.time()-t)*1000:.0f}ms")

    total = time.time() - t0
    
    # ── Simulate the best move ─────────────────────────────────────────────
    board_after_move, lines_cleared = simulate_drop(grid, best_rotation, best_col)
    
    # ── Final Output (Clean) ──────────────────────────────────────────────
    print(f"\n[PIPELINE] Detected: Active={piece_type} (pos={pos}), Next={next_type} | Time: {total*1000:.0f}ms")
    print(f"[ENGINE]   Best move: rotation col={best_col}  |  Lines cleared: {lines_cleared}")
    
    print(f"\n{'-'*60}")
    print(f"  ORIGINAL BOARD STATE")
    print(f"{'-'*60}")
    print_grid(grid)
    
    print(f"\n{'-'*60}")
    print(f"  BOARD STATE AFTER BEST MOVE (col={best_col})")
    print(f"{'-'*60}")
    print_grid(board_after_move)

    if debug:
        print(f"\n  Debug images saved to: {config.DEBUG_DIR}/")
        print("\n[SCORES MATRIX]")
        print_scores_matrix(scores_matrix)

    return {
        "grid":              grid,
        "piece_type":        piece_type,
        "piece_shape":       shape,
        "piece_pos":         pos,
        "next_type":         next_type,
        "next_shape":        next_shape,
        "best_rotation":     best_rotation,
        "best_col":          best_col,
        "scores_matrix":     scores_matrix,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetris Advisor — DIP Project")
    parser.add_argument("--input",    required=True, help="Path to screenshot")
    parser.add_argument("--debug",    action="store_true", help="Save intermediate debug images")
    parser.add_argument("--verbose",  action="store_true", help="Print detailed per-step logs")
    args = parser.parse_args()

    # CLI args override config defaults if specified
    run_pipeline(args.input, 
                 debug=args.debug if args.debug else None,
                 verbose=args.verbose if args.verbose else None)
