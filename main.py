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

import config
from src.preprocess      import load_image, preprocess
from src.board_detector  import detect_board, detect_next_piece_region
from src.grid_parser     import parse_grid, print_grid
from src.piece_detector  import detect_piece, detect_next_piece


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

    total = time.time() - t0
    
    # ── Final Output (Clean) ──────────────────────────────────────────────
    print(f"\n[PIPELINE] Detected: Active={piece_type} (pos={pos}), Next={next_type} | Time: {total*1000:.0f}ms")
    print_grid(grid)

    if debug:
        print(f"  Debug images saved to: {config.DEBUG_DIR}/")

    return {
        "grid":        grid,
        "piece_type":  piece_type,
        "piece_shape": shape,
        "piece_pos":   pos,
        "next_type":   next_type,
        "next_shape":  next_shape,
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
