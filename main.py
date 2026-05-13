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
from src.grid_parser     import parse_grid, print_grid, remove_active_piece_cells, validate_grid
from src.piece_detector  import detect_piece, detect_next_piece
from src.engine          import best_move, print_scores_matrix
from src.simulator       import simulate_drop, _get_column_offsets, _get_row_offsets
from src.visualizer      import annotate_move, draw_heatmap, rotation_degrees


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
    piece_type, shape, pos, rot, cells, actual_shape = detect_piece(
        board, grid, save_debug=debug, debug_prefix=prefix)
    if verbose: print(f"  ✓ Active piece      {(time.time()-t)*1000:.0f}ms")

    # ── Step 4b: Remove active piece cells from grid ───────────────────────
    # The raw grid includes active piece blocks; remove them so the engine
    # operates on settled blocks only.
    t = time.time()
    clean_grid = remove_active_piece_cells(grid, cells)
    if verbose: print(f"  ✓ Grid cleaned      {(time.time()-t)*1000:.0f}ms")

    # ── Step 4c: Validate grid using Tetris game logic ─────────────────────
    t = time.time()
    grid_valid, grid_reason = validate_grid(clean_grid, verbose=verbose)
    if verbose:
        status = "✓" if grid_valid else "✗"
        print(f"  {status} Grid validation    {(time.time()-t)*1000:.0f}ms — {grid_reason}")
    if not grid_valid:
        print(f"  [WARNING] {grid_reason}")
    # ── Step 5: Detect next piece ──────────────────────────────────────────
    t = time.time()
    next_region = detect_next_piece_region(img, save_debug=debug, debug_prefix=prefix)
    next_type, next_shape, _ = detect_next_piece(
        next_region, save_debug=debug, debug_prefix=prefix)
    if verbose: print(f"  ✓ Next piece        {(time.time()-t)*1000:.0f}ms")

    # ── Step 6: Two-piece lookahead decision engine ─────────────────────────
    # Use clean_grid (without active piece) for accurate simulation
    t = time.time()
    best_rotation, best_col, scores_matrix = best_move(
        clean_grid, shape, debug=False)
    if verbose: print(f"  ✓ Engine lookahead  {(time.time()-t)*1000:.0f}ms")

    # ── Simulate the best move ─────────────────────────────────────────────
    board_after_move, lines_cleared = simulate_drop(clean_grid, best_rotation, best_col)

    # ── Step 7: Render final overlays ──────────────────────────────────────
    t = time.time()
    finite_scores = scores_matrix[np.isfinite(scores_matrix)]
    best_score = float(finite_scores.max()) if finite_scores.size else None
    rotation_label = rotation_degrees(actual_shape, best_rotation)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.HEATMAP_DIR, exist_ok=True)

    annotated_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_annotated.jpg")
    heatmap_path = os.path.join(config.HEATMAP_DIR, f"{prefix}_heatmap.jpg")

    annotate_move(
        img,
        best_rotation,
        best_col,
        bbox,
        board=clean_grid,
        score=best_score,
        rotation_label=rotation_label,
        output_path=annotated_path,
    )
    draw_heatmap(img, scores_matrix, bbox, output_path=heatmap_path)
    if verbose: print(f"  ✓ Visualization    {(time.time()-t)*1000:.0f}ms")

    total = time.time() - t0
    
    # ── Final Output (Clean) ──────────────────────────────────────────────
    print(f"\n[PIPELINE] Detected: Active={piece_type} (pos={pos}), Next={next_type} | Time: {total*1000:.0f}ms")
    print(f"[ENGINE]   Best move: rotation col={best_col}  |  Lines cleared: {lines_cleared}")
    
    print(f"\n{'-'*60}")
    print(f"  SETTLED BOARD STATE (active piece removed)")
    print(f"{'-'*60}")
    print_grid(clean_grid)

    if debug:
        print(f"\n{'-'*60}")
        print(f"  RAW BOARD STATE (including active piece)")
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
        "grid":              clean_grid,
        "raw_grid":          grid,
        "piece_type":        piece_type,
        "piece_shape":       shape,
        "piece_pos":         pos,
        "piece_cells":       cells,
        "next_type":         next_type,
        "next_shape":        next_shape,
        "best_rotation":     best_rotation,
        "best_col":          best_col,
        "scores_matrix":     scores_matrix,
        "board_after_move":  board_after_move,
        "best_score":        best_score,
        "grid_valid":        grid_valid,
        "grid_reason":       grid_reason,
        "annotated_path":    annotated_path,
        "heatmap_path":      heatmap_path,
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
