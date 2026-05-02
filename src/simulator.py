# =============================================================================
# src/simulator.py — Phase 3: Board Simulator
#
# Drops a tetromino piece into a copy of the board, merges it, clears full
# lines, and returns the resulting state without mutating the original board.
#
# Public API:
#   simulate_drop(board, piece, col, debug=False) -> Tuple[np.ndarray, int]
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import Tuple

from src.move_generator import _trim


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_column_offsets(piece: np.ndarray) -> Tuple[int, int]:
    """
    Return (left_offset, right_offset) describing how many empty columns exist
    on the left and right of the piece within the 4×4 matrix.

    This is needed so that when the caller requests column=0 we correctly
    align the first *filled* column of the piece with column 0 of the board.

    Args:
        piece: 4×4 binary numpy array (the full rotation matrix).

    Returns:
        (left_offset, right_offset) — number of all-zero columns on each side.
    """
    col_sums = piece.sum(axis=0)
    filled_cols = np.where(col_sums > 0)[0]
    if len(filled_cols) == 0:
        return 0, 0
    left_offset  = int(filled_cols[0])
    right_offset = piece.shape[1] - 1 - int(filled_cols[-1])
    return left_offset, right_offset


def _get_row_offsets(piece: np.ndarray) -> int:
    """
    Return the number of all-zero rows above the first filled row in the piece
    matrix.  Pieces embedded in a 4×4 frame often have padding rows on top.

    Args:
        piece: 4×4 binary numpy array.

    Returns:
        top_offset — number of leading empty rows.
    """
    row_sums = piece.sum(axis=1)
    filled_rows = np.where(row_sums > 0)[0]
    if len(filled_rows) == 0:
        return 0
    return int(filled_rows[0])


def _check_collision(
    board: np.ndarray,
    piece: np.ndarray,
    board_row: int,
    board_col: int,
) -> bool:
    """
    Return True if placing the piece (anchored at board_row, board_col)
    overlaps any filled cell or falls outside the board boundaries.

    Vectorised: clips the piece-vs-board overlap to the valid region and
    uses a single numpy ``np.any`` call to detect filled-cell overlap.
    Out-of-bounds cells are detected by checking whether piece cells exist
    in the cropped-away margins.

    Args:
        board     : 20×10 binary numpy array (0=empty, 1=filled).
        piece     : 4×4 binary numpy array.
        board_row : Top-left row of the piece matrix on the board (may be < 0).
        board_col : Top-left column of the piece matrix on the board.

    Returns:
        True if collision detected, False otherwise.
    """
    board_rows, board_cols = board.shape
    ph, pw = piece.shape

    # Valid piece-row range that maps inside the board
    p_r0 = max(0, -board_row)
    p_r1 = min(ph, board_rows - board_row)
    p_c0 = max(0, -board_col)
    p_c1 = min(pw, board_cols - board_col)

    # If the valid inner region is empty the piece is entirely out-of-bounds
    if p_r0 >= p_r1 or p_c0 >= p_c1:
        return bool(piece.any())

    # Boundary collision: filled piece cells fall outside the valid crop
    if (p_r0 > 0  and piece[:p_r0,  :].any()):   return True
    if (p_r1 < ph and piece[p_r1:,  :].any()):   return True
    if (p_c0 > 0  and piece[:,  :p_c0].any()):   return True
    if (p_c1 < pw and piece[:, p_c1:].any()):    return True

    # Overlap collision: any filled piece cell sits on a filled board cell
    b_r0 = board_row + p_r0
    b_r1 = board_row + p_r1
    b_c0 = board_col + p_c0
    b_c1 = board_col + p_c1
    return bool(np.any(piece[p_r0:p_r1, p_c0:p_c1]
                       & board[b_r0:b_r1, b_c0:b_c1]))


def _find_drop_row(
    board: np.ndarray,
    piece: np.ndarray,
    anchor_col: int,
) -> int:
    """
    Analytically compute the landing anchor row for the piece without
    stepping through gravity one row at a time.

    For each piece column that contains filled cells, the bottommost filled
    piece row in that column constrains how far the piece can descend before
    it would overlap a board cell.  The landing row is the minimum across
    all piece columns of that constraint.

    Args:
        board      : 20×10 board (unmodified).
        piece      : 4×4 piece matrix.
        anchor_col : Left column of the 4×4 matrix on the board
                     (may be negative due to left_offset).

    Returns:
        Landing anchor row (int) — the topmost row of the 4×4 matrix when
        the piece has been dropped as far as possible.
    """
    board_rows, board_cols = board.shape
    ph, pw = piece.shape

    # Pre-compute column heights: index of first filled row from the top,
    # or board_rows if the column is empty.
    col_height = np.full(board_cols, board_rows, dtype=np.int32)
    for c in range(board_cols):
        filled = np.nonzero(board[:, c])[0]
        if len(filled):
            col_height[c] = int(filled[0])

    # One constraint per piece column that has filled cells
    max_anchor_row = board_rows - 1  # absolute ceiling (never exceeded)

    for pc in range(pw):
        bc = anchor_col + pc
        if bc < 0 or bc >= board_cols:
            continue
        col_cells = np.nonzero(piece[:, pc])[0]
        if not len(col_cells):
            continue
        bottom_pr = int(col_cells[-1])
        # Piece cell at (bottom_pr, pc) must land above board cell at col_height[bc]
        # anchor_row + bottom_pr <= col_height[bc] - 1
        max_for_col = col_height[bc] - bottom_pr - 1
        if max_for_col < max_anchor_row:
            max_anchor_row = max_for_col

    return max_anchor_row


def _merge_piece(
    board: np.ndarray,
    piece: np.ndarray,
    board_row: int,
    board_col: int,
) -> np.ndarray:
    """
    Return a new board with the piece written into it at (board_row, board_col).
    Only filled (=1) cells of the piece are written; zeros are transparent.

    Uses vectorised numpy assignment rather than Python loops.

    Args:
        board     : Original 20×10 board (not mutated).
        piece     : 4×4 binary piece matrix.
        board_row : Anchor row (top-left of piece matrix).
        board_col : Anchor column (top-left of piece matrix).

    Returns:
        New board array with the piece merged in.
    """
    new_board = board.copy()
    board_rows, board_cols = new_board.shape
    ph, pw = piece.shape

    # Crop indices (same logic as _check_collision)
    p_r0 = max(0, -board_row);  p_r1 = min(ph, board_rows - board_row)
    p_c0 = max(0, -board_col);  p_c1 = min(pw, board_cols - board_col)
    if p_r0 >= p_r1 or p_c0 >= p_c1:
        return new_board

    b_r0 = board_row + p_r0;  b_r1 = board_row + p_r1
    b_c0 = board_col + p_c0;  b_c1 = board_col + p_c1

    piece_crop = piece[p_r0:p_r1, p_c0:p_c1]
    mask = piece_crop == 1
    new_board[b_r0:b_r1, b_c0:b_c1][mask] = 1
    return new_board


def _clear_lines(board: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Remove all completely filled rows from the board and shift the remaining
    rows downward (gravity), filling the top with empty rows.

    Args:
        board: 20×10 binary numpy array (may be mutated internally; a copy is
               made inside simulate_drop before reaching here).

    Returns:
        (new_board, lines_cleared) — updated board and count of removed rows.
    """
    board_rows, board_cols = board.shape

    full_rows = np.where(board.sum(axis=1) == board_cols)[0]
    lines_cleared = len(full_rows)

    if lines_cleared == 0:
        return board, 0

    # Keep only non-full rows
    kept_rows = board[board.sum(axis=1) < board_cols]

    # Prepend empty rows to restore height
    empty_rows = np.zeros((lines_cleared, board_cols), dtype=board.dtype)
    new_board = np.vstack([empty_rows, kept_rows])

    return new_board, lines_cleared


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_drop(
    board: np.ndarray,
    piece: np.ndarray,
    col: int,
    debug: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Simulate dropping a tetromino piece into the board at the given column,
    then clear any complete lines and return the resulting board state.

    The original *board* is **never mutated**.

    Gravity model:
        The piece starts at row -(top_offset) so that only its actual filled
        cells are above the board.  It then falls one row at a time until a
        collision is detected, and is placed in the last non-colliding row.

    Column model:
        *col* refers to the left edge of the *trimmed* (tight) bounding box of
        the piece, matching the convention used by move_generator.get_rotations.
        Internally we convert this to the top-left corner of the full 4×4
        frame using the piece's left_offset.

    Args:
        board : 20×10 binary numpy array (0=empty, 1=filled).
        piece : 4×4 binary numpy array (rotation matrix from move_generator).
        col   : Target column for the left edge of the trimmed piece bounding
                box (0-indexed).
        debug : If True, print drop trace to stdout.

    Returns:
        (new_board, lines_cleared) where:
            new_board     — 20×10 numpy array after placement and line clearing.
            lines_cleared — number of full rows removed (0–4).

    Raises:
        ValueError if the piece has no filled cells.

    Edge cases handled:
        * Piece that collides immediately at spawn height → board unchanged,
          lines_cleared = 0.
        * No valid moves (full board) → handled by the engine layer, but this
          function returns the original board if spawn collision occurs.
        * col that would place the piece partly outside the board → collision
          detected immediately.
    """
    if board.ndim != 2 or board.shape[1] == 0:
        raise ValueError(f"Invalid board shape: {board.shape}")

    piece = np.asarray(piece, dtype=np.uint8)
    if piece.sum() == 0:
        raise ValueError("Piece matrix is empty (all zeros).")

    # Offsets within the 4×4 matrix
    left_offset, _  = _get_column_offsets(piece)
    top_offset      = _get_row_offsets(piece)

    # Convert trimmed-bounding-box column to full-matrix anchor column
    anchor_col = col - left_offset

    # Start the piece just above the board so the top *filled* row aligns
    # with board row 0
    start_row = -top_offset

    if debug:
        print(f"\n  [simulator] drop col={col}  anchor_col={anchor_col}  "
              f"left_offset={left_offset}  top_offset={top_offset}")

    # ── Gravity: compute landing row analytically ────────────────────────────
    drop_row = _find_drop_row(board, piece, anchor_col)

    if debug:
        print(f"  [simulator] piece lands at anchor_row={drop_row}")

    # ── Spawn collision check ────────────────────────────────────────────────
    # If the piece collides even at its starting position, the board is full
    # or the move is invalid.
    if _check_collision(board, piece, drop_row, anchor_col):
        if debug:
            print("  [simulator] spawn collision — board full or invalid move")
        return board.copy(), 0

    # ── Merge piece into board ───────────────────────────────────────────────
    placed_board = _merge_piece(board, piece, drop_row, anchor_col)

    if debug:
        print(f"  [simulator] piece placed at row={drop_row}, col={anchor_col}")

    # ── Clear full lines ─────────────────────────────────────────────────────
    final_board, lines_cleared = _clear_lines(placed_board)

    if debug:
        print(f"  [simulator] lines cleared: {lines_cleared}")

    return final_board, lines_cleared


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== simulator self-test ===\n")

    I_PIECE = np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], dtype=np.uint8)
    O_PIECE = np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]], dtype=np.uint8)

    # Test 1: I-piece drop into empty board → should sit on bottom
    empty = np.zeros((20, 10), dtype=np.uint8)
    b1, lc1 = simulate_drop(empty, I_PIECE, col=0, debug=False)
    assert b1[19, 0] == 1, "I-piece should reach the bottom row"
    assert lc1 == 0, "No lines cleared on empty board"
    print("  ✓ Test 1: I-piece lands on empty board")

    # Test 2: fill 9/10 cells in 4 rows (leave col 9 empty), drop vertical I
    # to fill col 9 across all 4 rows → 4 lines cleared
    I_VERT = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]], dtype=np.uint8)
    board2 = np.zeros((20, 10), dtype=np.uint8)
    board2[16:20, :9] = 1       # 9 cells filled per row in rows 16-19
    # vertical I-piece at col 9 fills the last cell in each of the 4 rows
    b2, lc2 = simulate_drop(board2, I_VERT, col=9, debug=False)
    assert lc2 == 4, f"Expected 4 lines cleared, got {lc2}"
    print("  ✓ Test 2: 4 lines cleared")

    # Test 3: full board at top → spawn collision, original returned
    full_board = np.ones((20, 10), dtype=np.uint8)
    b3, lc3 = simulate_drop(full_board, O_PIECE, col=0, debug=False)
    assert lc3 == 0
    print("  ✓ Test 3: spawn collision on full board handled")

    # Test 4: piece with column offset placed at col 0
    T_PIECE = np.array([[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,0]], dtype=np.uint8)
    b4, lc4 = simulate_drop(empty, T_PIECE, col=0, debug=False)
    assert b4[19, 0] == 1 or b4[18, 0] == 1
    print("  ✓ Test 4: T-piece placed at col 0")

    print("\n  Result: ALL PASS\n")
