# =============================================================================
# src/scorer.py — Phase 3: Dellacherie Heuristic Scorer
#
# Implements the classic Dellacherie evaluation function with all 6 features:
#   1. Landing Height     — vertical centre of the placed piece
#   2. Eroded Cells       — lines_cleared × piece cells erased in those lines
#   3. Row Transitions    — horizontal empty↔filled changes (including walls)
#   4. Column Transitions — vertical   empty↔filled changes (including floor)
#   5. Holes              — empty cells with at least one filled cell above them
#   6. Cumulative Wells   — depths of 1-wide gaps surrounded on both sides
#
# Score = -LandingHeight + ErodedCells - RowTransitions
#         - ColumnTransitions - 4*Holes - Wells
#
# Public API:
#   dellacherie_score(board, piece, lines_cleared, placement_row=None,
#                     placement_col=None, debug=False)
#   -> Tuple[float, Dict[str, float]]
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Feature 1 — Landing Height
# ---------------------------------------------------------------------------

def _landing_height(
    board: np.ndarray,
    piece: np.ndarray,
    placement_row: Optional[int],
    placement_col: Optional[int],
) -> float:
    """
    Compute the vertical position of the placed piece as measured from the
    bottom of the board.

    The landing height is defined as the row index of the *centre of mass*
    of the piece's filled cells after placement, measured from the bottom
    (row 0 = bottom, row 19 = top for a 20-row board).

    If placement_row / placement_col are not provided we fall back to
    inspecting the board directly — we look for the topmost group of cells
    that look like they belong to the piece (within a 4-row window starting
    from the first non-empty row).

    Args:
        board         : Post-placement 20×10 board (BEFORE line clearing).
        piece         : 4×4 piece matrix used for placement.
        placement_row : Top-left row of the 4×4 piece matrix on the board.
        placement_col : Top-left column of the 4×4 piece matrix on the board.

    Returns:
        Landing height as float (distance from bottom, 0-indexed).
    """
    board_rows = board.shape[0]

    if placement_row is not None and placement_col is not None:
        # Collect the board-space rows of all filled cells in the piece
        filled_rows = []
        for pr in range(piece.shape[0]):
            for pc in range(piece.shape[1]):
                if piece[pr, pc] == 1:
                    br = placement_row + pr
                    if 0 <= br < board_rows:
                        filled_rows.append(br)

        if filled_rows:
            # Convert from top-indexed to bottom-indexed
            centre = (min(filled_rows) + max(filled_rows)) / 2.0
            return board_rows - 1 - centre

    # Fallback: scan board for occupied cells from the top
    for r in range(board_rows):
        if board[r].sum() > 0:
            return float(board_rows - 1 - r)

    return 0.0


# ---------------------------------------------------------------------------
# Feature 2 — Eroded Cells
# ---------------------------------------------------------------------------

def _eroded_cells(
    board_before: np.ndarray,
    piece: np.ndarray,
    placement_row: Optional[int],
    placement_col: Optional[int],
    lines_cleared: int,
) -> float:
    """
    Count how many of the *piece's own cells* were erased during line clearing.

    Eroded Cells = lines_cleared × (piece cells that occupied cleared lines)

    Because this function is called AFTER line clearing has already happened,
    we need board_before (post-placement, pre-clearing) to identify which rows
    were cleared.

    Args:
        board_before  : Board after piece placement but before line clearing.
        piece         : 4×4 piece matrix.
        placement_row : Anchor row of the 4×4 matrix.
        placement_col : Anchor column of the 4×4 matrix.
        lines_cleared : Number of lines that were cleared.

    Returns:
        Eroded cells feature value (float).
    """
    if lines_cleared == 0:
        return 0.0

    board_rows, board_cols = board_before.shape

    # Identify which rows were fully filled (= the cleared ones)
    cleared_rows = set(
        int(r)
        for r in range(board_rows)
        if board_before[r].sum() == board_cols
    )

    if not cleared_rows or placement_row is None or placement_col is None:
        return 0.0

    # Count piece cells that fell on those rows
    piece_cells_in_cleared = 0
    for pr in range(piece.shape[0]):
        for pc in range(piece.shape[1]):
            if piece[pr, pc] == 1:
                br = placement_row + pr
                if br in cleared_rows:
                    piece_cells_in_cleared += 1

    return float(lines_cleared * piece_cells_in_cleared)


# ---------------------------------------------------------------------------
# Feature 3 — Row Transitions
# ---------------------------------------------------------------------------

def _row_transitions(board: np.ndarray) -> float:
    """
    Count the total number of horizontal transitions between filled (1) and
    empty (0) cells, treating the cells immediately outside the left and right
    walls as *filled* (virtual walls).

    A transition is counted wherever adjacent cells differ in occupancy.

    Args:
        board: 20×10 binary numpy array (post-clearing state).

    Returns:
        Row transitions count as float.
    """
    board_rows, board_cols = board.shape

    # Add virtual wall columns on both sides (value = 1)
    left_wall  = np.ones((board_rows, 1), dtype=board.dtype)
    right_wall = np.ones((board_rows, 1), dtype=board.dtype)
    augmented  = np.hstack([left_wall, board, right_wall])

    # Count changes: adjacent columns differ
    transitions = (augmented[:, :-1] != augmented[:, 1:]).sum()
    return float(transitions)


# ---------------------------------------------------------------------------
# Feature 4 — Column Transitions
# ---------------------------------------------------------------------------

def _column_transitions(board: np.ndarray) -> float:
    """
    Count vertical transitions between filled and empty cells, treating the
    virtual floor below the last row as *filled*.

    Args:
        board: 20×10 binary numpy array (post-clearing state).

    Returns:
        Column transitions count as float.
    """
    board_rows, board_cols = board.shape

    # Add virtual floor row (all 1s)
    floor    = np.ones((1, board_cols), dtype=board.dtype)
    augmented = np.vstack([board, floor])

    transitions = (augmented[:-1, :] != augmented[1:, :]).sum()
    return float(transitions)


# ---------------------------------------------------------------------------
# Feature 5 — Holes
# ---------------------------------------------------------------------------

def _holes(board: np.ndarray) -> float:
    """
    Count empty cells (0) that have at least one filled cell (1) directly
    above them in the same column.

    An empty cell is a *hole* if the column above it (from the top of the
    board to the cell's row) contains at least one filled cell.

    Args:
        board: 20×10 binary numpy array.

    Returns:
        Hole count as float.
    """
    board_rows, board_cols = board.shape
    hole_count = 0

    for c in range(board_cols):
        col = board[:, c]
        # Cumulative max from the top: once a 1 has appeared, all 0s below are holes
        filled_above = np.maximum.accumulate(col)
        # Holes: cell is 0 but at least one filled cell is above
        col_holes = ((col == 0) & (filled_above == 1)).sum()
        hole_count += int(col_holes)

    return float(hole_count)


# ---------------------------------------------------------------------------
# Feature 6 — Cumulative Wells
# ---------------------------------------------------------------------------

def _cumulative_wells(board: np.ndarray) -> float:
    """
    Compute the cumulative depth of all "wells" in the board.

    A well is a sequence of consecutive empty cells in a column where both
    neighbouring columns (or the wall) are filled at the same height.
    The contribution of each column c to the well score is:

        sum_{d=1}^{depth(c)} d  =  depth * (depth + 1) / 2

    where depth is the number of consecutive empty cells accessible from
    above that form the well segment.

    Args:
        board: 20×10 binary numpy array.

    Returns:
        Cumulative well score as float.
    """
    board_rows, board_cols = board.shape

    # Extend board with virtual wall columns filled with 1s
    left_wall  = np.ones((board_rows, 1), dtype=board.dtype)
    right_wall = np.ones((board_rows, 1), dtype=board.dtype)
    augmented  = np.hstack([left_wall, board, right_wall])
    # augmented columns: 0=left wall, 1..board_cols=board, board_cols+1=right wall

    total_wells = 0.0

    for c in range(1, board_cols + 1):   # iterate board columns (1-indexed in augmented)
        col       = augmented[:, c]
        left_col  = augmented[:, c - 1]
        right_col = augmented[:, c + 1]

        depth = 0
        for r in range(board_rows):
            if col[r] == 0 and left_col[r] == 1 and right_col[r] == 1:
                # This cell is part of a well
                depth += 1
                total_wells += depth
            else:
                depth = 0    # well segment broken

    return total_wells


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dellacherie_score(
    board_after: np.ndarray,
    piece: np.ndarray,
    lines_cleared: int,
    placement_row: Optional[int] = None,
    placement_col: Optional[int] = None,
    board_before_clear: Optional[np.ndarray] = None,
    debug: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate a board position using the Dellacherie heuristic.

    Formula:
        Score = -LandingHeight + ErodedCells
                - RowTransitions - ColumnTransitions
                - 4 * Holes - Wells

    Higher scores are better.  All penalty features are subtracted.

    Args:
        board_after         : 20×10 board AFTER piece placement AND line
                              clearing (the final resulting state).
        piece               : 4×4 binary matrix of the piece that was placed.
        lines_cleared       : Number of lines cleared during this placement.
        placement_row       : Top-left row of the 4×4 piece on the board
                              (BEFORE clearing).  Used for LandingHeight and
                              ErodedCells.  May be None — features fall back
                              gracefully.
        placement_col       : Top-left column of the 4×4 piece on the board
                              (BEFORE clearing).  Same optional semantics.
        board_before_clear  : Board state AFTER placement but BEFORE clearing
                              lines.  Required for accurate ErodedCells.
                              If None, ErodedCells = lines_cleared × 0.
        debug               : If True, print feature values to stdout.

    Returns:
        (score, features) where:
            score    — float scalar (higher = better).
            features — dict with keys:
                       'landing_height', 'eroded_cells', 'row_transitions',
                       'column_transitions', 'holes', 'wells', 'score'.
    """
    # ── Feature 1: Landing Height ─────────────────────────────────────────
    # Use board_before_clear if available for accurate piece-row detection
    board_for_height = board_before_clear if board_before_clear is not None else board_after
    lh = _landing_height(board_for_height, piece, placement_row, placement_col)

    # ── Feature 2: Eroded Cells ───────────────────────────────────────────
    if board_before_clear is not None:
        ec = _eroded_cells(board_before_clear, piece,
                           placement_row, placement_col, lines_cleared)
    else:
        ec = 0.0

    # ── Feature 3: Row Transitions ────────────────────────────────────────
    rt = _row_transitions(board_after)

    # ── Feature 4: Column Transitions ────────────────────────────────────
    ct = _column_transitions(board_after)

    # ── Feature 5: Holes ──────────────────────────────────────────────────
    ho = _holes(board_after)

    # ── Feature 6: Cumulative Wells ───────────────────────────────────────
    we = _cumulative_wells(board_after)

    # ── Final Score ───────────────────────────────────────────────────────
    score = -lh + ec - rt - ct - 4.0 * ho - we

    features: Dict[str, float] = {
        "landing_height":     lh,
        "eroded_cells":       ec,
        "row_transitions":    rt,
        "column_transitions": ct,
        "holes":              ho,
        "wells":              we,
        "score":              score,
    }

    if debug:
        print(f"\n  [scorer] Dellacherie features:")
        for k, v in features.items():
            print(f"    {k:<22}: {v:+.2f}")

    return score, features


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.simulator import simulate_drop

    print("\n=== scorer self-test ===\n")

    I_PIECE = np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], dtype=np.uint8)

    # --- Test 1: empty board, I-piece at col 0 ---
    empty = np.zeros((20, 10), dtype=np.uint8)

    # Manually compute placement_row for accurate features
    def _drop_with_meta(board, piece, col):
        """Helper: run simulate_drop and recover placement metadata."""
        from simulator import (
            _get_column_offsets, _get_row_offsets,
            _check_collision, _merge_piece
        )
        left_offset, _ = _get_column_offsets(piece)
        top_offset      = _get_row_offsets(piece)
        anchor_col      = col - left_offset
        drop_row        = -top_offset
        while not _check_collision(board, piece, drop_row + 1, anchor_col):
            drop_row += 1
        board_before = _merge_piece(board, piece, drop_row, anchor_col)
        board_after, lc = simulate_drop(board, piece, col)
        return board_after, board_before, lc, drop_row, anchor_col

    b_after, b_before, lc, pr, pc = _drop_with_meta(empty, I_PIECE, 0)
    score, feats = dellacherie_score(
        b_after, I_PIECE, lc,
        placement_row=pr, placement_col=pc,
        board_before_clear=b_before,
        debug=True,
    )
    assert feats["holes"] == 0.0, "No holes on empty-board I-piece drop"
    print(f"  ✓ Test 1 passed  (score={score:.2f})\n")

    # --- Test 2: board with a hole in col 5 — filled cell above empty cell ---
    board2 = np.zeros((20, 10), dtype=np.uint8)
    board2[17, :] = 1          # row 17 filled (above the hole)
    board2[18, :] = 1
    board2[18, 5] = 0          # punch a hole: row 18, col 5 is empty but row 17 col 5 is filled
    _, feats2 = dellacherie_score(board2, I_PIECE, 0, debug=False)
    assert feats2["holes"] >= 1.0, "Should detect at least 1 hole"
    print(f"  ✓ Test 2 passed  (holes={feats2['holes']:.0f})")

    # --- Test 3: well detection ---
    board3 = np.zeros((20, 10), dtype=np.uint8)
    board3[:, 0]  = 1   # left column filled
    board3[:, 2]  = 1   # col 2 filled — creates well in col 1
    _, feats3 = dellacherie_score(board3, I_PIECE, 0, debug=False)
    assert feats3["wells"] > 0, "Should detect wells"
    print(f"  ✓ Test 3 passed  (wells={feats3['wells']:.0f})")

    print("\n  Result: ALL PASS\n")
