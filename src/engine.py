# =============================================================================
# src/engine.py — Single-Piece Decision Engine
#
# Evaluates all valid placements of the active piece using a four-feature
# heuristic and returns the best (rotation, column) pair.
#
# Scoring formula (higher = better):
#
#   score = + LINE_CLEAR_WEIGHT  * lines_cleared²
#           - HOLE_WEIGHT        * holes
#           - HEIGHT_WEIGHT      * aggregate_height
#           - BUMPINESS_WEIGHT   * bumpiness
#
# Feature definitions
# -------------------
#   lines_cleared     : number of full rows removed after placement.
#                       Squared to strongly prefer multi-line clears.
#   holes             : empty cells with at least one filled cell above them
#                       in the same column (HIGHEST priority penalty).
#   aggregate_height  : sum of column heights (distance from board bottom to
#                       the topmost filled cell in each column).
#   bumpiness         : sum of absolute height differences between every pair
#                       of adjacent columns.
# =============================================================================

from __future__ import annotations

import time

import numpy as np

from src.move_generator import get_rotations
from src.simulator import simulate_drop, _check_collision, \
                          _get_column_offsets, _find_drop_row


# ---------------------------------------------------------------------------
# Heuristic weights
# ---------------------------------------------------------------------------

HOLE_WEIGHT        = 10.0   # strongest penalty — holes are very hard to fix
HEIGHT_WEIGHT      =  0.51  # penalise tall stacks
BUMPINESS_WEIGHT   =  0.18  # penalise uneven surfaces
LINE_CLEAR_WEIGHT  =  8.0   # reward clearing lines (applied to cleared²)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _column_heights(board: np.ndarray) -> np.ndarray:
    """
    Return the height of each column as a 1-D array.
    Height = number of rows from the bottom up to (and including) the topmost
    filled cell.  An empty column has height 0.

    Args:
        board: (rows x cols) binary numpy array, 0=empty 1=filled.

    Returns:
        heights: 1-D int32 array of length cols.
    """
    board_rows = board.shape[0]
    heights = np.zeros(board.shape[1], dtype=np.int32)
    for c in range(board.shape[1]):
        col = board[:, c]
        filled = np.nonzero(col)[0]
        if len(filled):
            # filled[0] is the topmost occupied row (0 = very top of board)
            heights[c] = board_rows - int(filled[0])
    return heights


def _count_holes(board: np.ndarray) -> int:
    """
    Count empty cells that have at least one filled cell above them in the
    same column.

    Args:
        board: (rows x cols) binary numpy array.

    Returns:
        Total hole count (int).
    """
    holes = 0
    for c in range(board.shape[1]):
        col = board[:, c]
        # Once a 1 has appeared scanning top-to-bottom, all 0s below it are holes
        filled_above = np.maximum.accumulate(col)
        holes += int(((col == 0) & (filled_above == 1)).sum())
    return holes


def _heuristic_score(board_after: np.ndarray, lines_cleared: int) -> float:
    """
    Compute the composite heuristic score for a board state produced after
    placing one piece.

    Priority order (reflected in weights):
        1. Minimise holes          (HOLE_WEIGHT = 10.0)
        2. Reward line clears      (LINE_CLEAR_WEIGHT = 8.0, applied to cleared²)
        3. Minimise pile height    (HEIGHT_WEIGHT = 0.51)
        4. Minimise bumpiness      (BUMPINESS_WEIGHT = 0.18)

    Args:
        board_after   : Board state after placement AND line clearing.
        lines_cleared : Number of lines removed during this placement.

    Returns:
        Scalar score (higher is better).
    """
    heights    = _column_heights(board_after)
    agg_height = int(heights.sum())
    bumpiness  = int(np.abs(np.diff(heights)).sum())
    holes      = _count_holes(board_after)

    score = (
          LINE_CLEAR_WEIGHT  * (lines_cleared ** 2)
        - HOLE_WEIGHT        * holes
        - HEIGHT_WEIGHT      * agg_height
        - BUMPINESS_WEIGHT   * bumpiness
    )
    return score


# ---------------------------------------------------------------------------
# Internal helper — simulate a drop and return the resulting board
# ---------------------------------------------------------------------------

def _try_placement(board, piece, col):
    """
    Drop *piece* at *col* on *board*.

    Returns:
        (board_after, lines_cleared) or (None, 0) if the move is invalid.
    """
    left_offset, _ = _get_column_offsets(piece)
    anchor_col     = col - left_offset
    drop_row       = _find_drop_row(board, piece, anchor_col)

    if _check_collision(board, piece, drop_row, anchor_col):
        return None, 0

    board_after, lines_cleared = simulate_drop(board, piece, col)
    return board_after, lines_cleared


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def best_move(board, active_piece, debug=False):
    """
    Find the best (rotation, column) for *active_piece* on *board*.

    Iterates over every unique rotation of the piece and every valid column
    for that rotation.  Each resulting board state is scored with
    _heuristic_score; the placement with the highest score is returned.

    Args:
        board        : 20x10 binary numpy array (0=empty, 1=filled).
        active_piece : 4x4 binary numpy array (current tetromino).
        debug        : Print per-move scores when True.

    Returns:
        (best_rotation, best_col, scores_matrix)
            best_rotation : 4x4 numpy array of the chosen rotation.
            best_col      : int column for the left edge of the trimmed piece.
            scores_matrix : (num_rotations x 10) float array; -inf = invalid.
    """
    t_start = time.time()

    rotations = get_rotations(active_piece)

    if not rotations:
        return active_piece.copy(), 0, np.full((1, 10), -np.inf)

    num_rots      = len(rotations)
    scores_matrix = np.full((num_rots, 10), -np.inf)

    best_score    = -1e18
    best_rotation = rotations[0][0]
    best_col      = 0

    for rot_idx, (rot_matrix, valid_cols) in enumerate(rotations):
        for col in valid_cols:

            board_after, lines_cleared = _try_placement(board, rot_matrix, col)

            if board_after is None:
                continue

            score = _heuristic_score(board_after, lines_cleared)
            scores_matrix[rot_idx, col] = score

            if debug:
                heights   = _column_heights(board_after)
                holes     = _count_holes(board_after)
                bumpiness = int(np.abs(np.diff(heights)).sum())
                print(
                    f"  rot={rot_idx} col={col:2d} | "
                    f"score={score:+7.2f}  "
                    f"lines={lines_cleared}  "
                    f"holes={holes}  "
                    f"height={int(heights.sum())}  "
                    f"bump={bumpiness}"
                )

            if score > best_score:
                best_score    = score
                best_rotation = rot_matrix.copy()
                best_col      = col

    elapsed = (time.time() - t_start) * 1000

    if debug:
        print(f"\n  Best -> col={best_col} | score={best_score:+.2f} | {elapsed:.1f} ms\n")

    if best_score == -1e18:
        return rotations[0][0].copy(), 0, scores_matrix

    return best_rotation, best_col, scores_matrix


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def print_scores_matrix(scores_matrix):
    """Pretty-print the (rotations x columns) score grid."""
    num_rots, num_cols = scores_matrix.shape
    header = "         " + "".join(f"  col{c:2d}" for c in range(num_cols))
    print(header)
    print("         " + "-" * (num_cols * 7))

    for r in range(num_rots):
        row_str = f"  rot {r}  |"
        for c in range(num_cols):
            val = scores_matrix[r, c]
            if np.isinf(val):
                row_str += "   -inf"
            else:
                row_str += f" {val:+6.1f}"
        print(row_str)
    print()