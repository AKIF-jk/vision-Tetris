# =============================================================================
# src/move_generator.py — Phase 3: Move Generation
#
# Generates all unique rotations of a tetromino piece and computes valid
# column placements for each rotation on a 10-column Tetris board.
#
# Public API:
#   get_rotations(shape_matrix, board_cols=10) -> List[Tuple[np.ndarray, List[int]]]
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BOARD_COLS: int = 10   # standard Tetris board width


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trim(matrix: np.ndarray) -> np.ndarray:
    """
    Crop a 4×4 (or larger) rotation matrix so that it contains no all-zero
    border rows or columns.  This gives us a tight bounding box for the piece,
    which makes overlap and column-validity checks simpler.

    Args:
        matrix: 2-D binary numpy array (0 = empty, 1 = filled).

    Returns:
        Tightly cropped 2-D numpy array with the same dtype.
    """
    rows = np.any(matrix, axis=1)
    cols = np.any(matrix, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return matrix[rmin : rmax + 1, cmin : cmax + 1]


def _matrix_hash(matrix: np.ndarray) -> bytes:
    """
    Convert a numpy array to a canonical bytes representation suitable for use
    as a dictionary key / set member.  Trim before hashing so that a 4×4 array
    with the same filled cells in different positions still compares equal.

    Args:
        matrix: 2-D binary numpy array.

    Returns:
        Bytes object uniquely representing the *trimmed* shape.
    """
    trimmed = _trim(matrix)
    return trimmed.tobytes() + bytes(trimmed.shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_rotations(
    shape_matrix: np.ndarray,
    board_cols: int = BOARD_COLS,
    debug: bool = False,
) -> List[Tuple[np.ndarray, List[int]]]:
    """
    Enumerate all *unique* rotations of a tetromino and, for each rotation,
    compute the set of valid left-edge columns where the piece fits entirely
    within the board.

    Duplicate rotations are removed via shape hashing — e.g. the O-piece
    yields 1 rotation, the I-piece yields 2, S/Z/I yield 2, and T/J/L yield 4.

    Args:
        shape_matrix: 4×4 binary numpy array representing the piece in its
                      default orientation (0 = empty, 1 = filled).
        board_cols  : Number of columns on the board (default 10).
        debug       : If True, print rotation details to stdout.

    Returns:
        List of (rotation_matrix, valid_columns) tuples where:
            rotation_matrix — the full 4×4 array for that rotation (uint8).
            valid_columns   — sorted list of integers [0 .. board_cols-1]
                              indicating every column index where the *left
                              edge of the trimmed bounding box* can be placed
                              without the piece exceeding the board boundary.

    Notes:
        * The returned rotation_matrix is always the full 4×4 form (NOT
          trimmed) so that callers receive a consistent shape for simulation.
        * The valid_columns are computed from the trimmed width so placements
          are tight against both walls.

    Examples:
        >>> import numpy as np
        >>> O = np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]], dtype=np.uint8)
        >>> rots = get_rotations(O)
        >>> len(rots)   # O-piece has only one unique rotation
        1
        >>> rots[0][1]  # valid columns for a 2-wide piece on 10-col board
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
    """
    seen_hashes: set = set()
    unique_rotations: List[Tuple[np.ndarray, List[int]]] = []

    base = np.asarray(shape_matrix, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError(f"shape_matrix must be 2-D, got shape {base.shape}")

    for k in range(4):          # 0°, 90°, 180°, 270°
        rotated = np.rot90(base, k=k)
        h = _matrix_hash(rotated)

        if h in seen_hashes:
            if debug:
                print(f"  [move_gen] rotation {k*90:3d}° — DUPLICATE, skipped")
            continue
        seen_hashes.add(h)

        # Trim to compute the true piece width for boundary checks
        trimmed = _trim(rotated)
        piece_width = trimmed.shape[1]

        # Valid left-edge columns: piece must not exceed right boundary
        valid_cols = list(range(board_cols - piece_width + 1))

        if debug:
            print(
                f"  [move_gen] rotation {k*90:3d}° — "
                f"trimmed size {trimmed.shape}, "
                f"valid cols: {valid_cols}"
            )

        # Pad rotated matrix back to 4×4 if np.rot90 changed the shape
        # (it shouldn't for square input, but be defensive)
        if rotated.shape != (4, 4):
            padded = np.zeros((4, 4), dtype=np.uint8)
            padded[: rotated.shape[0], : rotated.shape[1]] = rotated
            rotated = padded

        unique_rotations.append((rotated.copy(), valid_cols))

    if not unique_rotations and debug:
        print("  [move_gen] WARNING: no valid rotations found (empty piece?)")

    return unique_rotations


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SHAPES = {
        "I": np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], dtype=np.uint8),
        "O": np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]], dtype=np.uint8),
        "T": np.array([[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,0]], dtype=np.uint8),
        "S": np.array([[0,0,0,0],[0,1,1,0],[1,1,0,0],[0,0,0,0]], dtype=np.uint8),
        "Z": np.array([[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]], dtype=np.uint8),
        "J": np.array([[0,0,0,0],[1,1,1,0],[0,0,1,0],[0,0,0,0]], dtype=np.uint8),
        "L": np.array([[0,0,0,0],[1,1,1,0],[1,0,0,0],[0,0,0,0]], dtype=np.uint8),
    }
    expected_rotations = {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}

    print("\n=== move_generator self-test ===")
    all_ok = True
    for name, shape in SHAPES.items():
        rots = get_rotations(shape, debug=False)
        expected = expected_rotations[name]
        status = "✓" if len(rots) == expected else "✗"
        if status == "✗":
            all_ok = False
        print(f"  {status} {name}-piece: {len(rots)} unique rotation(s) "
              f"(expected {expected})")
        for rot_mat, vcols in rots:
            print(f"      valid cols: {vcols}")
    print(f"\n  Result: {'ALL PASS' if all_ok else 'FAILURES DETECTED'}\n")
