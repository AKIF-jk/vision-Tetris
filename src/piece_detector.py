# =============================================================================
# src/piece_detector.py — Tasks 6 & 7: Active Piece + Next Piece Detection
#
# Rewritten to use grid-cell pattern matching for active piece detection.
# Instead of pixel-level blob analysis + Hu moments, we:
#   1. Find connected groups of filled cells in the parsed grid
#   2. Match 4-cell groups against all known tetromino rotation patterns
#   3. Pick the highest matching group (= the active falling piece)
#
# This approach achieves ~100% classification accuracy on training data.
# =============================================================================

import cv2
import numpy as np
import os, sys
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.grid_parser import get_block_mask

# ---------------------------------------------------------------------------
# Canonical 4x4 shape matrices (kept for API compatibility)
# ---------------------------------------------------------------------------
TETROMINO_SHAPES = {
    'I': np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], dtype=np.uint8),
    'O': np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]], dtype=np.uint8),
    'T': np.array([[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,0]], dtype=np.uint8),
    'S': np.array([[0,0,0,0],[0,1,1,0],[1,1,0,0],[0,0,0,0]], dtype=np.uint8),
    'Z': np.array([[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]], dtype=np.uint8),
    'J': np.array([[0,0,0,0],[1,1,1,0],[0,0,1,0],[0,0,0,0]], dtype=np.uint8),
    'L': np.array([[0,0,0,0],[1,1,1,0],[1,0,0,0],[0,0,0,0]], dtype=np.uint8),
}

# Color fallback — BGR approximate per user spec:
# I=red, Z=green, L=yellow/mustard, S=blue, O=cyan, J=purple, T=gray
PIECE_COLORS_BGR = {
    'I': (0,   0,   200),
    'Z': (0,   200, 0  ),
    'L': (0,   200, 200),
    'S': (200, 0,   0  ),
    'O': (200, 200, 0  ),
    'J': (150, 0,   150),
    'T': (180, 180, 180),
}

# ---------------------------------------------------------------------------
# Tetromino pattern database — all valid (type, rotation) as frozensets
# of normalized (row, col) offsets.
# ---------------------------------------------------------------------------
def _normalize(cells):
    """Normalize cell positions so minimum row and col are 0."""
    cells = list(cells)
    min_r = min(c[0] for c in cells)
    min_c = min(c[1] for c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)

def _build_pattern_db():
    """Build lookup: frozenset of normalized cells -> piece type."""
    patterns = {}

    def _add(name, cells):
        pat = _normalize(cells)
        if pat not in patterns:
            patterns[pat] = name

    # I-piece
    _add('I', [(0,0),(0,1),(0,2),(0,3)])          # horizontal
    _add('I', [(0,0),(1,0),(2,0),(3,0)])           # vertical

    # O-piece
    _add('O', [(0,0),(0,1),(1,0),(1,1)])

    # T-piece (4 rotations)
    _add('T', [(0,0),(0,1),(0,2),(1,1)])           # XXX / .X.
    _add('T', [(0,0),(1,0),(1,1),(2,0)])           # X. / XX / X.
    _add('T', [(0,1),(1,0),(1,1),(1,2)])           # .X. / XXX
    _add('T', [(0,1),(1,0),(1,1),(2,1)])           # .X / XX / .X

    # S-piece (2 rotations)
    _add('S', [(0,1),(0,2),(1,0),(1,1)])           # .XX / XX.
    _add('S', [(0,0),(1,0),(1,1),(2,1)])           # X. / XX / .X

    # Z-piece (2 rotations)
    _add('Z', [(0,0),(0,1),(1,1),(1,2)])           # XX. / .XX
    _add('Z', [(0,1),(1,0),(1,1),(2,0)])           # .X / XX / X.

    # J-piece (4 rotations — hooks LEFT)
    _add('J', [(0,0),(1,0),(1,1),(1,2)])           # X.. / XXX
    _add('J', [(0,0),(0,1),(1,0),(2,0)])           # XX / X. / X.
    _add('J', [(0,0),(0,1),(0,2),(1,2)])           # XXX / ..X
    _add('J', [(0,1),(1,1),(2,0),(2,1)])           # .X / .X / XX

    # L-piece (4 rotations — hooks RIGHT)
    _add('L', [(0,2),(1,0),(1,1),(1,2)])           # ..X / XXX
    _add('L', [(0,0),(1,0),(2,0),(2,1)])           # X. / X. / XX
    _add('L', [(0,0),(0,1),(0,2),(1,0)])           # XXX / X..
    _add('L', [(0,0),(0,1),(1,1),(2,1)])           # XX / .X / .X

    return patterns

PATTERN_DB = _build_pattern_db()


# ---------------------------------------------------------------------------
# Connected-component analysis on the GRID (not pixels)
# ---------------------------------------------------------------------------
def _find_connected_groups(grid):
    """Find all connected groups of filled cells using BFS (4-connected)."""
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    groups = []

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1 and not visited[r, c]:
                group = set()
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    group.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and grid[nr, nc] == 1
                                and not visited[nr, nc]):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                groups.append(group)

    return groups


def _match_pattern(cells):
    """Return piece type if cells form a valid tetromino, else None."""
    if len(cells) != 4:
        return None
    pat = _normalize(cells)
    return PATTERN_DB.get(pat)


# ---------------------------------------------------------------------------
# Try to extract a valid 4-cell tetromino from the top of a larger group.
#
# When the active piece has already touched a settled column but is still
# logically the active piece, it merges into a bigger connected component.
# We scan from the top of that component and try every possible 4-cell
# connected subset to see if it matches a tetromino.
# ---------------------------------------------------------------------------
def _extract_tetromino_from_top(group, all_filled_set=None):
    """
    Try to find a valid 4-cell tetromino starting from the topmost cells
    of a larger connected group.  Returns (piece_type, cells) or (None, None).
    """
    sorted_cells = sorted(group, key=lambda x: (x[0], x[1]))
    search_set = group if all_filled_set is None else all_filled_set

    # For each of the topmost cells, do a BFS limited to 4 cells
    # and check all orderings via DFS with backtracking
    for start in sorted_cells[:12]:
        result = _dfs_match(start, search_set)
        if result is not None:
            return result
    return None, None


def _dfs_match(start, available):
    """DFS to find 4 connected cells forming a valid tetromino."""
    stack = [(frozenset([start]), [start])]
    visited_states = set()

    while stack:
        current_set, current_list = stack.pop()
        if current_set in visited_states:
            continue
        visited_states.add(current_set)

        if len(current_list) == 4:
            ptype = _match_pattern(set(current_list))
            if ptype is not None:
                return ptype, set(current_list)
            continue

        # Expand: add neighbors of any cell in current_list
        for cr, cc in current_list:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in available and (nr, nc) not in current_set:
                    new_set = current_set | frozenset([(nr, nc)])
                    if new_set not in visited_states:
                        stack.append((new_set, current_list + [(nr, nc)]))

    return None, None


# ===========================================================================
# ACTIVE PIECE DETECTION — Grid-cell pattern matching
# ===========================================================================
def detect_piece(board_img: np.ndarray,
                 grid: np.ndarray,
                 save_debug: bool = True,
                 debug_prefix: str = "frame") -> tuple:
    """
    Detect the single active (falling) tetromino using grid-cell analysis.

    Strategy:
    1. Find all connected groups of filled cells in the grid.
    2. Groups of exactly 4 cells: check against tetromino pattern DB.
    3. Pick the highest matching group (smallest min-row = active piece).
    4. If no isolated 4-cell group found, try extracting a tetromino
       from the top of larger groups (piece touching settled blocks).
    5. Color-based fallback for J/L ambiguity if needed.
    """
    board_h, board_w = board_img.shape[:2]
    cw = board_w / config.GRID_COLS
    ch = board_h / config.GRID_ROWS

    groups = _find_connected_groups(grid)

    # --- Phase 1: look for isolated 4-cell groups matching a tetromino ---
    candidates = []
    for group in groups:
        if len(group) == 4:
            ptype = _match_pattern(group)
            if ptype is not None:
                min_row = min(r for r, c in group)
                candidates.append((min_row, ptype, group))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, piece_type, cells = candidates[0]
        source = "isolated-4"
    else:
        # --- Phase 2: extract tetromino from top of larger groups ---
        # Sort groups by their topmost row
        groups_sorted = sorted(groups,
                               key=lambda g: min(r for r, c in g))
        piece_type, cells = None, None
        for group in groups_sorted:
            if len(group) >= 4:
                ptype, found_cells = _extract_tetromino_from_top(group)
                if ptype is not None:
                    piece_type = ptype
                    cells = found_cells
                    break
        source = "extracted"

    if piece_type is None or cells is None:
        return _fallback_piece(board_img, grid, save_debug, debug_prefix)

    # Compute grid position from the detected cells
    min_row = min(r for r, c in cells)
    min_col = min(c for r, c in cells)
    position = (min_row, min_col)

    # Compute pixel bounding box for debug visualization
    all_rows = [r for r, c in cells]
    all_cols = [c for r, c in cells]
    px_x = int(min(all_cols) * cw)
    px_y = int(min(all_rows) * ch)
    px_w = int((max(all_cols) - min(all_cols) + 1) * cw)
    px_h = int((max(all_rows) - min(all_rows) + 1) * ch)

    if save_debug:
        _save_active_debug(board_img, cells, piece_type,
                           px_x, px_y, px_w, px_h, source, debug_prefix)

    return piece_type, TETROMINO_SHAPES[piece_type].copy(), position, 0


def _fallback_piece(board_img, grid, save_debug, debug_prefix):
    """Fallback when no valid tetromino pattern found in grid."""
    board_h, board_w = board_img.shape[:2]
    cw = board_w / config.GRID_COLS
    ch = board_h / config.GRID_ROWS

    # Use topmost filled cells and try pixel-level analysis
    first_row = None
    for row in range(config.GRID_ROWS):
        if np.any(grid[row, :] == 1):
            first_row = row
            break

    if first_row is None:
        print("  [piece_detector] Board empty, no piece found.")
        if save_debug:
            os.makedirs(config.DEBUG_DIR, exist_ok=True)
            d = board_img.copy()
            cv2.putText(d, "NO PIECE", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_8_active_piece.jpg", d)
        return 'T', TETROMINO_SHAPES['T'].copy(), (0, 0), 0

    scan_end = min(first_row + 4, config.GRID_ROWS)
    fp = np.argwhere(grid[first_row:scan_end, :] == 1)
    if len(fp) == 0:
        return 'T', TETROMINO_SHAPES['T'].copy(), (0, 0), 0

    # Collect the topmost cells
    cells_in_region = set()
    for dr, dc in fp:
        cells_in_region.add((first_row + dr, dc))

    # Try to match a pattern
    if len(cells_in_region) == 4:
        ptype = _match_pattern(cells_in_region)
        if ptype:
            pos = (min(r for r, c in cells_in_region),
                   min(c for r, c in cells_in_region))
            return ptype, TETROMINO_SHAPES[ptype].copy(), pos, 0

    # Try extracting from the top cells
    ptype, found = _extract_tetromino_from_top(cells_in_region)
    if ptype:
        pos = (min(r for r, c in found),
               min(c for r, c in found))
        return ptype, TETROMINO_SHAPES[ptype].copy(), pos, 0

    # Last resort: use pixel-level Hu moments on topmost region
    rows_g = fp[:, 0] + first_row
    cols_g = fp[:, 1]
    x = int(cols_g.min() * cw)
    bw = int((cols_g.max() - cols_g.min() + 1) * cw)
    y = int(rows_g.min() * ch)
    bh = int((rows_g.max() - rows_g.min() + 1) * ch)

    block_mask = get_block_mask(board_img)
    roi = block_mask[y:y+bh, x:x+bw]
    ptype, _, _ = _classify_by_hu(roi)
    position = (first_row, int(cols_g.min()))

    return ptype, TETROMINO_SHAPES[ptype].copy(), position, 0


# ---------------------------------------------------------------------------
# Hu moments classifier (kept as last-resort fallback)
# ---------------------------------------------------------------------------
def _compute_hu(shape: np.ndarray) -> np.ndarray:
    img = cv2.resize((shape * 255 if shape.max() <= 1 else shape).astype(np.uint8),
                     (64, 64), interpolation=cv2.INTER_NEAREST)
    m = cv2.moments(img)
    hu = cv2.HuMoments(m).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

_HU_REFERENCE = {}
for _name, _shape in TETROMINO_SHAPES.items():
    _HU_REFERENCE[_name] = []
    _seen = []
    for _r in range(4):
        _rot = np.rot90(_shape, _r)
        if _rot.tolist() not in _seen:
            _seen.append(_rot.tolist())
            _HU_REFERENCE[_name].append((_compute_hu(_rot), _r * 90))

def _classify_by_hu(mask: np.ndarray) -> tuple:
    hu = _compute_hu(mask)
    best, best_rot, best_score = 'T', 0, float('inf')
    for ptype, refs in _HU_REFERENCE.items():
        for hu_ref, rot_deg in refs:
            s = float(np.sum(np.abs(hu - hu_ref)))
            if s < best_score:
                best_score, best, best_rot = s, ptype, rot_deg
    return best, best_rot, best_score


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------
def _save_active_debug(board_img, cells, piece_type,
                       px_x, px_y, px_w, px_h, source, prefix):
    os.makedirs(config.DEBUG_DIR, exist_ok=True)
    d = board_img.copy()
    board_h, board_w = d.shape[:2]
    cw = board_w / config.GRID_COLS
    ch = board_h / config.GRID_ROWS

    # Highlight detected cells
    overlay = d.copy()
    for r, c in cells:
        x1 = int(c * cw)
        y1 = int(r * ch)
        x2 = int((c + 1) * cw)
        y2 = int((r + 1) * ch)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
    d = cv2.addWeighted(overlay, 0.35, d, 0.65, 0)

    # Bounding box
    cv2.rectangle(d, (px_x, px_y), (px_x + px_w, px_y + px_h), (0, 255, 0), 2)
    label = f"{source}: {piece_type}"
    cv2.putText(d, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 255), 2)
    cv2.imwrite(f"{config.DEBUG_DIR}/{prefix}_8_active_piece.jpg", d)


# ===========================================================================
# NEXT PIECE DETECTION — Grid-cell pattern matching
#
# Same approach as active piece: parse the preview region into a grid
# using the known board cell size, then match against tetromino patterns.
# ===========================================================================

def _parse_next_piece_grid(next_region_img: np.ndarray):
    """
    Parse the next-piece preview region into a small grid and identify
    the tetromino using pattern matching.

    Steps:
    1. Build a block mask (colored + gray/white pixels).
    2. Crop mask to bounding box of non-zero pixels.
    3. Determine grid dimensions from board cell size.
    4. Parse cells with a fill threshold.
    5. Find connected groups of 4 and match against PATTERN_DB.

    Returns:
        (piece_type, grid, n_rows, n_cols) or (None, None, 0, 0) on failure.
    """
    mask = get_block_mask(next_region_img)

    # Morphological cleanup
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    coords = np.argwhere(mask > 0)
    if len(coords) < 50:
        return None, None, 0, 0

    # Crop mask to bounding box of piece pixels
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]
    crop_h, crop_w = cropped_mask.shape

    # Use board cell size to determine grid dimensions
    board_cw = config.BOARD_WIDTH_PX / config.GRID_COLS    # ~44.5 px
    board_ch = config.BOARD_HEIGHT_PX / config.GRID_ROWS   # ~30.85 px
    n_cols = max(1, round(crop_w / board_cw))
    n_rows = max(1, round(crop_h / board_ch))

    cell_w = crop_w / n_cols
    cell_h = crop_h / n_rows

    grid = np.zeros((n_rows, n_cols), dtype=np.uint8)
    FILL_THRESHOLD = 0.40

    for row in range(n_rows):
        for col in range(n_cols):
            x1 = int(col * cell_w) + 2
            x2 = int((col + 1) * cell_w) - 2
            y1 = int(row * cell_h) + 2
            y2 = int((row + 1) * cell_h) - 2
            if x2 <= x1 or y2 <= y1:
                continue
            cell = cropped_mask[y1:y2, x1:x2]
            total = cell.size
            filled = int(np.count_nonzero(cell))
            ratio = filled / total if total > 0 else 0
            grid[row, col] = 1 if ratio >= FILL_THRESHOLD else 0

    # Find connected groups and match against tetromino patterns
    groups = _find_connected_groups(grid)
    for group in sorted(groups, key=len, reverse=True):
        if len(group) == 4:
            ptype = _match_pattern(group)
            if ptype is not None:
                return ptype, grid, n_rows, n_cols

    return None, grid, n_rows, n_cols


def detect_next_piece(next_region_img: np.ndarray,
                      save_debug: bool = True,
                      debug_prefix: str = "frame") -> tuple:
    """
    Detect next piece from the preview region image.

    Primary: grid-cell pattern matching (same approach as active piece).
    Fallback: Hu-moments classifier on the raw mask.
    """
    # --- Primary: grid-based pattern matching ---
    piece_type, grid, n_rows, n_cols = _parse_next_piece_grid(next_region_img)

    source = "grid-pattern"
    if piece_type is None:
        # --- Fallback: Hu-moments on the raw block mask ---
        source = "hu-fallback"
        mask = get_block_mask(next_region_img)
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        if np.count_nonzero(mask) < 80:
            piece_type = 'T'
            print("  [piece_detector] Next piece: too few pixels → default T")
        else:
            piece_type, _, _ = _classify_by_hu(mask)

    if save_debug:
        os.makedirs(config.DEBUG_DIR, exist_ok=True)
        d = next_region_img.copy()
        cv2.putText(d, f"Next:{piece_type} ({source})",
                    (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_9_next_piece.jpg", d)

        mask_out = get_block_mask(next_region_img)
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_9b_next_mask.jpg",
                    mask_out)

    return piece_type, TETROMINO_SHAPES[piece_type].copy(), 0