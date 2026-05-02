# Phase 3 Engine Integration — Quick Reference

## What Was Integrated?

Four new Python modules implementing Phase 3 of the Tetris Advisor system:

| Module | Purpose | Key Function |
|--------|---------|---|
| `src/move_generator.py` | Generate tetromino rotations | `get_rotations(shape)` → list of (rotation, valid_cols) |
| `src/simulator.py` | Drop pieces and update board | `simulate_drop(board, piece, col)` → (new_board, lines_cleared) |
| `src/scorer.py` | Evaluate board positions | `dellacherie_score(board, piece, lines)` → (score, features) |
| `src/engine.py` | Two-level lookahead search | `best_move(board, active, next)` → (rotation, col, scores) |

## How It Works

```
Current board + Active piece + Next piece
           ↓
    ┌─────────────┐
    │ Engine      │
    │ Level 1:    │ All rotations × columns of active piece
    │ Level 2:    │ For each L1, all rotations × columns of next piece
    │ Evaluate:   │ Dellacherie score on final board state
    │ Select:     │ L1 move that maximizes best L2 score
    └─────────────┘
           ↓
    Best column + Optimal rotation + Scores matrix
```

## Using the Integration

### Basic Usage (main.py)

```python
from main import run_pipeline

result = run_pipeline("screenshot.jpg")

# result now includes:
best_col = result["best_col"]                # int (0-9)
best_rotation = result["best_rotation"]      # 4×4 numpy array
scores_matrix = result["scores_matrix"]      # (R × 10) numpy array
```

### Command Line

```bash
# Normal mode
python main.py --input screenshot.jpg

# Debug mode (shows scores matrix)
python main.py --input screenshot.jpg --debug

# Verbose mode (shows timing for each step)
python main.py --input screenshot.jpg --verbose
```

### Direct Engine Access

```python
import numpy as np
from src.engine import best_move, print_scores_matrix

# Create or load board (20×10 numpy array)
board = np.zeros((20, 10), dtype=np.uint8)

# Create piece shapes (4×4 numpy arrays)
active_piece = np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], dtype=np.uint8)
next_piece = np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]], dtype=np.uint8)

# Get best move
rotation, col, scores = best_move(board, active_piece, next_piece, debug=True)

# Print the scores matrix
print_scores_matrix(scores)
```

## Output Explanation

### Best Column (0-9)
The recommended column to place the piece, where 0 is the leftmost column.

### Best Rotation
The 4×4 matrix representing the piece in its optimal rotation.

### Scores Matrix
A (R × 10) array where:
- **Rows** = unique rotations (1-4 depending on piece)
- **Columns** = board columns (0-9)
- **Values** = Dellacherie scores
- **-inf** = invalid placement (piece goes out of bounds)

Higher scores are better. The algorithm selects the position with the highest score.

## Dellacherie Scoring Features

The engine evaluates board positions using 6 features:

1. **Landing Height** (-): Lower is better (penalties for high drops)
2. **Eroded Cells** (+): More is better (reward for clearing lines)
3. **Row Transitions** (-): Fewer is better (penalties for scattered pieces)
4. **Column Transitions** (-): Fewer is better (penalties for vertical gaps)
5. **Holes** (-): Fewer is better (penalties for unfillable gaps)
6. **Wells** (-): Fewer is better (penalties for deep 1-width gaps)

```
Score = -Landing_Height + Eroded_Cells - Row_Transitions 
        - Column_Transitions - 4×Holes - Wells
```

## Performance

| Operation | Time |
|-----------|------|
| Move generation | ~5 ms |
| Board simulation (× 40-160) | ~50-200 ms |
| Scoring (× 40-160) | ~50-200 ms |
| **Total engine** | **~100-350 ms** |
| **Full pipeline** | **~1400-1500 ms** |

## Deduplication Logic

The engine automatically handles piece rotation deduplication:

- **O-piece**: 1 unique rotation
- **I-piece**: 2 unique rotations
- **S/Z-piece**: 2 unique rotations each
- **T/J/L-piece**: 4 unique rotations each

This ensures efficient search without redundant evaluations.

## Error Handling

The engine handles edge cases gracefully:

```python
# Empty/invalid board → returns fallback (first rotation, col 0)
rotation, col, scores = best_move(invalid_board, piece, next)

# Full board (no valid moves) → returns all -inf scores
# Spawn collision → detected and handled

# All returned matrices are numpy arrays (uint8 for shapes, float for scores)
```

## Integration with Phase 4 (Visualization)

The scores matrix is designed to work with Phase 4 heatmap visualization:

```python
# Phase 4 will use this matrix to show:
# - Color-coded board overlay
# - Column recommendations
# - Rotation alternatives
# - Score per position

# Simply pass the scores_matrix from run_pipeline() to Phase 4
visualize_heatmap(result["scores_matrix"], result["best_col"])
```

## Testing

Run the validation suite with the engine enabled:

```bash
python test_validation.py --train-dir data/train --annotations data/train/_annotations.csv
```

This processes all training images through the full pipeline (including the new engine).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src.engine'` | Ensure you're running from project root and imports use `src.` prefix |
| `ValueError: Expected 20×10 board` | Board must be exactly (20, 10). Check grid parsing output |
| Engine returns -inf for all positions | Board is likely full. Check grid state or reset. |
| Slow execution | Normal for first run. Subsequent runs may be faster due to Python JIT compilation |

## Configuration

No configuration needed. The engine uses hardcoded Dellacherie weights per the specification.

To experiment with different weights, modify `scorer.py`:

```python
# Current weights in dellacherie_score():
score = -lh + ec - rt - ct - 4.0 * ho - we
#        ↑   ↑   ↑   ↑    ↑     ↑
#        1   1   1   1    4     1

# Try adjusting the coefficients above
```

## API Summary

### Main Entry Points

```python
# main.py
run_pipeline(input_path, debug=False, verbose=False) → dict

# engine.py
best_move(board, active_piece, next_piece, debug=False) 
    → (best_rotation, best_col, scores_matrix)

print_scores_matrix(scores_matrix) → None (prints to stdout)

# scorer.py
dellacherie_score(board, piece, lines_cleared, ...) 
    → (score, features_dict)

# simulator.py
simulate_drop(board, piece, col, debug=False) 
    → (new_board, lines_cleared)

# move_generator.py
get_rotations(shape, board_cols=10, debug=False) 
    → list of (rotation_matrix, valid_columns)
```

## Next Steps

1. **Phase 4**: Implement heatmap visualization using `scores_matrix`
2. **Testing**: Validate move recommendations against human expert gameplay
3. **Optimization**: Profile and optimize hot paths if needed
4. **Integration**: Connect to UI for real-time recommendations
