# Vision-Tetris Unified Final Report

This is the single combined report for the Phase 4 and Phase 5 deliverables.
It covers the theoretical basis, the final integration work, and the evaluation summary in one document.

## 1. Project Scope

Vision-Tetris is a computer-vision-based Tetris advisor that reads a screenshot, extracts the board state, identifies the active and next pieces, computes the best move, and renders the result back onto the image.

The final pipeline now runs through:

1. Preprocessing
2. Board detection and perspective correction
3. Grid parsing
4. Active piece detection
5. Next piece detection
6. Move search and scoring
7. Final annotation and confidence visualization

Key source files:

- [main.py](main.py)
- [src/preprocess.py](src/preprocess.py)
- [src/board_detector.py](src/board_detector.py)
- [src/grid_parser.py](src/grid_parser.py)
- [src/piece_detector.py](src/piece_detector.py)
- [src/engine.py](src/engine.py)
- [src/simulator.py](src/simulator.py)
- [src/visualizer.py](src/visualizer.py)

## 2. Theoretical Basis

### 2.1 Preprocessing

The preprocessing stage converts the raw screenshot into a cleaner representation for downstream image analysis.

Theoretical rationale:

- Grayscale conversion reduces the channel dimension and removes color variability.
- Gaussian blur suppresses isolated sensor noise and small aliasing artifacts.
- Otsu thresholding selects a binary cutoff by maximizing between-class variance, which is appropriate when the image contains two dominant intensity groups.

This makes the board easier to segment under different game themes and brightness conditions.

### 2.2 Board Localization and Homography

The board detector uses known board corners and a perspective transform to normalize the playfield into a consistent rectangular view.

Theoretical rationale:

- A homography maps one planar quadrilateral to another.
- This corrects perspective skew from the original screenshot.
- Once the board is rectified, each cell can be measured using fixed geometry instead of unstable pixel heuristics.

This step is the backbone of the pipeline because every later module assumes a stable board coordinate system.

### 2.3 Grid Parsing

The grid parser divides the warped board into a 20×10 matrix and classifies each cell as filled or empty using pixel intensity and color thresholds.

Theoretical rationale:

- Cell-wise thresholding is robust once the board has been rectified.
- The parser uses the average fill ratio inside each cell instead of relying on a single pixel.
- This reduces noise sensitivity and handles partially shaded blocks better than a point sample.

The result is a binary board matrix that can be used directly by the simulator and scoring engine.

### 2.4 Piece Detection

The active piece detector groups filled cells and matches them against canonical tetromino patterns.

Theoretical rationale:

- Connected-component analysis groups adjacent occupied cells.
- Shape normalization removes translation dependence.
- Tetromino pattern matching captures shape identity in a way that is invariant to board position.

The assignment notes also mention Hu moments as the theoretical basis for invariant shape classification. In the implemented pipeline, the same invariance goal is achieved more directly through normalized grid-cell pattern matching, which is more stable on the available screenshots.

### 2.5 Move Evaluation

The move engine evaluates placements using a heuristic search with simulation.

Theoretical rationale:

- Each candidate move is simulated to observe the resulting board state.
- The scoring logic rewards line clears and penalizes holes, height, and rough surfaces.
- This mirrors classic Tetris evaluation heuristics such as Dellacherie-style feature scoring.

The score is not just a local board-quality estimate; it is used inside the full pipeline to rank legal moves and select the best recommendation.

### 2.6 Visualization

The final overlay stage makes the recommendation visible to the user.

Theoretical rationale:

- The recommended column is highlighted to show the search outcome.
- The ghost piece illustrates the landing result of the chosen move.
- The heatmap normalizes the move scores per column so the best candidate is visually obvious.

This turns the numeric engine result into a usable visual explanation.

## 3. Final Integration

The final integration point is [main.py](main.py), which now chains all modules in one pass:

1. Load the screenshot
2. Preprocess it
3. Detect and warp the board
4. Parse the 20×10 grid
5. Detect the active piece
6. Detect the next-piece preview
7. Compute the best move
8. Simulate the best drop
9. Render the annotated move overlay
10. Render the confidence heatmap

The Phase 4 visualization work is implemented in [src/visualizer.py](src/visualizer.py).

Outputs now include:

- Annotated screenshot in `output/`
- Heatmap screenshot in `output/heatmap/`
- Console output with detected pieces, chosen move, and timing

## 4. Evaluation Summary

### 4.1 Validation Results

The saved validation results file reports:

- 107 labeled images tested
- 107 correct piece detections
- 100.0% piece classification accuracy

This is a strong indicator that the board parsing and piece detection pipeline is working consistently on the provided training set.

### 4.2 Integration Status

The final code path has been checked after the Phase 4 changes and the touched Python files compile cleanly.

The pipeline now produces both deliverables requested for the output layer:

- Move annotation overlay
- Confidence heatmap overlay

### 4.3 What Was Evaluated

The final integration was evaluated at three levels:

- Static validation of the edited Python files
- Consistency of the full pipeline wiring in `main.py`
- Accuracy evidence from the existing labeled validation set

### 4.4 Current Limitation

A separate cell-level ground-truth benchmark for grid parsing is not present in the current workspace artifacts.
That means the report can confirm integration and piece-detection accuracy, but not claim a fresh numeric grid-parsing score without adding a dedicated board-state annotation set.

## 5. Failure Cases and Robustness Notes

The main failure modes for this kind of pipeline are:

- Low-contrast themes
- UI elements overlapping the playfield
- Small perspective drift in the screenshot
- Pieces partially merging with settled blocks

The current implementation mitigates these issues by:

- Using a warped board view instead of raw screenshot geometry
- Classifying full cell areas rather than single pixels
- Keeping the move evaluation independent of the visual theme
- Rendering overlays after the board has already been localized

## 6. Deliverable Summary

This single report covers the theoretical and final integration requirements in one place.
The code deliverables for Phase 4 are implemented in the repository, and the final visual outputs are now generated by the main pipeline.
