# Vision-Based Tetris Advisor

A computer vision system that analyzes Tetris gameplay screenshots, extracts the board state, identifies pieces, and computes the optimal move using a Dellacherie-scored two-piece lookahead AI engine. Built as a Digital Image Processing (DIP) course project.

![Pipeline](doc/pipeline.png)
*Input screenshot → annotated output with ghost piece and confidence heatmap*

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Image Processing Techniques](#image-processing-techniques)
- [AI Engine](#ai-engine)
- [Output](#output)
- [Project Structure](#project-structure)
- [Results](#results)
- [Configuration](#configuration)

---

## Overview

The system takes a Tetris gameplay screenshot and performs seven stages of processing:

1. **Preprocess** — Grayscale conversion, Gaussian blur, Otsu binarization
2. **Board Detection** — Perspective warp to extract the playfield
3. **Grid Parsing** — HSV color masking, cell fill-ratio analysis, 20×10 grid extraction
4. **Piece Detection** — Connected-component matching + Hu moments fallback
5. **AI Engine** — Move generation, board simulation, Dellacherie scoring, two-piece lookahead
6. **Visualization** — Ghost piece overlay, column highlight, confidence heatmap

---

## Pipeline

```
Input Screenshot
      │
      ▼
┌─────────────────┐
│   Preprocess    │  cv2.cvtColor (BGR→Gray), cv2.GaussianBlur,
│                 │  cv2.threshold (Otsu)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Board Detection │  cv2.getPerspectiveTransform,
│                 │  cv2.warpPerspective
└────────┬────────┘
         ▼
┌─────────────────┐
│   Grid Parsing  │  cv2.cvtColor (BGR→HSV), cv2.inRange,
│                 │  cv2.bitwise_or, cell fill-ratio ≥ 50%
└────────┬────────┘
         ▼
┌─────────────────┐
│ Piece Detection │  Connected-component (BFS) + pattern matching
│                 │  → Hu moments fallback (cv2.moments, cv2.HuMoments)
│                 │  → Morphological cleanup (cv2.morphologyEx)
└────────┬────────┘
         ▼
┌─────────────────┐
│   AI Engine     │  np.rot90 rotations, gravity simulation,
│                 │  Dellacherie 6-feature scoring, lookahead
└────────┬────────┘
         ▼
┌─────────────────┐
│  Visualization  │  cv2.fillConvexPoly, cv2.addWeighted,
│                 │  cv2.putText, Red-Yellow-Green heatmap
└────────┬────────┘
         ▼
Annotated Output + Heatmap
```

---

## Installation

### Prerequisites

- Python 3.9+
- OpenCV 4.x
- NumPy

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/vision-Tetris.git
cd vision-Tetris

# Create virtual environment
python3 -m venv .venv

# Install dependencies
.venv/bin/pip install -r requirements.txt
```

### Dependencies

```
opencv-python==4.9.0.80
numpy==1.26.4
matplotlib==3.8.4
Pillow==10.3.0
```

---

## Usage

### Process a single screenshot

```bash
.venv/bin/python main.py --input data/train/sample.jpg
```

### Options

| Flag | Description |
|---|---|
| `--input PATH` | Path to screenshot (required) |
| `--debug` | Save intermediate DIP debug images to `output/debug/` |
| `--verbose` | Print per-step timing and module logs |

```bash
# Full example
.venv/bin/python main.py --input data/train/sample.jpg --debug --verbose
```

### Batch process all images

```bash
.venv/bin/python batch_process.py
```

Processes every image in `data/`, saves stitched composites (original + annotated + heatmap) to `output/results/`.

### Test a single image

```bash
.venv/bin/python test_single.py data/train/sample.jpg
```

### Run validation against ground truth

```bash
.venv/bin/python test_validation.py --verbose --save-results
```

Tests all images in `data/train/` against `_annotations.csv`, reports per-class accuracy and confusion matrix.

---

## Image Processing Techniques

### 1. Grayscale Conversion (`cv2.cvtColor`)

Reduces 3-channel BGR to single-channel intensity. Eliminates color variability, simplifies thresholding, and reduces compute.

### 2. Gaussian Blur (`cv2.GaussianBlur`)

5×5 kernel convolution that suppresses high-frequency sensor noise and aliasing artifacts before binarization, preventing spurious pixels.

### 3. Otsu Thresholding (`cv2.threshold` + `THRESH_OTSU`)

Automatically determines the optimal threshold by maximizing between-class variance of pixel intensities. Handles varied lighting conditions without manual tuning.

### 4. HSV Color Space & Color Thresholding (`cv2.cvtColor` BGR→HSV + `cv2.inRange`)

HSV separates hue from saturation/value, making color-based segmentation robust to brightness variations. Two complementary masks detect:
- **Colored blocks** (high saturation, moderate-high value)
- **Gray/white blocks** (low saturation, high value — for ghost pieces)

### 5. Bitwise Operations (`cv2.bitwise_or`)

Combines the colored and gray-block masks into a single unified block mask.

### 6. Perspective Transform (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`)

Computes a homography from known board corners to a rectangle, then warps the board region to correct perspective skew. Also applied to the next-piece preview region.

### 7. Morphological Operations (`cv2.morphologyEx`)

- **Closing** (dilation → erosion): Fills small holes inside next-piece blocks
- **Opening** (erosion → dilation): Removes spurious noise pixels from the mask

### 8. Cell Fill-Ratio Analysis

Rather than classifying cells by a single pixel, examines all pixels in each cell interior. A cell is marked filled if ≥50% of its pixels are block pixels — making classification robust to partial shading, thin grid lines, and anti-aliasing.

### 9. Connected-Component Analysis (custom BFS)

Labels groups of adjacent filled cells using 4-directional traversal on the 20×10 grid. Groups of exactly 4 cells are matched against a tetromino pattern database. For merged groups, DFS extracts valid 4-cell subsets from the top.

### 10. Hu Moments (`cv2.moments` → `cv2.HuMoments`)

Seven translation-, scale-, and rotation-invariant shape descriptors. Serves as a fallback classifier when grid-pattern matching fails — compares against pre-computed reference values using sum-of-absolute-differences.

### 11. Weighted Image Blending (`cv2.addWeighted`)

Blends annotation overlays (ghost piece, column highlight, heatmap) with the original screenshot at configurable opacity (α = 0.4).

### 12. Drawing Primitives

`cv2.rectangle`, `cv2.putText`, `cv2.fillConvexPoly`, `cv2.polylines`, `cv2.line` for rendering bounding boxes, text annotations, ghost pieces, column highlights, grid lines, and heatmap bars.

---

## AI Engine

### Move Generation (`src/move_generator.py`)

Generates all unique rotations of the current tetromino via `np.rot90`, deduplicates identical shapes, and computes valid drop columns for each rotation.

### Board Simulation (`src/simulator.py`)

Simulates piece placement: gravity drop with collision detection, piece merging, and line clearing. Returns the resulting board state.

### Dellacherie Scoring (`src/scorer.py`)

Scores each resulting board state using six heuristic features:

| Feature | Description |
|---|---|
| **Landing Height** | Height of the placed piece's highest point (lower is better) |
| **Eroded Cells** | Number of lines cleared × number of cells contributed (reward line clears) |
| **Row Transitions** | Number of filled-to-empty or empty-to-filled transitions across all rows |
| **Column Transitions** | Same as row transitions but along columns |
| **Holes** | Empty cells with at least one filled cell above them |
| **Cumulative Wells** | Deep vertical gaps weighted by depth |

### Two-Piece Lookahead (`src/engine.py`)

Evaluates the current piece and the next piece together, finding the sequence of two moves that produces the best combined board state. The best move for the current piece is selected and returned.

---

## Output

### Annotated Image

![Annotated](doc/annotated_example.jpg)

The annotated screenshot shows:
- **Green column highlight** — the recommended column for placement
- **Ghost piece** — semi-transparent piece at the landing position
- **Info panel** — rotation angle, column, and Dellacherie score

### Confidence Heatmap

![Heatmap](doc/heatmap_example.jpg)

A per-column heatmap overlay using a Red→Yellow→Green palette:
- **Green** — high-scoring columns (preferred)
- **Yellow** — moderate scores
- **Red** — low-scoring columns (avoid)

### Batch Stitched Results

`batch_process.py` generates side-by-side composites: Original | Annotated | Heatmap.

### Debug Images (with `--debug` flag)

| File | Description |
|---|---|
| `debug/prefix_1_gray.jpg` | Grayscale conversion |
| `debug/prefix_2_blurred.jpg` | After Gaussian blur |
| `debug/prefix_3_binary.jpg` | After Otsu binarization |
| `debug/prefix_board.jpg` | Warped board region |
| `debug/prefix_next_region.jpg` | Warped next-piece region |
| `debug/prefix_grid_mask.png` | Combined HSV block mask |
| `debug/prefix_grid_overlay.jpg` | Grid with cell fill-ratio labels |
| `debug/prefix_active_piece.jpg` | Active piece highlighted |
| `debug/prefix_next_piece.jpg` | Next piece detection |
| `debug/prefix_region_annotated.jpg` | Board/next region bounding boxes |

---

## Project Structure

```
vision-Tetris/
├── main.py                          # Entry point, 7-stage pipeline
├── config.py                        # Central configuration
├── requirements.txt                 # Dependencies
├── batch_process.py                 # Batch processing + stitching
├── test_single.py                   # Single-image test with ground truth
├── test_validation.py               # Batch validation against annotations
├── analyze_results.py               # Validation analysis + statistics
├── data/                            # Dataset
│   ├── train/                       # Training images + _annotations.csv
│   ├── test/                        # Test images
│   └── valid/                       # Validation images
├── input/                           # Input screenshots directory
├── output/                          # Output directory
│   ├── debug/                       # Intermediate DIP debug images
│   ├── heatmap/                     # Heatmap images
│   ├── video/                       # Video mode output
│   └── results/                     # Batch stitched composites
├── src/
│   ├── preprocess.py                # Load, grayscale, blur, threshold
│   ├── board_detector.py            # Perspective warp, region extraction
│   ├── grid_parser.py               # HSV masking, grid parsing, validation
│   ├── piece_detector.py            # Connected-components, pattern matching, Hu moments
│   ├── move_generator.py            # Rotation generation, column validation
│   ├── simulator.py                 # Drop physics, collision, line clear
│   ├── scorer.py                    # Dellacherie 6-feature scoring
│   ├── engine.py                    # Two-piece lookahead engine
│   └── visualizer.py                # Annotation overlays, heatmap rendering
├── tests/                           # Unit tests (empty — run via test_validation.py)
└── *.md                             # Documentation files
```

---

## Results

On a validation set of 107 labeled training images, the system achieves:

| Metric | Value |
|---|---|
| Piece classification accuracy | ~100% |
| Per-piecetype accuracy (I/O/J/L/S/T/Z) | ~100% across all types |
| Pipeline latency | ~100-300ms per frame |

Validation is run via:

```bash
.venv/bin/python test_validation.py --verbose --save-results
```

Outputs per-class accuracy, confusion matrix, and saves `validation_results.csv`.

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `BOARD_CORNERS` | `(26,18), (471,18), (26,635), (471,635)` | Board region pixel coordinates |
| `NEXT_PIECE_CORNERS` | `(470,440), (640,440), (470,630), (640,630)` | Next-piece preview region |
| `GAUSSIAN_BLUR_KERNEL` | `(5, 5)` | Blur kernel size |
| `MIN_SATURATION` | `60` | Minimum S for colored blocks |
| `MIN_VALUE` | `60` | Minimum V for colored blocks |
| `MAX_SAT_GRAY` | `30` | Max S for gray/white blocks |
| `MIN_VAL_GRAY` | `150` | Min V for gray/white blocks |
| `HEATMAP_ALPHA` | `0.4` | Overlay opacity |
| `FRAME_DIFF_THRESHOLD` | `15` | Frame change threshold for video mode |

---

## Academic Context

This project was developed as part of a Digital Image Processing (DIP) course. It demonstrates practical application of:

- Image preprocessing and enhancement
- Color space analysis and segmentation
- Geometric transformations
- Morphological image processing
- Shape analysis and feature extraction
- Image compositing and visualization

---

## License

MIT
