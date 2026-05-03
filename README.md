# Vision-Based Tetris Advisor

A powerful computer vision tool designed to analyze Tetris gameplay screenshots in real-time. The system perceives the board state, identifies the current and next pieces, and calculates the mathematically optimal move using advanced Digital Image Processing (DIP) and game theory.

---

## 🚀 Current Status: Phases 1, 2 & 3 Complete ✅

We have successfully implemented the complete vision pipeline with AI-driven decision making. The system can now "see" the game board, analyze it in real-time, and recommend the mathematically optimal move using a two-piece lookahead algorithm.

### ✅ Implemented Features (Phase 1, 2 & 3)

#### 1. Advanced DIP Preprocessing
- **Adaptive Thresholding**: Uses Otsu's method to handle varied lighting.
- **Noise Reduction**: Gaussian filtering and morphological operations.

#### 2. Robust Board & Grid Analysis
- **Perspective Extraction**: Isolates the board region and applies a warp transform.
- **Precision Grid Parser**: Divides the board into a 20x10 matrix.

#### 3. Deterministic Piece Detection
- **Pattern Matching Engine**: Matches grid connectivity against a tetromino database.
- **Active & Next Piece Identification**: Locates and classifies both current and upcoming pieces.

#### 4. High-Performance AI Engine (Phase 3)
- **Move Generator**: Generates all valid unique rotations and drop positions.
- **Board Simulator**: Predicting board states after piece placement.
- **Dellacherie Algorithm**: Implementation of the 6-feature scoring heuristic.
- **Lookahead Engine**: Two-piece lookahead for globally optimal moves.

---

## ⏭ Next Phases: Visualization & Optimization (Phases 4 & 5)
We are now entering the refinement and visualization stages:
- **Phase 4**: Move recommendation overlays, heatmaps, and real-time video support.
- **Phase 5**: Advanced search (Expectiminimax), performance tuning, and web dashboard.

See [PHASE4_PHASE5_PLAN.md](PHASE4_PHASE5_PLAN.md) for the detailed roadmap.

---

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/[your-username]/vision-Tetris.git
   cd vision-Tetris
   ```

2. **Set up the environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/python3 -m pip install -r requirements.txt
   ```

---

## 📖 Usage

### Setup Virtual Environment (First Time Only)
```bash
source .venv/bin/activate
```

### Run the Pipeline on a Single Screenshot

**Using the virtual environment:**
```bash
.venv/bin/python main.py --input data/train/sample_screenshot.jpg
```

Or if you've activated the venv:
```bash
python main.py --input data/train/sample_screenshot.jpg
```

**Optional Flags:**
- `--verbose`: Print detailed per-step timing and module status.
- `--debug`: Save intermediate DIP images (masks, warped boards, etc.) to `output/debug/` and display the AI scores matrix.

**Full Example with All Options:**
```bash
.venv/bin/python main.py --input data/train/sample_screenshot.jpg --verbose --debug
```

---

## 👥 Group Members
- [Your Name]
- [Group Member Name]
- [Group Member Name]
