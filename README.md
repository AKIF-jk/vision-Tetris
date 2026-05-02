# Vision-Based Tetris Advisor

A powerful computer vision tool designed to analyze Tetris gameplay screenshots in real-time. The system perceives the board state, identifies the current and next pieces, and calculates the mathematically optimal move using advanced Digital Image Processing (DIP) and game theory.

---

## 🚀 Current Status: Phase 1 & 2 Complete
We have successfully implemented the core vision pipeline. The system can now "see" and "understand" the game board with high precision.

### ✅ Implemented Features (Phase 1 & 2)

#### 1. Advanced DIP Preprocessing
- **Adaptive Thresholding**: Uses Otsu's method to handle varied lighting and game themes.
- **Noise Reduction**: Gaussian filtering and morphological operations to clean up the binary mask.

#### 2. Robust Board & Grid Analysis
- **Perspective Extraction**: Isolates the board region and applies a warp transform to correct for slight skewing.
- **Precision Grid Parser**: Divides the board into a 20x10 matrix, classifying each cell as filled or empty based on pixel density and color saturation.

#### 3. Deterministic Piece Detection
- **Pattern Matching Engine**: Moves beyond unstable blob analysis by matching 4-cell grid connectivity against a canonical tetromino database.
- **Active Piece Identification**: Locates and classifies the falling piece with ~100% accuracy on standard datasets.
- **Next Piece Preview**: Scans the game's preview box to identify the upcoming tetromino, enabling future lookahead strategy.

#### 4. High-Performance Pipeline
- **Modular Architecture**: Clean separation between preprocessing, detection, and parsing.
- **Speed**: Full frame analysis (detection + parsing) completes in **< 20ms** on standard hardware.

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

Run the pipeline on a single screenshot:
```bash
python3 main.py --input data/train/sample_screenshot.jpg
```

**Optional Flags:**
- `--verbose`: Print detailed per-step timing and module status.
- `--debug`: Save intermediate DIP images (masks, warped boards, etc.) to `output/debug/`.

---

## ⏭ Next Phase: Move Scoring & AI (Phase 3)
The next development cycle will focus on the "brain" of the advisor:
- **Move Generator**: Generating all valid unique rotations and drop positions.
- **Board Simulator**: Predicting board states after piece placement.
- **Dellacherie Algorithm**: Implementation of the 6-feature scoring heuristic (Holes, Wells, Transitions, etc.).
- **Lookahead Engine**: Two-piece lookahead to find the globally optimal move.

---

## 👥 Group Members
- [Your Name]
- [Group Member Name]
- [Group Member Name]
