# Roadmap: Phases 4 & 5 — Vision-Tetris Advisor

With **Phase 3 (AI Engine Integration)** successfully completed, we now have a system that can "see" the board and "think" ahead. The next phases focus on **making the AI's logic visible** and **optimizing its intelligence** for professional-grade play.

---

## 🎨 Phase 4: Output & Visualization
**Goal:** Implement visual feedback mechanisms to display AI recommendations and confidence.

### 1. Move Annotation Overlay (Task 12)
*   **Visual Ghost Piece:** Draw a "piece ghost" at the recommended (rotation, column) using `cv2` drawing functions.
*   **Column Highlighting:** Highlight the target column with a colored rectangle.
*   **Annotation:** Add text for the recommended rotation and the move score.
*   **Specification:** Color: Green for best move, white text. Must not obscure board content.
*   **Function:** `annotate_move(screenshot, best_rotation, best_col, bbox) -> annotated_img`.

### 2. Confidence Heatmap Overlay (Task 13)
*   **Score Normalization:** Flatten scores to per-column max score and normalize to 0–1.
*   **Colormap Mapping:** Use `cv2.applyColorMap` (COLORMAP_RdYlGn) to map normalized scores to colors.
*   **Transparency:** Overlay semi-transparent (alpha=0.4) colored bars on each column.
*   **Color Legend:** Draw a legend in the corner (Low = Red, High = Green).
*   **Function:** `draw_heatmap(screenshot, scores_matrix, bbox) -> heatmap_img`.

---

## 📜 Phase 5: Reporting, Documentation & Submission
**Goal:** Formalize the project for academic submission, demonstrating deep understanding of Digital Image Processing (DIP) principles and pipeline integration.

### 1. DIP Theoretical Report (Academic Focus)
*   **Algorithm Rationale:** Document the mathematical basis for key DIP steps (Otsu's binarization, Gaussian kernel selection, Homography for perspective correction).
*   **Feature Analysis:** Explain the use of Hu Moments for shape invariant classification and the logic behind cell-intensity thresholding for grid parsing.
*   **Learning Demonstration:** Section dedicated to "Challenges & DIP Solutions" (e.g., handling variable lighting via adaptive thresholding, morphological noise filtering).

### 2. Comprehensive Documentation & Architecture
*   **Architecture Diagram:** Create a visual flow of the DIP pipeline from raw pixel input to AI move recommendation.
*   **Module API Guide:** Detailed documentation for `preprocess.py`, `board_detector.py`, `piece_detector.py`, and `visualizer.py`.
*   **Setup & Usage:** Professional `README` refinement with environment specs and dependency logic.

### 3. Final Integration & Evaluation (Task 14 & 15)
*   **Unified Pipeline:** Final validation of the `main.py` entry point chaining all 6 modules.
*   **Accuracy Benchmark:** Final report on grid parsing (>90%) and piece detection (>85%) across the test dataset.
*   **Failure Case Study:** Analysis of edge cases (overlapping UI elements, low-contrast themes) and how DIP robustness was improved to handle them.

### 4. Video Presentation & Demo (Stretch Goal)
*   **Visual Results:** Exported video of the advisor in action, highlighting real-time DIP performance.
*   **Process Walkthrough:** A brief video or document explaining the "Behind the Scenes" of the frame differencing and piece spawn detection logic.

---

## 📅 Task Checklist

| Phase | Task # | Title | Status |
| :--- | :--- | :--- | :--- |
| Phase 4 | 12 | Move annotation overlay | To Do |
| Phase 4 | 13 | Confidence heatmap overlay | To Do |
| Phase 5 | 14 | Final Pipeline Integration | To Do |
| Phase 5 | 15 | Accuracy & Failure Analysis | To Do |
| Phase 5 | 17 | DIP Theoretical Report | To Do |
| Phase 5 | 18 | Final Documentation & Submission | To Do |

---

## 🚀 Next Immediate Action
We will begin **Task 12** by creating the `src/visualizer.py` module to handle the move annotation overlay.
