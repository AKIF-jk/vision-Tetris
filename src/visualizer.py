from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

import config
from src.simulator import simulate_drop


def rotation_degrees(reference_piece: np.ndarray, rotated_piece: np.ndarray) -> Optional[int]:
    """Return the clockwise rotation angle in degrees if the matrices match."""
    reference = np.asarray(reference_piece, dtype=np.uint8)
    rotated = np.asarray(rotated_piece, dtype=np.uint8)

    for turns in range(4):
        if np.array_equal(np.rot90(reference, k=turns), rotated):
            return turns * 90
    return None


def _board_corners() -> np.ndarray:
    return np.asarray(config.BOARD_CORNERS, dtype=np.float32)


def _lerp(point_a: np.ndarray, point_b: np.ndarray, factor: float) -> np.ndarray:
    return point_a * (1.0 - factor) + point_b * factor


def _quad_point(corners: np.ndarray, u: float, v: float) -> np.ndarray:
    top_edge = _lerp(corners[0], corners[1], u)
    bottom_edge = _lerp(corners[2], corners[3], u)
    return _lerp(top_edge, bottom_edge, v)


def _cell_polygon(corners: np.ndarray, row: int, col: int) -> np.ndarray:
    u0 = col / config.GRID_COLS
    u1 = (col + 1) / config.GRID_COLS
    v0 = row / config.GRID_ROWS
    v1 = (row + 1) / config.GRID_ROWS

    points = np.array(
        [
            _quad_point(corners, u0, v0),
            _quad_point(corners, u1, v0),
            _quad_point(corners, u1, v1),
            _quad_point(corners, u0, v1),
        ],
        dtype=np.float32,
    )
    return np.round(points).astype(np.int32)


def _column_polygon(corners: np.ndarray, col: int) -> np.ndarray:
    u0 = col / config.GRID_COLS
    u1 = (col + 1) / config.GRID_COLS

    points = np.array(
        [
            _quad_point(corners, u0, 0.0),
            _quad_point(corners, u1, 0.0),
            _quad_point(corners, u1, 1.0),
            _quad_point(corners, u0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.round(points).astype(np.int32)


def _make_overlay(base_image: np.ndarray) -> np.ndarray:
    return np.zeros_like(base_image)


def _blend_overlay(base_image: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(base_image, 1.0, overlay, alpha, 0.0)


def _save_image(image: np.ndarray, output_path: Optional[str]) -> None:
    if not output_path:
        return
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    cv2.imwrite(output_path, image)


def _draw_text_panel(image: np.ndarray, lines: list[str], anchor: tuple[int, int]) -> None:
    x, y = anchor
    line_height = 22
    width = 250
    height = max(54, 18 + line_height * len(lines))

    cv2.rectangle(image, (x, y), (x + width, y + height), (18, 18, 18), -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 1)

    for idx, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (x + 12, y + 26 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            config.TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )


def annotate_move(
    screenshot: np.ndarray,
    best_rotation: np.ndarray,
    best_col: int,
    bbox: tuple[int, int, int, int],
    board: Optional[np.ndarray] = None,
    score: Optional[float] = None,
    rotation_label: Optional[int] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Draw the recommended move overlay on the raw screenshot."""
    annotated = screenshot.copy()
    overlay = _make_overlay(annotated)
    corners = _board_corners()

    # Keep a subtle bbox cue for screenshots where the board corners are not obvious.
    cv2.rectangle(
        overlay,
        (bbox[0], bbox[1]),
        (bbox[2], bbox[3]),
        (0, 200, 0),
        1,
    )

    # Highlight the recommended target column.
    column_polygon = _column_polygon(corners, best_col)
    cv2.fillConvexPoly(overlay, column_polygon, (0, 120, 0))
    cv2.polylines(overlay, [column_polygon], True, config.BEST_MOVE_COLOR, 2)

    # Draw the ghost piece at the landing location if the current board is known.
    if board is not None:
        board_after, _ = simulate_drop(board, best_rotation, best_col)
        placed_cells = np.argwhere((board_after == 1) & (board == 0))
        for row, col in placed_cells:
            cell_polygon = _cell_polygon(corners, int(row), int(col))
            cv2.fillConvexPoly(overlay, cell_polygon, config.GHOST_PIECE_COLOR)
            cv2.polylines(overlay, [cell_polygon], True, config.BEST_MOVE_COLOR, 1)

    annotated = _blend_overlay(annotated, overlay, config.HEATMAP_ALPHA)

    rotation_text = f"rotation: {rotation_label}°" if rotation_label is not None else "rotation: unknown"
    score_text = f"score: {score:.2f}" if score is not None else "score: n/a"
    panel_width = 250
    panel_x = min(max(10, bbox[2] + 12), max(10, annotated.shape[1] - panel_width - 10))
    _draw_text_panel(annotated, ["Best move", rotation_text, f"column: {best_col}", score_text], (panel_x, 14))

    _save_image(annotated, output_path)
    return annotated


def draw_heatmap(
    screenshot: np.ndarray,
    scores_matrix: np.ndarray,
    bbox: tuple[int, int, int, int],
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Render a per-column confidence heatmap over the board region."""
    heatmap = screenshot.copy()
    overlay = _make_overlay(heatmap)
    corners = _board_corners()

    column_scores = np.full(scores_matrix.shape[1], -np.inf, dtype=np.float32)
    for col in range(scores_matrix.shape[1]):
        valid = scores_matrix[:, col][np.isfinite(scores_matrix[:, col])]
        if valid.size:
            column_scores[col] = float(valid.max())

    finite_mask = np.isfinite(column_scores)
    if finite_mask.any():
        finite_values = column_scores[finite_mask]
        min_score = float(finite_values.min())
        max_score = float(finite_values.max())
        if np.isclose(max_score, min_score):
            normalized = np.where(finite_mask, 0.5, 0.0)
        else:
            normalized = np.where(
                finite_mask,
                (column_scores - min_score) / (max_score - min_score),
                0.0,
            )
    else:
        normalized = np.zeros_like(column_scores, dtype=np.float32)

    palette_input = np.clip(normalized * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
    palette = cv2.applyColorMap(palette_input, cv2.COLORMAP_RdYlGn)

    for col in range(scores_matrix.shape[1]):
        color = tuple(int(channel) for channel in palette[col, 0])
        column_polygon = _column_polygon(corners, col)
        cv2.fillConvexPoly(overlay, column_polygon, color)
        cv2.polylines(overlay, [column_polygon], True, (255, 255, 255), 1)

    heatmap = _blend_overlay(heatmap, overlay, config.HEATMAP_ALPHA)

    legend_x = max(10, heatmap.shape[1] - 210)
    legend_y = 14
    cv2.rectangle(heatmap, (legend_x, legend_y), (legend_x + 190, legend_y + 72), (18, 18, 18), -1)
    cv2.rectangle(heatmap, (legend_x, legend_y), (legend_x + 190, legend_y + 72), (0, 255, 0), 1)
    cv2.putText(heatmap, "Heatmap", (legend_x + 10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.TEXT_COLOR, 1, cv2.LINE_AA)

    low_color = tuple(int(v) for v in cv2.applyColorMap(np.array([[0]], dtype=np.uint8), cv2.COLORMAP_RdYlGn)[0, 0])
    high_color = tuple(int(v) for v in cv2.applyColorMap(np.array([[255]], dtype=np.uint8), cv2.COLORMAP_RdYlGn)[0, 0])

    cv2.rectangle(heatmap, (legend_x + 10, legend_y + 30), (legend_x + 36, legend_y + 48), low_color, -1)
    cv2.rectangle(heatmap, (legend_x + 10, legend_y + 50), (legend_x + 36, legend_y + 68), high_color, -1)
    cv2.putText(heatmap, "Low", (legend_x + 44, legend_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(heatmap, "High", (legend_x + 44, legend_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.TEXT_COLOR, 1, cv2.LINE_AA)

    _save_image(heatmap, output_path)
    return heatmap