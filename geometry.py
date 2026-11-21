from __future__ import annotations

from collections.abc import Sequence

import numpy as np


Point2D = tuple[float, float]


def _points_array(points: Sequence[Point2D]) -> np.ndarray | None:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        return None
    return array


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _ordered_quad(points: np.ndarray) -> np.ndarray | None:
    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    ordered = points[order]
    edges = np.roll(ordered, -1, axis=0) - ordered
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.any(np.abs(crosses) < 1e-6) or not (np.all(crosses > 0) or np.all(crosses < 0)):
        return None
    return ordered


def is_valid_triangle(points: Sequence[Point2D]) -> bool:
    """Accept a clearly non-degenerate triangle with reasonably balanced sides."""
    array = _points_array(points)
    if array is None or len(array) != 3:
        return False

    sides = np.array(
        [
            np.linalg.norm(array[0] - array[1]),
            np.linalg.norm(array[1] - array[2]),
            np.linalg.norm(array[2] - array[0]),
        ]
    )
    if sides.min() < 0.025 or sides.max() / sides.min() > 2.6:
        return False

    return _polygon_area(array) >= 0.0015


def is_valid_square(points: Sequence[Point2D]) -> bool:
    """Validate a four-corner, near-square polygon using its sides and diagonals."""
    array = _points_array(points)
    if array is None or len(array) != 4:
        return False

    ordered = _ordered_quad(array)
    if ordered is None:
        return False

    edges = np.roll(ordered, -1, axis=0) - ordered
    side_lengths = np.linalg.norm(edges, axis=1)
    if side_lengths.min() < 0.025 or side_lengths.max() / side_lengths.min() > 1.75:
        return False

    diagonals = np.array(
        [
            np.linalg.norm(ordered[0] - ordered[2]),
            np.linalg.norm(ordered[1] - ordered[3]),
        ]
    )
    if diagonals.min() < 0.04 or diagonals.max() / diagonals.min() > 1.4:
        return False

    cos_angles = np.abs(
        np.sum(edges * np.roll(edges, -1, axis=0), axis=1)
        / (side_lengths * np.roll(side_lengths, -1))
    )
    if np.any(cos_angles > 0.45):
        return False

    return _polygon_area(ordered) >= 0.003


def draw_geometry(
    canvas,
    label: str | None,
    points: Sequence[Point2D] = (),
) -> None:
    """Draw the recognized shape in a Tk-compatible canvas."""
    canvas.delete("all")
    width = max(canvas.winfo_width(), 1)
    height = max(canvas.winfo_height(), 1)
    center_x, center_y = width / 2, height / 2
    canvas.create_oval(
        center_x - 2,
        center_y - 2,
        center_x + 2,
        center_y + 2,
        fill="#404040",
        outline="",
    )

    if label is None:
        canvas.create_text(
            center_x,
            center_y + 48,
            text="Waiting for a gesture",
            fill="#858585",
            font=("Helvetica", 12),
        )
        return

    color = "#B9E6C5"
    pad = min(width, height) * 0.2
    if label == "CIRCLE":
        radius = min(width, height) * 0.27
        canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            outline=color,
            width=5,
        )
    elif label == "V SHAPE":
        span = min(width, height) * 0.32
        canvas.create_line(
            center_x - span,
            center_y - span * 0.7,
            center_x,
            center_y + span,
            center_x + span,
            center_y - span * 0.7,
            fill=color,
            width=6,
            joinstyle="round",
        )
    elif label in {"TRIANGLE", "SQUARE"} and len(points) in {3, 4}:
        array = np.asarray(points, dtype=np.float64)
        minimum = array.min(axis=0)
        maximum = array.max(axis=0)
        extent = np.maximum(maximum - minimum, 0.03)
        scale = min((width - 2 * pad) / extent[0], (height - 2 * pad) / extent[1])
        placed = (array - (minimum + maximum) / 2) * scale
        placed[:, 0] += center_x
        placed[:, 1] += center_y
        flattened = placed.ravel().tolist()
        canvas.create_polygon(
            *flattened,
            fill="",
            outline=color,
            width=5,
            joinstyle="round",
        )
        for x, y in placed:
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="")

    canvas.create_text(
        center_x,
        height - 19,
        text=label,
        fill="#F2F2F2",
        font=("Avenir Next", 11, "bold"),
    )
