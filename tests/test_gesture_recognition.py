import unittest
from dataclasses import dataclass

import numpy as np

from geometry import is_valid_square, is_valid_triangle
from gesture_recognition import (
    GestureRecognizer,
    GestureSmoother,
    RecognitionResult,
)


@dataclass
class HandFixture:
    landmarks: np.ndarray


def make_hand(*, circle: bool = False) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = (0.50, 0.80, 0.0)
    points[1] = (0.44, 0.70, 0.0)
    points[2] = (0.41, 0.66, 0.0)
    points[3] = (0.38, 0.62, 0.0)
    points[4] = (0.38, 0.65, 0.0)

    points[5] = (0.43, 0.56, 0.0)
    points[6] = (0.43, 0.45, 0.0)
    points[7] = (0.43, 0.35, 0.0)
    points[8] = (0.43, 0.24, 0.0)

    points[9] = (0.50, 0.54, 0.0)
    points[10] = (0.50, 0.42, 0.0)
    points[11] = (0.50, 0.31, 0.0)
    points[12] = (0.50, 0.19, 0.0)

    points[13] = (0.56, 0.56, 0.0)
    points[14] = (0.56, 0.49, 0.0)
    points[15] = (0.56, 0.51, 0.0)
    points[16] = (0.56, 0.55, 0.0)

    points[17] = (0.62, 0.57, 0.0)
    points[18] = (0.62, 0.52, 0.0)
    points[19] = (0.62, 0.56, 0.0)
    points[20] = (0.62, 0.60, 0.0)

    if circle:
        points[4, :2] = points[8, :2] + (0.01, 0.01)
    return points


def make_corner_hand(
    index_tip: tuple[float, float],
    middle_tip: tuple[float, float],
    center_x: float,
) -> HandFixture:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = (center_x, 0.62, 0.0)
    points[1:5] = (
        (center_x + 0.10, 0.60, 0.0),
        (center_x + 0.12, 0.58, 0.0),
        (center_x + 0.14, 0.57, 0.0),
        (center_x + 0.14, 0.59, 0.0),
    )
    points[5:9] = (
        (center_x, 0.56, 0.0),
        (center_x, 0.55, 0.0),
        (center_x, 0.48, 0.0),
        (*index_tip, 0.0),
    )
    points[9:13] = (
        (center_x + 0.02, 0.56, 0.0),
        (center_x + 0.02, 0.55, 0.0),
        (center_x + 0.02, 0.48, 0.0),
        (*middle_tip, 0.0),
    )
    points[13:17] = (
        (center_x + 0.05, 0.57, 0.0),
        (center_x + 0.05, 0.52, 0.0),
        (center_x + 0.05, 0.56, 0.0),
        (center_x + 0.05, 0.60, 0.0),
    )
    points[17:21] = (
        (center_x + 0.08, 0.58, 0.0),
        (center_x + 0.08, 0.53, 0.0),
        (center_x + 0.08, 0.57, 0.0),
        (center_x + 0.08, 0.61, 0.0),
    )
    return HandFixture(points)


def make_pointing_hand(index_tip: tuple[float, float], center_x: float) -> HandFixture:
    points = make_hand().copy()
    points += (center_x - 0.5, 0.0, 0.0)
    points[0] = (center_x, 0.80, 0.0)
    points[5] = (center_x - 0.06, 0.58, 0.0)
    points[6] = (center_x - 0.06, 0.60, 0.0)
    points[7] = (center_x - 0.12, 0.60, 0.0)
    points[8] = (*index_tip, 0.0)
    points[10] = (center_x, 0.58, 0.0)
    points[11] = (center_x, 0.67, 0.0)
    points[12] = (center_x, 0.72, 0.0)
    points[14] = (center_x + 0.06, 0.60, 0.0)
    points[16] = (center_x + 0.06, 0.68, 0.0)
    points[18] = (center_x + 0.12, 0.62, 0.0)
    points[20] = (center_x + 0.12, 0.70, 0.0)
    return HandFixture(points)


class GeometryTests(unittest.TestCase):
    def test_triangle_requires_three_non_collinear_corners(self) -> None:
        self.assertTrue(is_valid_triangle(((0.2, 0.2), (0.5, 0.2), (0.35, 0.48))))
        self.assertFalse(is_valid_triangle(((0.2, 0.2), (0.4, 0.2), (0.6, 0.2))))

    def test_square_rejects_stretched_rectangle(self) -> None:
        self.assertTrue(is_valid_square(((0.3, 0.3), (0.5, 0.3), (0.5, 0.5), (0.3, 0.5))))
        self.assertFalse(is_valid_square(((0.1, 0.35), (0.8, 0.35), (0.8, 0.55), (0.1, 0.55))))


class GestureRecognitionTests(unittest.TestCase):
    def test_circle_takes_priority_for_thumb_index_pinch(self) -> None:
        result = GestureRecognizer().recognize([HandFixture(make_hand(circle=True))])
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "CIRCLE")

    def test_v_shape_requires_two_extended_adjacent_fingers(self) -> None:
        hand = HandFixture(make_hand())
        result = GestureRecognizer().recognize([hand])
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "V SHAPE")

    def test_two_hand_triangle_uses_three_valid_fingertip_corners(self) -> None:
        v_hand = make_corner_hand((0.30, 0.35), (0.45, 0.35), 0.30)
        point_hand = make_pointing_hand((0.45, 0.63), 0.70)
        result = GestureRecognizer().recognize([v_hand, point_hand])
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "TRIANGLE")
        self.assertEqual(len(result.points), 3)

    def test_two_hand_square_uses_four_convex_fingertip_corners(self) -> None:
        left_hand = make_corner_hand((0.30, 0.30), (0.30, 0.50), 0.30)
        right_hand = make_corner_hand((0.50, 0.30), (0.50, 0.50), 0.50)
        result = GestureRecognizer().recognize([left_hand, right_hand])
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "SQUARE")
        self.assertEqual(len(result.points), 4)

    def test_smoother_waits_for_majority_and_clears_after_stable_empty_frames(self) -> None:
        smoother = GestureSmoother(window_size=5, votes_required=3)
        circle = RecognitionResult("CIRCLE", "Circle")
        v_shape = RecognitionResult("V SHAPE", "V Shape")

        self.assertIsNone(smoother.update(circle))
        self.assertIsNone(smoother.update(circle))
        self.assertEqual(smoother.update(circle).label, "CIRCLE")
        self.assertEqual(smoother.update(v_shape).label, "CIRCLE")
        self.assertEqual(smoother.update(v_shape).label, "CIRCLE")
        self.assertEqual(smoother.update(v_shape).label, "V SHAPE")
        self.assertEqual(smoother.update(None).label, "V SHAPE")
        self.assertEqual(smoother.update(None).label, "V SHAPE")
        self.assertIsNone(smoother.update(None))


if __name__ == "__main__":
    unittest.main()
