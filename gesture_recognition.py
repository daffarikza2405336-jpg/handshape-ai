from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from geometry import is_valid_square, is_valid_triangle


class HandLike(Protocol):
    landmarks: np.ndarray


@dataclass(frozen=True)
class RecognitionResult:
    label: str
    shape_name: str
    points: tuple[tuple[float, float], ...] = ()


class GestureRecognizer:
    """Rule-based recognition over normalized 21-point hand landmarks."""

    def recognize(self, hands: list[HandLike]) -> RecognitionResult | None:
        if not hands:
            return None

        # A pinch is the least ambiguous one-hand gesture, so give it priority.
        for hand in hands:
            if self._is_circle(hand.landmarks):
                return RecognitionResult("CIRCLE", "Circle")

        if len(hands) >= 2:
            ordered_hands = sorted(hands[:2], key=self._hand_center_x)
            first, second = ordered_hands
            first_v = self._is_v_shape(first.landmarks)
            second_v = self._is_v_shape(second.landmarks)

            if first_v and second_v:
                corners = (
                    self._xy(first.landmarks[[8, 12]]).tolist()
                    + self._xy(second.landmarks[[8, 12]]).tolist()
                )
                if is_valid_square(corners):
                    return RecognitionResult(
                        "SQUARE",
                        "Square",
                        tuple((float(x), float(y)) for x, y in corners),
                    )

            first_pointer = self._is_index_point(first.landmarks)
            second_pointer = self._is_index_point(second.landmarks)
            if first_v and second_pointer:
                corners = self._xy(
                    np.vstack((first.landmarks[[8, 12]], second.landmarks[[8]]))
                )
                if is_valid_triangle(corners):
                    return RecognitionResult(
                        "TRIANGLE",
                        "Triangle",
                        tuple((float(x), float(y)) for x, y in corners),
                    )
            if second_v and first_pointer:
                corners = self._xy(
                    np.vstack((second.landmarks[[8, 12]], first.landmarks[[8]]))
                )
                if is_valid_triangle(corners):
                    return RecognitionResult(
                        "TRIANGLE",
                        "Triangle",
                        tuple((float(x), float(y)) for x, y in corners),
                    )

        for hand in hands:
            if self._is_v_shape(hand.landmarks):
                return RecognitionResult("V SHAPE", "V Shape")
        return None

    @staticmethod
    def _xy(points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64)[..., :2]

    @staticmethod
    def _hand_center_x(hand: HandLike) -> float:
        return float(np.mean(hand.landmarks[[0, 5, 9, 13, 17], 0]))

    @classmethod
    def _palm_scale(cls, landmarks: np.ndarray) -> float:
        xy = cls._xy(landmarks)
        width = np.linalg.norm(xy[5] - xy[17])
        length = np.linalg.norm(xy[0] - xy[9])
        return max(float(width), float(length), 0.04)

    @classmethod
    def _is_circle(cls, landmarks: np.ndarray) -> bool:
        xy = cls._xy(landmarks)
        distance = np.linalg.norm(xy[4] - xy[8])
        return distance / cls._palm_scale(landmarks) <= 0.36

    @classmethod
    def _finger_extended(cls, landmarks: np.ndarray, tip: int, joint: int) -> bool:
        xy = cls._xy(landmarks)
        wrist_distance = np.linalg.norm(xy[tip] - xy[0])
        joint_distance = np.linalg.norm(xy[joint] - xy[0])
        return wrist_distance > joint_distance * 1.16

    @classmethod
    def _thumb_extended(cls, landmarks: np.ndarray) -> bool:
        return cls._finger_extended(landmarks, 4, 3)

    @classmethod
    def _finger_states(cls, landmarks: np.ndarray) -> tuple[bool, bool, bool, bool]:
        return (
            cls._finger_extended(landmarks, 8, 6),
            cls._finger_extended(landmarks, 12, 10),
            cls._finger_extended(landmarks, 16, 14),
            cls._finger_extended(landmarks, 20, 18),
        )

    @classmethod
    def _is_v_shape(cls, landmarks: np.ndarray) -> bool:
        index, middle, ring, pinky = cls._finger_states(landmarks)
        spacing = np.linalg.norm(cls._xy(landmarks[8]) - cls._xy(landmarks[12]))
        return (
            index
            and middle
            and not ring
            and not pinky
            and not cls._thumb_extended(landmarks)
            and spacing / cls._palm_scale(landmarks) >= 0.22
        )

    @classmethod
    def _is_index_point(cls, landmarks: np.ndarray) -> bool:
        index, middle, ring, pinky = cls._finger_states(landmarks)
        return (
            index
            and not middle
            and not ring
            and not pinky
            and not cls._thumb_extended(landmarks)
        )


class GestureSmoother:
    """Use a short majority window so transient frame changes do not flash in the UI."""

    def __init__(self, window_size: int = 7, votes_required: int = 4) -> None:
        if window_size < 1 or votes_required < 1 or votes_required > window_size:
            raise ValueError("votes_required must be between 1 and window_size")
        self._history: deque[RecognitionResult | None] = deque(maxlen=window_size)
        self._votes_required = votes_required
        self._stable_result: RecognitionResult | None = None

    def update(self, result: RecognitionResult | None) -> RecognitionResult | None:
        self._history.append(result)
        counts = Counter(item.label if item else None for item in self._history)
        winner, votes = counts.most_common(1)[0]
        if votes < self._votes_required:
            return self._stable_result

        if winner is None:
            self._stable_result = None
        else:
            self._stable_result = next(
                item for item in reversed(self._history) if item and item.label == winner
            )
        return self._stable_result
