from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from config import (
    HAND_MODEL_PATH,
    HAND_SCORE_THRESHOLD,
    MAX_HANDS,
    PALM_MODEL_PATH,
    PALM_SCORE_THRESHOLD,
)


@dataclass(frozen=True)
class DetectedHand:
    """A detected hand with 21 normalized (x, y, z) landmarks."""

    landmarks: np.ndarray
    confidence: float
    handedness: str


class HandDetector:
    """OpenCV DNN palm detection followed by 21-point hand-pose estimation.

    The ONNX models are from OpenCV Zoo and are distributed with their
    Apache-2.0 notices in assets/licenses.
    """

    PALM_INPUT_SIZE = 192
    HAND_INPUT_SIZE = 224
    PALM_STRIDES = (8, 16, 16, 16)

    def __init__(
        self,
        palm_model_path: Path = PALM_MODEL_PATH,
        hand_model_path: Path = HAND_MODEL_PATH,
        max_hands: int = MAX_HANDS,
    ) -> None:
        missing = [path for path in (palm_model_path, hand_model_path) if not path.is_file()]
        if missing:
            missing_list = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "Required ONNX model file(s) are missing:\n"
                f"{missing_list}\n"
                "Restore the model files under assets/models before starting the app."
            )
        if max_hands < 1:
            raise ValueError("max_hands must be at least 1")

        self.max_hands = max_hands
        self._palm_net = cv.dnn.readNet(str(palm_model_path))
        self._hand_net = cv.dnn.readNet(str(hand_model_path))
        self._palm_anchors = self._create_palm_anchors()

    def detect(self, frame: np.ndarray) -> list[DetectedHand]:
        """Return up to max_hands detections without failing on an empty frame."""
        if frame is None or frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return []

        palms = self._detect_palms(frame)
        detected: list[DetectedHand] = []
        height, width = frame.shape[:2]
        depth_scale = float(max(width, height))

        for palm in palms:
            pose = self._estimate_hand_pose(frame, palm)
            if pose is None:
                continue
            landmarks, confidence, handedness = pose
            if confidence < HAND_SCORE_THRESHOLD:
                continue

            normalized = landmarks.copy()
            normalized[:, 0] /= width
            normalized[:, 1] /= height
            normalized[:, 2] /= depth_scale
            normalized[:, 0:2] = np.clip(normalized[:, 0:2], 0.0, 1.0)
            detected.append(
                DetectedHand(
                    landmarks=normalized,
                    confidence=confidence,
                    handedness=handedness,
                )
            )

        detected.sort(key=lambda hand: float(np.mean(hand.landmarks[:, 0])))
        return detected[: self.max_hands]

    @classmethod
    def _create_palm_anchors(cls) -> np.ndarray:
        anchors: list[tuple[float, float]] = []
        for stride in cls.PALM_STRIDES:
            grid_size = cls.PALM_INPUT_SIZE // stride
            for row in range(grid_size):
                for column in range(grid_size):
                    center = (
                        (column + 0.5) / grid_size,
                        (row + 0.5) / grid_size,
                    )
                    anchors.extend((center, center))
        return np.asarray(anchors, dtype=np.float32)

    @staticmethod
    def _letterbox_palm(frame: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        input_size = HandDetector.PALM_INPUT_SIZE
        height, width = frame.shape[:2]
        ratio = min(input_size / height, input_size / width)
        resized_width = max(1, int(width * ratio))
        resized_height = max(1, int(height * ratio))
        resized = cv.resize(frame, (resized_width, resized_height))

        pad_width = input_size - resized_width
        pad_height = input_size - resized_height
        left, top = pad_width // 2, pad_height // 2
        padded = cv.copyMakeBorder(
            resized,
            top,
            pad_height - top,
            left,
            pad_width - left,
            cv.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        rgb = cv.cvtColor(padded, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pad_bias = np.array([left, top], dtype=np.float32) / ratio
        return rgb[np.newaxis, ...], ratio, pad_bias

    def _detect_palms(self, frame: np.ndarray) -> list[np.ndarray]:
        height, width = frame.shape[:2]
        input_blob, _, pad_bias = self._letterbox_palm(frame)
        self._palm_net.setInput(input_blob)
        outputs = self._palm_net.forward(self._palm_net.getUnconnectedOutLayersNames())
        if len(outputs) < 2:
            raise RuntimeError("The palm ONNX model returned an unexpected output layout.")

        raw_boxes = np.asarray(outputs[0])[0]
        raw_scores = np.asarray(outputs[1])[0].reshape(-1)
        if len(raw_boxes) != len(self._palm_anchors) or len(raw_scores) != len(self._palm_anchors):
            raise RuntimeError("Palm model output does not match the generated anchor layout.")

        scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores.astype(np.float64), -80, 80)))
        keep = np.flatnonzero(scores >= PALM_SCORE_THRESHOLD)
        if not len(keep):
            return []

        scale = float(max(width, height))
        box_delta = raw_boxes[keep, :4] / self.PALM_INPUT_SIZE
        selected_anchors = self._palm_anchors[keep]
        centers = (box_delta[:, :2] + selected_anchors) * scale
        sizes = box_delta[:, 2:] * scale
        top_left = centers - sizes / 2 - pad_bias
        bottom_right = centers + sizes / 2 - pad_bias
        boxes = np.concatenate((top_left, bottom_right), axis=1)
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width - 1)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height - 1)

        landmark_delta = raw_boxes[keep, 4:18].reshape(-1, 7, 2)
        palm_landmarks = (
            landmark_delta / self.PALM_INPUT_SIZE + selected_anchors[:, None, :]
        ) * scale - pad_bias

        nms_boxes = [
            [
                float(box[0]),
                float(box[1]),
                float(max(box[2] - box[0], 1)),
                float(max(box[3] - box[1], 1)),
            ]
            for box in boxes
        ]
        selected_indices = cv.dnn.NMSBoxes(
            nms_boxes,
            scores[keep].astype(float).tolist(),
            PALM_SCORE_THRESHOLD,
            0.3,
            top_k=64,
        )
        if len(selected_indices) == 0:
            return []

        selected = np.asarray(selected_indices).reshape(-1)
        selected = sorted(selected, key=lambda index: scores[keep[index]], reverse=True)
        palms: list[np.ndarray] = []
        for index in selected[: self.max_hands]:
            palm = np.concatenate(
                (
                    boxes[index],
                    palm_landmarks[index].reshape(-1),
                    np.array([scores[keep[index]]], dtype=np.float32),
                )
            )
            palms.append(palm.astype(np.float32))
        return palms

    @staticmethod
    def _crop_and_pad(
        image: np.ndarray,
        box: np.ndarray,
        *,
        before_rotation: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        bounds = np.asarray(box, dtype=np.float32).copy()
        width_height = bounds[1] - bounds[0]
        if np.any(width_height <= 0):
            return None

        if before_rotation:
            shift = np.array([0.0, 0.0], dtype=np.float32)
            enlarge_factor = 4.0
        else:
            shift = np.array([0.0, -0.4], dtype=np.float32) * width_height
            enlarge_factor = 3.0

        bounds += shift
        center = bounds.mean(axis=0)
        half_size = (bounds[1] - bounds[0]) * enlarge_factor / 2
        bounds = np.array([center - half_size, center + half_size], dtype=np.float32)
        bounds = np.rint(bounds).astype(np.int32)

        height, width = image.shape[:2]
        bounds[:, 0] = np.clip(bounds[:, 0], 0, width)
        bounds[:, 1] = np.clip(bounds[:, 1], 0, height)
        x1, y1 = bounds[0]
        x2, y2 = bounds[1]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if before_rotation:
            side = int(np.ceil(np.linalg.norm(crop.shape[:2])))
        else:
            side = max(crop.shape[:2])
        pad_height = side - crop.shape[0]
        pad_width = side - crop.shape[1]
        left, top = pad_width // 2, pad_height // 2
        padded = cv.copyMakeBorder(
            crop,
            top,
            pad_height - top,
            left,
            pad_width - left,
            cv.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        bias = np.array([x1 - left, y1 - top], dtype=np.float32)
        return padded, bounds.astype(np.float32), bias

    def _estimate_hand_pose(
        self,
        frame: np.ndarray,
        palm: np.ndarray,
    ) -> tuple[np.ndarray, float, str] | None:
        palm_bounds = palm[:4].reshape(2, 2)
        first_crop = self._crop_and_pad(frame, palm_bounds, before_rotation=True)
        if first_crop is None:
            return None
        crop, palm_crop_bounds, pad_bias = first_crop
        rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)

        palm_landmarks = palm[4:18].reshape(7, 2) - pad_bias
        wrist = palm_landmarks[0]
        middle_base = palm_landmarks[2]
        radians = np.pi / 2 - np.arctan2(
            -(middle_base[1] - wrist[1]),
            middle_base[0] - wrist[0],
        )
        radians -= 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = float(np.rad2deg(radians))

        center = palm_crop_bounds.mean(axis=0) - pad_bias
        rotation = cv.getRotationMatrix2D(tuple(center), angle, 1.0)
        rotated = cv.warpAffine(rgb, rotation, (rgb.shape[1], rgb.shape[0]))
        homogeneous = np.column_stack((palm_landmarks, np.ones(7)))
        rotated_points = np.column_stack(
            (homogeneous @ rotation[0], homogeneous @ rotation[1])
        )
        rotated_bounds = np.array(
            [rotated_points.min(axis=0), rotated_points.max(axis=0)],
            dtype=np.float32,
        )
        second_crop = self._crop_and_pad(rotated, rotated_bounds, before_rotation=False)
        if second_crop is None:
            return None
        hand_crop, rotated_bounds, _ = second_crop

        resized = cv.resize(
            hand_crop,
            (self.HAND_INPUT_SIZE, self.HAND_INPUT_SIZE),
            interpolation=cv.INTER_AREA,
        )
        input_blob = (resized.astype(np.float32) / 255.0)[np.newaxis, ...]
        self._hand_net.setInput(input_blob)
        outputs = self._hand_net.forward(self._hand_net.getUnconnectedOutLayersNames())
        if len(outputs) < 4:
            raise RuntimeError("The hand-pose ONNX model returned an unexpected output layout.")

        confidence = float(np.asarray(outputs[1]).reshape(-1)[0])
        if confidence < HAND_SCORE_THRESHOLD:
            return None
        handedness_score = float(np.asarray(outputs[2]).reshape(-1)[0])
        landmarks = np.asarray(outputs[0])[0].reshape(21, 3).astype(np.float32)
        landmarks = self._restore_landmarks(
            landmarks,
            rotated_bounds,
            angle,
            rotation,
            pad_bias,
        )
        handedness = "Right" if handedness_score >= 0.5 else "Left"
        return landmarks, confidence, handedness

    @classmethod
    def _restore_landmarks(
        cls,
        landmarks: np.ndarray,
        rotated_bounds: np.ndarray,
        angle: float,
        rotation: np.ndarray,
        pad_bias: np.ndarray,
    ) -> np.ndarray:
        restored = landmarks.copy()
        box_size = rotated_bounds[1] - rotated_bounds[0]
        scale = float(np.max(box_size / cls.HAND_INPUT_SIZE))
        restored[:, :2] = (restored[:, :2] - cls.HAND_INPUT_SIZE / 2) * scale
        restored[:, 2] *= scale

        coordinate_rotation = cv.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated_xy = restored[:, :2] @ coordinate_rotation[:, :2]
        inverse_rotation = cv.invertAffineTransform(rotation)
        rotated_center = rotated_bounds.mean(axis=0).reshape(1, 1, 2)
        original_center = cv.transform(rotated_center, inverse_rotation).reshape(2)
        restored[:, :2] = rotated_xy + original_center + pad_bias
        return restored
