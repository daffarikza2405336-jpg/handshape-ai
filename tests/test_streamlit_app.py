from __future__ import annotations

import unittest

import cv2 as cv
import numpy as np

from streamlit_app import analyze_image


class EmptyDetector:
    def detect(self, _frame: np.ndarray) -> list:
        return []


class StreamlitAnalysisTests(unittest.TestCase):
    def test_valid_image_returns_analysis_without_detections(self) -> None:
        ok, encoded = cv.imencode(".jpg", np.zeros((96, 128, 3), dtype=np.uint8))
        self.assertTrue(ok)

        analysis = analyze_image(encoded.tobytes(), EmptyDetector())

        self.assertEqual(analysis.hand_count, 0)
        self.assertIsNone(analysis.result)
        self.assertEqual(analysis.annotated_frame.shape, (96, 128, 3))

    def test_invalid_image_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "JPEG atau PNG"):
            analyze_image(b"not an image", EmptyDetector())

    def test_oversized_input_is_rejected_before_decode(self) -> None:
        with self.assertRaisesRegex(ValueError, "8 MB"):
            analyze_image(b"x" * (8 * 1024 * 1024 + 1), EmptyDetector())


if __name__ == "__main__":
    unittest.main()
