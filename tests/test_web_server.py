from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2 as cv
import numpy as np

from web_server import create_web_server


class EmptyDetector:
    def detect(self, _frame: np.ndarray) -> list:
        return []


class WebServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_web_server(EmptyDetector(), host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_homepage_and_health_endpoint_are_served(self) -> None:
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn("HandShape AI", page)
            self.assertIn("Daffa Rikza Mansuri", page)
            self.assertIn("240533610287", page)
            self.assertIn("Azzahra Brilian Kusuma", page)
            self.assertIn("240533600041", page)
            self.assertEqual(page.count('<p class="team-label">DEVELOPER</p>'), 2)

        with urlopen(f"{self.base_url}/api/health") as response:
            self.assertEqual(json.load(response), {"status": "ok", "model_loaded": True})

    def test_team_photos_are_served_as_jpeg(self) -> None:
        for photo in ("daffa.jpeg", "zahra.jpeg"):
            with urlopen(f"{self.base_url}/assets/{photo}") as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                self.assertGreater(int(response.headers["Content-Length"]), 0)

    def test_animation_module_is_served(self) -> None:
        with urlopen(f"{self.base_url}/animations.js") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), "text/javascript")
            self.assertIn(b"ScrollTrigger", response.read())

    def test_recognition_returns_landmark_ready_payload(self) -> None:
        ok, encoded = cv.imencode(".jpg", np.zeros((96, 128, 3), dtype=np.uint8))
        self.assertTrue(ok)
        request = Request(
            f"{self.base_url}/api/recognize",
            data=encoded.tobytes(),
            headers={
                "Content-Type": "application/octet-stream",
                "X-Session-ID": "testsession12345678",
            },
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.load(response)

        self.assertEqual(payload["hand_count"], 0)
        self.assertEqual(payload["hands"], [])
        self.assertIsNone(payload["result"])
        self.assertEqual(payload["total_detections"], 0)
        self.assertGreaterEqual(payload["processing_ms"], 0)

    def test_invalid_frame_is_rejected(self) -> None:
        request = Request(
            f"{self.base_url}/api/recognize",
            data=b"not an image",
            headers={
                "Content-Type": "application/octet-stream",
                "X-Session-ID": "testsession12345678",
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)
        self.assertEqual(raised.exception.code, 400)
        response = raised.exception
        self.assertIn("JPEG", json.load(response)["error"])
        response.close()

    def test_oversized_jpeg_dimensions_are_rejected_before_decode(self) -> None:
        oversized_header = b"\xff\xd8\xff\xc0\x00\x07\x08\x03\xe8\x05\x01"
        request = Request(
            f"{self.base_url}/api/recognize",
            data=oversized_header,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Session-ID": "testsession12345678",
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)
        self.assertEqual(raised.exception.code, 400)
        response = raised.exception
        self.assertIn("dimensions", json.load(response)["error"])
        response.close()

    def test_unknown_route_returns_not_found(self) -> None:
        with self.assertRaises(HTTPError) as raised:
            urlopen(f"{self.base_url}/not-a-route")
        self.assertEqual(raised.exception.code, 404)
        raised.exception.close()


if __name__ == "__main__":
    unittest.main()
