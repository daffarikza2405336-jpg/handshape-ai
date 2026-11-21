from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import cv2 as cv
import numpy as np

from config import ASSETS_DIR, BASE_DIR
from gesture_recognition import GestureRecognizer, GestureSmoother
from hand_detection import HandDetector


LOGGER = logging.getLogger(__name__)
WEB_DIR = BASE_DIR / "web"
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_FRAME_WIDTH = 1280
MAX_FRAME_HEIGHT = 960
MAX_SESSIONS = 128
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,80}$")
JPEG_START_OF_FRAME_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}
STATIC_FILES = {
    "/": (WEB_DIR / "index.html", "text/html; charset=utf-8"),
    "/index.html": (WEB_DIR / "index.html", "text/html; charset=utf-8"),
    "/styles.css": (WEB_DIR / "styles.css", "text/css; charset=utf-8"),
    "/app.js": (WEB_DIR / "app.js", "text/javascript; charset=utf-8"),
    "/assets/logo.png": (ASSETS_DIR / "logo.png", "image/png"),
    "/assets/daffa.jpeg": (WEB_DIR / "assets" / "daffa.jpeg", "image/jpeg"),
    "/assets/zahra.jpeg": (WEB_DIR / "assets" / "zahra.jpeg", "image/jpeg"),
}


class RecognitionService:
    def __init__(self, detector: HandDetector) -> None:
        self._detector = detector
        self._recognizer = GestureRecognizer()
        self._sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def recognize(self, frame_bytes: bytes, session_id: str) -> dict[str, Any]:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("Invalid browser session ID.")
        if not frame_bytes or len(frame_bytes) > MAX_FRAME_BYTES:
            raise ValueError("Frame must be a non-empty image smaller than 2 MB.")

        frame_width, frame_height = self._jpeg_dimensions(frame_bytes)
        if frame_width > MAX_FRAME_WIDTH or frame_height > MAX_FRAME_HEIGHT:
            raise ValueError(
                f"Frame dimensions must not exceed {MAX_FRAME_WIDTH}×{MAX_FRAME_HEIGHT}."
            )
        encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv.imdecode(encoded, cv.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Frame is not a valid JPEG image.")

        started = perf_counter()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = {
                    "smoother": GestureSmoother(),
                    "last_label": None,
                    "total_detections": 0,
                }
            self._sessions[session_id] = session
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)

            hands = self._detector.detect(frame)
            result = session["smoother"].update(self._recognizer.recognize(hands))
            label = result.label if result else None
            if label and label != session["last_label"]:
                session["total_detections"] += 1
            session["last_label"] = label
            total_detections = session["total_detections"]

        elapsed_ms = (perf_counter() - started) * 1000
        result_payload = None
        if result is not None:
            result_payload = {
                "label": result.label,
                "shape_name": result.shape_name,
                "points": result.points,
            }
        return {
            "hand_count": len(hands),
            "hands": [
                {
                    "handedness": hand.handedness,
                    "confidence": round(float(hand.confidence), 3),
                    "landmarks": hand.landmarks[:, :2].round(5).tolist(),
                }
                for hand in hands
            ],
            "result": result_payload,
            "processing_ms": round(elapsed_ms, 2),
            "total_detections": total_detections,
        }

    @staticmethod
    def _jpeg_dimensions(frame_bytes: bytes) -> tuple[int, int]:
        if not frame_bytes.startswith(b"\xff\xd8"):
            raise ValueError("Frame must be a JPEG image.")

        offset = 2
        while offset < len(frame_bytes):
            if frame_bytes[offset] != 0xFF:
                offset += 1
                continue
            while offset < len(frame_bytes) and frame_bytes[offset] == 0xFF:
                offset += 1
            if offset >= len(frame_bytes):
                break

            marker = frame_bytes[offset]
            offset += 1
            if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if marker == 0xDA:
                break
            if offset + 2 > len(frame_bytes):
                break
            segment_length = int.from_bytes(frame_bytes[offset : offset + 2], "big")
            if segment_length < 2 or offset + segment_length > len(frame_bytes):
                break
            if marker in JPEG_START_OF_FRAME_MARKERS:
                if segment_length < 7:
                    break
                height = int.from_bytes(frame_bytes[offset + 3 : offset + 5], "big")
                width = int.from_bytes(frame_bytes[offset + 5 : offset + 7], "big")
                if width == 0 or height == 0:
                    break
                return width, height
            offset += segment_length
        raise ValueError("Frame is not a valid JPEG image.")


def create_web_server(
    detector: HandDetector,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    service = RecognitionService(detector)

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == "/api/health":
                self._send_json(200, {"status": "ok", "model_loaded": True})
                return

            static_file = STATIC_FILES.get(route)
            if static_file is None:
                self._send_json(404, {"error": "Not found."})
                return
            path, content_type = static_file
            try:
                contents = path.read_bytes()
            except OSError:
                LOGGER.exception("Could not read web asset %s", path)
                self._send_json(500, {"error": "A web asset could not be loaded."})
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(contents)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(contents)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/recognize":
                self._send_json(404, {"error": "Not found."})
                return
            if self.headers.get_content_type() != "application/octet-stream":
                self._send_json(415, {"error": "Send the frame as application/octet-stream."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length header."})
                return
            if content_length <= 0 or content_length > MAX_FRAME_BYTES:
                self._send_json(413, {"error": "Frame must be between 1 byte and 2 MB."})
                return

            session_id = self.headers.get("X-Session-ID", "")
            try:
                frame_bytes = self.rfile.read(content_length)
                payload = service.recognize(frame_bytes, session_id)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            except (cv.error, RuntimeError) as error:
                LOGGER.exception("Hand recognition failed.")
                self._send_json(500, {"error": f"Recognition failed: {error}"})
                return
            self._send_json(200, payload)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format_string % args)

    class ThreadedServer(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    return ThreadedServer((host, port), RequestHandler)
