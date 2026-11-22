from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ASSETS_DIR, BASE_DIR


LOGGER = logging.getLogger(__name__)
WEB_DIR = BASE_DIR / "web"
app = Flask(__name__)
recognition_service: RecognitionService | None = None


def get_recognition_service() -> RecognitionService:
    global recognition_service
    if recognition_service is None:
        from hand_detection import HandDetector
        from web_server import RecognitionService

        recognition_service = RecognitionService(HandDetector())
    return recognition_service


@app.get("/api/health")
def health() -> Response:
    return jsonify(status="ok", model_loaded=True)


@app.post("/api/recognize")
def recognize() -> tuple[Response, int] | Response:
    if request.content_type != "application/octet-stream":
        return jsonify(error="Send the frame as application/octet-stream."), 415

    session_id = request.headers.get("X-Session-ID", "")
    frame_bytes = request.get_data(cache=False)
    try:
        payload = get_recognition_service().recognize(frame_bytes, session_id)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except (RuntimeError, OSError, ImportError) as error:
        LOGGER.exception("Hand recognition failed.")
        return jsonify(error=f"Recognition failed: {error}"), 500

    return jsonify(payload)


@app.get("/")
@app.get("/index.html")
def index() -> Response:
    return send_file(WEB_DIR / "index.html")


@app.get("/styles.css")
def styles() -> Response:
    return send_file(WEB_DIR / "styles.css", mimetype="text/css")


@app.get("/app.js")
def app_js() -> Response:
    return send_file(WEB_DIR / "app.js", mimetype="text/javascript")


@app.get("/assets/<path:filename>")
def assets(filename: str) -> Response:
    web_asset = WEB_DIR / "assets" / filename
    if web_asset.is_file():
        return send_file(web_asset)

    root_asset = ASSETS_DIR / filename
    if root_asset.is_file():
        return send_file(root_asset)

    return jsonify(error="Not found."), 404
