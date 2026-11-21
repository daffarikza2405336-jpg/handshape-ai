from __future__ import annotations

import logging
import os

from hand_detection import HandDetector
from web_server import create_web_server


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    detector = HandDetector()
    server = create_web_server(detector, host, port)
    logging.info("HandShape AI web app listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping HandShape AI web app.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
