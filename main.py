from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2 as cv
import customtkinter as ctk
import numpy as np
from tkinter import messagebox

from config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    HAND_CONNECTIONS,
    RESULTS_DIR,
)
from dashboard import Dashboard
from gesture_recognition import GestureRecognizer, GestureSmoother, RecognitionResult
from hand_detection import DetectedHand, HandDetector


@dataclass(frozen=True)
class CameraSnapshot:
    frame: np.ndarray
    hand_count: int
    result: RecognitionResult | None
    fps: float
    total_detections: int


class CameraWorker:
    def __init__(self, detector: HandDetector, camera_index: int = 0) -> None:
        self._detector = detector
        self._camera_index = camera_index
        self._recognizer = GestureRecognizer()
        self._smoother = GestureSmoother()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: CameraSnapshot | None = None
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name="handshape-camera", daemon=True)
        self._total_detections = 0
        self._last_label: str | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def is_stopping(self) -> bool:
        return self._stop_event.is_set()

    def latest(self) -> CameraSnapshot | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        capture = None
        try:
            if sys.platform == "darwin":
                capture = cv.VideoCapture(self._camera_index, cv.CAP_AVFOUNDATION)
            else:
                capture = cv.VideoCapture(self._camera_index)
            if not capture.isOpened():
                self.error = (
                    "Webcam could not be opened. Check the camera connection and "
                    "macOS camera permission, then try again."
                )
                return

            capture.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            capture.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
            previous_label: str | None = None
            smoothed_fps = 0.0

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.error = "The webcam stopped returning frames."
                    break

                frame = cv.flip(frame, 1)
                started = time.perf_counter()
                hands = self._detector.detect(frame)
                raw_result = self._recognizer.recognize(hands)
                result = self._smoother.update(raw_result)
                elapsed = max(time.perf_counter() - started, 1e-6)
                instantaneous_fps = 1.0 / elapsed
                smoothed_fps = (
                    instantaneous_fps
                    if smoothed_fps == 0
                    else 0.8 * smoothed_fps + 0.2 * instantaneous_fps
                )

                current_label = result.label if result else None
                if current_label is not None and current_label != previous_label:
                    self._total_detections += 1
                previous_label = current_label

                annotated = self._draw_overlay(frame, hands, result)
                snapshot = CameraSnapshot(
                    frame=annotated,
                    hand_count=len(hands),
                    result=result,
                    fps=smoothed_fps,
                    total_detections=self._total_detections,
                )
                with self._lock:
                    self._latest = snapshot
        except (cv.error, OSError, RuntimeError, ValueError) as error:
            self.error = str(error)
        finally:
            if capture is not None:
                capture.release()

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        hands: list[DetectedHand],
        result: RecognitionResult | None,
    ) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        for hand in hands:
            points = hand.landmarks
            for start, end in HAND_CONNECTIONS:
                start_point = (int(points[start, 0] * width), int(points[start, 1] * height))
                end_point = (int(points[end, 0] * width), int(points[end, 1] * height))
                cv.line(output, start_point, end_point, (190, 190, 190), 2, cv.LINE_AA)
            for point in points:
                location = (int(point[0] * width), int(point[1] * height))
                cv.circle(output, location, 4, (245, 245, 245), -1, cv.LINE_AA)
                cv.circle(output, location, 2, (100, 100, 100), -1, cv.LINE_AA)

        if result is not None:
            cv.rectangle(output, (12, 12), (216, 52), (15, 15, 15), -1)
            cv.putText(
                output,
                result.label,
                (24, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.72,
                (235, 235, 235),
                2,
                cv.LINE_AA,
            )
        return output


class HandShapeApplication:
    def __init__(self, detector: HandDetector) -> None:
        self._detector = detector
        self._worker: CameraWorker | None = None
        self._last_snapshot: CameraSnapshot | None = None
        self._closing = False
        self.dashboard = Dashboard(
            on_start=self.start_camera,
            on_stop=self.stop_camera,
            on_capture=self.capture_result,
        )
        self.dashboard.refresh_history(self._list_screenshots())
        self.dashboard.protocol("WM_DELETE_WINDOW", self.close)
        self.dashboard.after(30, self._poll_camera)

    def run(self) -> None:
        self.dashboard.mainloop()

    def start_camera(self) -> None:
        if self._worker is not None and self._worker.is_alive:
            return
        self._last_snapshot = None
        self._worker = CameraWorker(self._detector)
        self._worker.start()
        self.dashboard.set_camera_status("Connecting")
        self.dashboard.set_detection_status("Waiting")
        self.dashboard.show_camera_message("Connecting to webcam…")
        self.dashboard.set_controls_enabled(start=False, stop=True)

    def stop_camera(self) -> None:
        worker = self._worker
        if worker is None:
            self._reset_camera_ui()
            return

        worker.stop()
        worker.join(timeout=0.25)
        if worker.is_alive:
            self.dashboard.set_camera_status("Stopping")
            self.dashboard.set_detection_status("Stopping")
            self.dashboard.set_controls_enabled(start=False, stop=False)
            return

        self._worker = None
        self._reset_camera_ui()

    def _reset_camera_ui(self) -> None:
        self._last_snapshot = None
        self.dashboard.set_camera_status("Stopped")
        self.dashboard.set_detection_status("Idle")
        self.dashboard.show_camera_message("Camera is stopped\n\nPress Start Camera to begin")
        self.dashboard.set_controls_enabled(start=True, stop=False)

    def capture_result(self) -> None:
        if self._last_snapshot is None:
            self.dashboard.show_notice(
                "Camera not ready",
                "Start the camera and wait for a frame before capturing.",
                error=True,
            )
            return

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = RESULTS_DIR / f"handshape_{timestamp}.png"
        if not cv.imwrite(str(destination), self._last_snapshot.frame):
            self.dashboard.show_notice(
                "Capture failed",
                f"Could not write screenshot to:\n{destination}",
                error=True,
            )
            return

        self.dashboard.refresh_history(self._list_screenshots())
        self.dashboard.show_notice(
            "Capture saved",
            f"Screenshot saved to:\n{destination}",
        )

    def _poll_camera(self) -> None:
        if self._closing:
            return

        worker = self._worker
        if worker is not None:
            if worker.is_stopping and not worker.is_alive:
                self._worker = None
                self._reset_camera_ui()
            elif worker.is_stopping:
                self.dashboard.set_camera_status("Stopping")
                self.dashboard.set_controls_enabled(start=False, stop=False)
            elif worker.error and not worker.is_alive:
                self.dashboard.set_camera_status("Unavailable", error=True)
                self.dashboard.set_detection_status("Camera error", warning=True)
                self.dashboard.show_camera_message(worker.error)
                self.dashboard.set_controls_enabled(start=True, stop=False)
                self._worker = None
            elif (snapshot := worker.latest()) is not None:
                self._last_snapshot = snapshot
                frame_rgb = cv.cvtColor(snapshot.frame, cv.COLOR_BGR2RGB)
                self.dashboard.show_camera_frame(
                    frame_rgb,
                    snapshot.hand_count,
                    snapshot.result,
                    snapshot.fps,
                    snapshot.total_detections,
                )
                self.dashboard.set_camera_status("Running")
                self.dashboard.set_controls_enabled(start=False, stop=True)
            elif not worker.is_alive:
                self.dashboard.set_camera_status("Stopped")
                self.dashboard.set_controls_enabled(start=True, stop=False)
                self._worker = None

        self.dashboard.after(30, self._poll_camera)

    @staticmethod
    def _list_screenshots() -> list[Path]:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return sorted(RESULTS_DIR.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)

    def close(self) -> None:
        self._closing = True
        if self._worker is not None:
            self._worker.stop()
            self._worker.join(timeout=1.5)
            self._worker = None
        self.dashboard.destroy()


def main() -> None:
    try:
        detector = HandDetector()
    except (FileNotFoundError, cv.error, RuntimeError, ValueError) as error:
        root = ctk.CTk()
        root.withdraw()
        messagebox.showerror("HandShape AI could not start", str(error))
        root.destroy()
        raise SystemExit(1) from error

    application = HandShapeApplication(detector)
    application.run()


if __name__ == "__main__":
    main()
