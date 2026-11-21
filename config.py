from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
MODELS_DIR = ASSETS_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "screenshots"

PALM_MODEL_PATH = MODELS_DIR / "palm_detection.onnx"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarks.onnx"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MAX_HANDS = 2
PALM_SCORE_THRESHOLD = 0.65
HAND_SCORE_THRESHOLD = 0.7

APP_TITLE = "HandShape AI"
PROJECT_TITLE = (
    "Implementasi MediaPipe dan OpenCV untuk Pengenalan Bentuk Geometri "
    "Berdasarkan Gestur Jari Tangan Secara Real-Time Menggunakan Python."
)

COLOR_BACKGROUND = "#070809"
COLOR_PANEL = "#101315"
COLOR_PANEL_LIGHT = "#1B2022"
COLOR_PANEL_HOVER = "#19201D"
COLOR_PANEL_BORDER = "#3A4444"
COLOR_TEXT = "#F4F4F2"
COLOR_MUTED = "#92969A"
COLOR_ACCENT = "#B9E6C5"
COLOR_SUCCESS = "#8ED8A2"
COLOR_WARNING = "#E8C77A"
COLOR_DANGER = "#F08B83"

# Landmark pairs follow the 21-point hand landmark order used by the models.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
