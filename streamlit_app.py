from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image

from config import HAND_CONNECTIONS
from gesture_recognition import GestureRecognizer, RecognitionResult
from hand_detection import HandDetector


BASE_DIR = Path(__file__).resolve().parent
WEB_ASSETS_DIR = BASE_DIR / "web" / "assets"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_WIDTH = 1280
MAX_IMAGE_HEIGHT = 960
MAX_SOURCE_DIMENSION = 4096
MODEL_LOCK = Lock()


@dataclass(frozen=True)
class ImageAnalysis:
    annotated_frame: np.ndarray
    hand_count: int
    result: RecognitionResult | None
    processing_ms: float


def analyze_image(image_bytes: bytes, detector: HandDetector) -> ImageAnalysis:
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Gambar harus berukuran antara 1 byte dan 8 MB.")

    try:
        image = Image.open(BytesIO(image_bytes))
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("File yang dipilih bukan gambar JPEG atau PNG yang valid.") from error

    with image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("Gunakan gambar JPEG atau PNG.")
        width, height = image.size
        if width > MAX_SOURCE_DIMENSION or height > MAX_SOURCE_DIMENSION:
            raise ValueError("Dimensi gambar tidak boleh melebihi 4096 × 4096 piksel.")
        image.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        frame = cv.cvtColor(np.asarray(image.convert("RGB")), cv.COLOR_RGB2BGR)

    started = perf_counter()
    with MODEL_LOCK:
        hands = detector.detect(frame)
        result = GestureRecognizer().recognize(hands)
    elapsed_ms = (perf_counter() - started) * 1000
    annotated = frame.copy()

    for hand in hands:
        points = np.rint(hand.landmarks[:, :2] * np.array([frame.shape[1], frame.shape[0]])).astype(int)
        for start, end in HAND_CONNECTIONS:
            cv.line(
                annotated,
                tuple(points[start]),
                tuple(points[end]),
                (185, 230, 197),
                2,
                cv.LINE_AA,
            )
        for point in points:
            cv.circle(annotated, tuple(point), 3, (236, 248, 239), -1, cv.LINE_AA)

    if result and len(result.points) >= 3:
        polygon = np.rint(
            np.asarray(result.points) * np.array([frame.shape[1], frame.shape[0]])
        ).astype(np.int32)
        cv.polylines(annotated, [polygon], True, (185, 230, 197), 3, cv.LINE_AA)

    return ImageAnalysis(
        annotated_frame=annotated,
        hand_count=len(hands),
        result=result,
        processing_ms=elapsed_ms,
    )


@st.cache_resource
def load_detector() -> HandDetector:
    return HandDetector()


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp {
          background:
            radial-gradient(ellipse at 82% 4%, rgba(49, 90, 67, .22), transparent 34rem),
            #070809;
          color: #f4f4f2;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 2rem; }
        .eyebrow { color: #b9e6c5; font: 10px monospace; letter-spacing: .16em; text-transform: uppercase; }
        .hero { padding: 3.2rem 0 2.6rem; border-bottom: 1px solid rgba(255,255,255,.12); }
        .hero h1 { font-size: clamp(2.6rem, 7vw, 5.7rem); line-height: .98; letter-spacing: -.065em; margin: .8rem 0 1rem; }
        .hero h1 span { color: #a7b0aa; }
        .hero-copy { max-width: 630px; color: #a4aaa7; font-size: 1.05rem; }
        .section-title { margin: 3.3rem 0 1.1rem; }
        .section-title h2 { font-size: clamp(1.7rem, 4vw, 2.7rem); letter-spacing: -.05em; margin: .35rem 0; }
        .gesture-card, .team-card, .result-card {
          border: 1px solid rgba(255,255,255,.13); border-radius: 16px;
          background: linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.015));
          padding: 1.15rem; min-height: 145px;
        }
        .gesture-card small { color: #b9e6c5; font: 9px monospace; letter-spacing: .12em; }
        .gesture-card h3 { margin: .8rem 0 .4rem; font-size: 1.2rem; }
        .gesture-card p, .muted { color: #9da4a0; font-size: .92rem; }
        .result-value { color: #b9e6c5; font: 1.8rem monospace; }
        .team-card { min-height: 0; padding: 0 0 1.15rem; overflow: hidden; }
        .team-card h3, .team-card p { margin-left: 1.15rem; margin-right: 1.15rem; }
        .team-card h3 { margin-top: .9rem; margin-bottom: .35rem; font-size: 1.2rem; }
        .team-card p { color: #a4aaa7; font-size: .85rem; }
        div[data-testid="stCameraInput"] { border: 1px solid rgba(185,230,197,.25); border-radius: 14px; padding: 1rem; }
        .privacy-note { color: #838b86; font-size: .8rem; }
        @media (max-width: 640px) {
          [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
          .hero { padding-top: 2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_team() -> None:
    st.markdown(
        """
        <div class="section-title" id="about">
          <div class="eyebrow">03 / THE PEOPLE BEHIND THE PROJECT</div>
          <h2>Dua developer. Satu ide.</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(2, gap="large")
    team = (
        ("daffa.jpeg", "Daffa Rikza Mansuri", "240533610287"),
        ("zahra.jpeg", "Azzahra Kusuma", "240533600041"),
    )
    for column, (photo, name, student_id) in zip(columns, team, strict=True):
        with column:
            st.image(str(WEB_ASSETS_DIR / photo), use_container_width=True)
            st.markdown(
                f"""
                <div class="team-card">
                  <p class="eyebrow">DEVELOPER</p>
                  <h3>{name}</h3>
                  <p>NIM &nbsp; {student_id}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="HandShape AI — Gestures in motion",
        page_icon="✋",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _render_styles()

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">COMPUTER VISION, IN MOTION</div>
          <h1>Gestur tangan.<br><span>Bentuk baru.</span></h1>
          <p class="hero-copy">
            Ambil foto gestur tangan dan lihat HandShape AI mengenali bentuknya.
            Gambar diproses langsung dan tidak disimpan.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-title" id="gestures">
          <div class="eyebrow">01 / GESTURE LIBRARY</div>
          <h2>Satu tangan. Empat kemungkinan.</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gestures = (
        ("GESTURE 01", "Circle", "Satukan ibu jari dan telunjuk."),
        ("GESTURE 02", "V Shape", "Angkat telunjuk dan jari tengah."),
        ("GESTURE 03", "Triangle", "Bentuk tiga sudut dengan kedua tangan."),
        ("GESTURE 04", "Square", "Susun empat sudut dengan dua tangan."),
    )
    for column, (number, name, description) in zip(
        st.columns(4, gap="small"), gestures, strict=True
    ):
        with column:
            st.markdown(
                f"""
                <div class="gesture-card">
                  <small>{number}</small><h3>{name}</h3><p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-title" id="studio">
          <div class="eyebrow">02 / INTERACTIVE STUDIO</div>
          <h2>Ambil gambar. Lihat hasilnya.</h2>
          <p class="muted">Gunakan kamera perangkat atau unggah gambar tangan yang jelas.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    camera_tab, upload_tab = st.tabs(["Kamera", "Unggah gambar"])
    with camera_tab:
        camera_image = st.camera_input("Ambil foto gestur tangan", key="camera_image")
    with upload_tab:
        uploaded_image = st.file_uploader(
            "Pilih foto gestur tangan",
            type=["jpg", "jpeg", "png"],
            key="uploaded_image",
        )

    selected_image = camera_image or uploaded_image
    if selected_image is not None:
        try:
            with st.spinner("Memproses gambar dengan model HandShape AI..."):
                analysis = analyze_image(selected_image.getvalue(), load_detector())
        except ValueError as error:
            st.error(str(error))
        else:
            image_column, result_column = st.columns([1.2, 0.8], gap="large")
            with image_column:
                st.image(
                    analysis.annotated_frame,
                    channels="BGR",
                    caption="Landmark tangan yang terdeteksi",
                    use_container_width=True,
                )
            with result_column:
                st.markdown('<div class="eyebrow">LIVE ANALYSIS</div>', unsafe_allow_html=True)
                gesture = analysis.result.label if analysis.result else "—"
                shape = analysis.result.shape_name if analysis.result else "Gestur belum dikenali"
                st.markdown(
                    f"""
                    <div class="result-card">
                      <p class="eyebrow">GESTUR TERDETEKSI</p>
                      <div class="result-value">{gesture}</div>
                      <p class="muted">{shape}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                metric_columns = st.columns(2)
                metric_columns[0].metric("Tangan", analysis.hand_count)
                metric_columns[1].metric("Waktu proses", f"{analysis.processing_ms:.0f} ms")
                if analysis.hand_count == 0:
                    st.info("Tidak ada tangan terdeteksi. Coba foto dengan pencahayaan lebih baik.")
    else:
        st.info("Aktifkan kamera atau unggah gambar untuk mulai mengenali gestur.")

    st.markdown(
        '<p class="privacy-note">Foto hanya diproses untuk pengenalan gestur dan tidak disimpan.</p>',
        unsafe_allow_html=True,
    )
    _render_team()
    st.markdown(
        '<p class="privacy-note">HANDSHAPE AI &nbsp; / &nbsp; REAL-TIME GESTURE RECOGNITION</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
