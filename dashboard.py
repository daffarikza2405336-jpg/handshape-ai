from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk
from tkinter import Canvas, messagebox

from config import (
    APP_TITLE,
    COLOR_ACCENT,
    COLOR_BACKGROUND,
    COLOR_DANGER,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PANEL_BORDER,
    COLOR_PANEL_HOVER,
    COLOR_PANEL_LIGHT,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_WARNING,
    LOGO_PATH,
    PROJECT_TITLE,
)
from geometry import draw_geometry
from gesture_recognition import RecognitionResult


FONT_DISPLAY = "Avenir Next"
FONT_MONO = "Menlo"
LINE_COLOR = "#30383A"


class Dashboard(ctk.CTk):
    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_capture: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_capture = on_capture
        self._camera_image: ctk.CTkImage | None = None
        self._current_result: RecognitionResult | None = None
        self._hero_phase = 0.0
        self._hero_animation_id: str | None = None
        self._hero_ribbons: list[tuple[int, float, float, float, float, float]] = []

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.title(APP_TITLE)
        self.geometry("1240x900")
        self.minsize(880, 700)
        self.configure(fg_color=COLOR_BACKGROUND)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_pages()
        self.show_page("Dashboard")
        self.set_camera_status("Stopped")
        self.set_controls_enabled(start=True, stop=False)

    def _build_header(self) -> None:
        header = ctk.CTkFrame(
            self,
            height=68,
            corner_radius=0,
            fg_color=COLOR_PANEL,
            border_width=1,
            border_color=COLOR_PANEL_BORDER,
        )
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="w", padx=(28, 12))
        logo = ImageOps.grayscale(Image.open(LOGO_PATH).convert("RGBA")).convert("RGBA")
        self._logo_image = ctk.CTkImage(light_image=logo, dark_image=logo, size=(35, 35))
        ctk.CTkLabel(brand, image=self._logo_image, text="").pack(side="left")
        ctk.CTkLabel(
            brand,
            text="HANDSHAPE",
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=15, weight="bold"),
        ).pack(side="left", padx=(9, 3))
        ctk.CTkLabel(
            brand,
            text="AI",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=11),
        ).pack(side="left", pady=(3, 0))

        nav = ctk.CTkFrame(header, fg_color="transparent")
        nav.grid(row=0, column=1)
        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        for name, label in (
            ("Dashboard", "HOME"),
            ("Live Detection", "LIVE DEMO"),
            ("History", "ARCHIVE"),
            ("About Project", "PROJECT"),
        ):
            button = ctk.CTkButton(
                nav,
                text=label,
                width=76 if name != "Live Detection" else 96,
                height=34,
                corner_radius=18,
                fg_color="transparent",
                hover_color=COLOR_PANEL_LIGHT,
                text_color=COLOR_MUTED,
                font=ctk.CTkFont(family=FONT_MONO, size=9),
                command=lambda page=name: self.show_page(page),
            )
            button.pack(side="left", padx=4)
            self._nav_buttons[name] = button

        status_pill = ctk.CTkFrame(
            header,
            fg_color=COLOR_PANEL_LIGHT,
            border_width=1,
            border_color=COLOR_PANEL_BORDER,
            corner_radius=18,
        )
        status_pill.grid(row=0, column=2, sticky="e", padx=24, pady=14)
        self._header_status = ctk.CTkLabel(
            status_pill,
            text="●  SYSTEM STANDBY",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=9),
        )
        self._header_status.pack(padx=12, pady=7)

    def _build_pages(self) -> None:
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.grid(row=1, column=0, sticky="nsew", padx=26, pady=(0, 0))
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        self._pages: dict[str, ctk.CTkScrollableFrame] = {}
        self._build_landing_page(container)
        self._build_history_page(container)
        self._build_about_page(container)

    def _build_landing_page(self, parent: ctk.CTkFrame) -> None:
        page = ctk.CTkScrollableFrame(
            parent,
            fg_color=COLOR_BACKGROUND,
            corner_radius=0,
            scrollbar_button_color="#34383B",
            scrollbar_button_hover_color="#555B5F",
        )
        page.grid_columnconfigure(0, weight=1)
        self._pages["Dashboard"] = page
        self._pages["Live Detection"] = page

        self._landing_page = page
        self._hero_canvas = Canvas(
            page,
            height=500,
            bg=COLOR_BACKGROUND,
            highlightthickness=0,
            bd=0,
        )
        self._hero_canvas.grid(row=0, column=0, sticky="ew", pady=(12, 22))
        self._hero_canvas.bind("<Configure>", self._render_hero)

        intro = ctk.CTkFrame(page, fg_color="transparent")
        intro.grid(row=1, column=0, sticky="ew", pady=(24, 35))
        intro.grid_columnconfigure((0, 1), weight=1, uniform="intro")

        description = ctk.CTkFrame(intro, fg_color="transparent")
        description.grid(row=0, column=0, sticky="nsew", padx=(12, 32))
        ctk.CTkLabel(
            description,
            text="TENTANG PROYEK  /  01",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=9),
        ).pack(anchor="w", pady=(4, 15))
        ctk.CTkLabel(
            description,
            text="Dari gestur\nmenjadi geometri.",
            text_color=COLOR_TEXT,
            justify="left",
            font=ctk.CTkFont(family=FONT_DISPLAY, size=29, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            description,
            text=(
                "HandShape AI membaca 21 titik landmark tangan melalui webcam dan "
                "menerjemahkan gerakan menjadi bentuk geometri secara real-time. "
                "Pengenalan berbasis aturan memeriksa posisi jari dan susunan titik "
                "untuk mengenali Circle, V Shape, Triangle, dan Square."
            ),
            text_color=COLOR_MUTED,
            justify="left",
            wraplength=480,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=12),
        ).pack(anchor="w", pady=(12, 0))

        creators = ctk.CTkFrame(
            intro,
            fg_color=COLOR_PANEL,
            corner_radius=20,
            border_width=1,
            border_color=COLOR_PANEL_BORDER,
        )
        creators.grid(row=0, column=1, sticky="nsew", padx=(8, 12))
        ctk.CTkLabel(
            creators,
            text="ORANG DI BALIK PROYEK",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=9),
        ).pack(anchor="w", padx=20, pady=(17, 10))
        self._creator_name(creators, "KREATOR", "Azzahra Brilian")
        self._creator_name(creators, "DEVELOPER", "Daffa Rikza Mansuri")
        ctk.CTkLabel(
            creators,
            text="COMPUTER VISION  ·  OPENCV DNN  ·  21 LANDMARKS",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=7),
        ).pack(anchor="w", padx=20, pady=(4, 16))
        self._bind_card_interaction(creators)

        features = ctk.CTkFrame(page, fg_color="transparent")
        features.grid(row=2, column=0, sticky="ew", pady=(4, 47))
        features.grid_columnconfigure(0, weight=1)
        feature_heading = ctk.CTkFrame(features, fg_color="transparent")
        feature_heading.grid(row=0, column=0, sticky="ew", padx=12, pady=(0, 16))
        feature_heading.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            feature_heading,
            text="Dari gestur menjadi geometri.",
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=21),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            feature_heading,
            text="GESTURE LIBRARY  /  04",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        ).grid(row=0, column=1, sticky="e")

        feature_row = ctk.CTkFrame(features, fg_color="transparent")
        feature_row.grid(row=1, column=0, sticky="ew")
        feature_row.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="features")
        for column, details in enumerate(
            (
                ("01", "Circle", "Satukan ibu jari dan telunjuk membentuk lingkaran.", "circle"),
                ("02", "V Shape", "Angkat dua jari untuk membentuk tanda V.", "v"),
                ("03", "Triangle", "Bentuk tiga sudut dengan kedua tangan.", "triangle"),
                ("04", "Square", "Susun empat titik sudut menjadi persegi.", "square"),
            )
        ):
            self._feature_card(feature_row, column, *details)

        process = ctk.CTkFrame(page, fg_color="transparent")
        process.grid(row=3, column=0, sticky="ew", pady=(0, 48))
        process.grid_columnconfigure(0, weight=1)
        process_heading = ctk.CTkFrame(process, fg_color="transparent")
        process_heading.grid(row=0, column=0, sticky="ew", padx=12, pady=(0, 14))
        process_heading.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            process_heading,
            text="Dari kamera ke bentuk.",
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=21),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            process_heading,
            text="REAL-TIME PIPELINE",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        ).grid(row=0, column=1, sticky="e")

        process_row = ctk.CTkFrame(process, fg_color="transparent")
        process_row.grid(row=1, column=0, sticky="ew")
        process_row.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="process")
        for column, (number, title, detail) in enumerate(
            (
                ("01", "Capture", "Webcam menangkap gerakan tangan."),
                ("02", "Track", "Model melacak 21 landmark."),
                ("03", "Recognize", "Aturan geometri membaca gestur."),
                ("04", "Visualize", "Bentuk tampil secara langsung."),
            )
        ):
            step = ctk.CTkFrame(
                process_row,
                fg_color=COLOR_PANEL,
                corner_radius=18,
                border_width=1,
                border_color=COLOR_PANEL_BORDER,
            )
            step.grid(row=0, column=column, sticky="nsew", padx=5)
            ctk.CTkLabel(
                step,
                text=number,
                text_color=COLOR_ACCENT,
                font=ctk.CTkFont(family=FONT_MONO, size=9),
            ).pack(anchor="w", padx=15, pady=(14, 8))
            ctk.CTkLabel(
                step,
                text=title,
                text_color=COLOR_TEXT,
                font=ctk.CTkFont(family=FONT_DISPLAY, size=14, weight="bold"),
            ).pack(anchor="w", padx=15)
            ctk.CTkLabel(
                step,
                text=detail,
                text_color=COLOR_MUTED,
                justify="left",
                wraplength=155,
                font=ctk.CTkFont(family=FONT_DISPLAY, size=10),
            ).pack(anchor="w", padx=15, pady=(6, 14))
            self._bind_card_interaction(step, self.scroll_to_live)

        live_heading = ctk.CTkFrame(page, fg_color="transparent")
        live_heading.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 17))
        live_heading.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            live_heading,
            text="02  /  LIVE SESSION",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=9),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            live_heading,
            text="Arahkan tanganmu. Lihat bentuknya.",
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=22),
        ).grid(row=1, column=0, sticky="w", pady=(7, 0))

        camera_frame = ctk.CTkFrame(
            page,
            fg_color=COLOR_PANEL,
            corner_radius=20,
            border_width=1,
            border_color=COLOR_PANEL_BORDER,
        )
        self._live_section = camera_frame
        camera_frame.grid(row=5, column=0, sticky="ew", pady=(0, 27))
        camera_frame.grid_columnconfigure(0, weight=1)

        camera_top = ctk.CTkFrame(camera_frame, fg_color="transparent", height=43)
        camera_top.grid(row=0, column=0, sticky="ew", padx=17, pady=(7, 0))
        camera_top.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            camera_top,
            text="CAM 01    /    PALM & LANDMARK FEED",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=9),
        ).grid(row=0, column=0, sticky="w")
        self._camera_status_badge = ctk.CTkLabel(
            camera_top,
            text="●  STANDBY",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=9),
        )
        self._camera_status_badge.grid(row=0, column=1, sticky="e")

        preview_shell = ctk.CTkFrame(camera_frame, fg_color="#050606", corner_radius=12)
        preview_shell.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 0))
        preview_shell.grid_columnconfigure(0, weight=1)
        preview_shell.grid_rowconfigure(0, weight=1)
        self._camera_preview = ctk.CTkLabel(
            preview_shell,
            text="CAMERA IS ON STANDBY\n\nSELECT START CAMERA TO BEGIN",
            fg_color="#050606",
            corner_radius=12,
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            height=420,
        )
        self._camera_preview.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self._camera_preview.bind("<Configure>", self._resize_camera_image)

        self._recognition_overlay = ctk.CTkFrame(
            preview_shell,
            fg_color=COLOR_PANEL,
            corner_radius=12,
            border_width=1,
            border_color="#41474A",
        )
        self._recognition_overlay.place(relx=0.018, rely=0.97, anchor="sw")
        ctk.CTkLabel(
            self._recognition_overlay,
            text="LIVE RECOGNITION  /  01",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        ).pack(anchor="w", padx=13, pady=(9, 3))
        self._overlay_gesture = ctk.CTkLabel(
            self._recognition_overlay,
            text="STANDBY",
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=20, weight="bold"),
        )
        self._overlay_gesture.pack(anchor="w", padx=13)
        self._overlay_shape = ctk.CTkLabel(
            self._recognition_overlay,
            text="PRESS START TO BEGIN",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        )
        self._overlay_shape.pack(anchor="w", padx=13, pady=(2, 9))

        camera_bottom = ctk.CTkFrame(camera_frame, fg_color="transparent")
        camera_bottom.grid(row=2, column=0, sticky="ew", padx=17, pady=(10, 12))
        camera_bottom.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            camera_bottom,
            text="LIVE  /  640 × 480",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        ).grid(row=0, column=0, sticky="w")
        camera_actions = ctk.CTkFrame(camera_bottom, fg_color="transparent")
        camera_actions.grid(row=0, column=2, sticky="e")
        self._start_button = self._action_button(
            camera_actions,
            "START CAMERA",
            self._on_start,
            primary=True,
        )
        self._start_button.pack(side="left", padx=(0, 8))
        self._stop_button = self._action_button(camera_actions, "STOP", self._on_stop)
        self._stop_button.pack(side="left", padx=(0, 8))
        self._capture_button = self._action_button(
            camera_actions,
            "CAPTURE ↗",
            self._on_capture,
        )
        self._capture_button.pack(side="left")

        analysis_header = ctk.CTkFrame(page, fg_color="transparent")
        analysis_header.grid(row=6, column=0, sticky="ew", pady=(1, 13))
        analysis_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            analysis_header,
            text="ANALYSIS",
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=18),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            analysis_header,
            text="FRAME-BY-FRAME INTERPRETATION",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        ).grid(row=0, column=1, sticky="e")
        self._rule(page, row=7)

        analysis = ctk.CTkFrame(page, fg_color="transparent")
        analysis.grid(row=8, column=0, sticky="ew", pady=(0, 17))
        analysis.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="analysis")

        self._gesture_value = self._analysis_tile(
            analysis,
            0,
            "01  /  GESTURE",
            "—",
            "WAITING FOR INPUT",
        )
        self._shape_value = self._analysis_tile(
            analysis,
            1,
            "02  /  SHAPE",
            "—",
            "GEOMETRY CLASS",
        )
        self._hands_value = self._analysis_tile(
            analysis,
            2,
            "03  /  HANDS",
            "00",
            "LANDMARK TRACKS",
        )

        metrics = ctk.CTkFrame(analysis, fg_color="transparent")
        metrics.grid(row=0, column=3, sticky="nsew", padx=(12, 0))
        metrics.grid_columnconfigure(0, weight=1)
        self._metric_values: dict[str, ctk.CTkLabel] = {}
        for row, (key, title, initial) in enumerate(
            (
                ("fps", "PROCESSING SPEED", "0 FPS"),
                ("total", "TOTAL DETECTIONS", "0"),
            )
        ):
            if row:
                self._rule(metrics, row=row * 2 - 1)
            ctk.CTkLabel(
                metrics,
                text=title,
                text_color=COLOR_MUTED,
                font=ctk.CTkFont(family=FONT_MONO, size=8),
            ).grid(row=row * 2, column=0, sticky="w", pady=(9, 3))
            value = ctk.CTkLabel(
                metrics,
                text=initial,
                text_color=COLOR_TEXT,
                font=ctk.CTkFont(family=FONT_DISPLAY, size=16),
            )
            value.grid(row=row * 2 + 1, column=0, sticky="w", pady=(0, 8))
            self._metric_values[key] = value

        self._shape_canvas = Canvas(
            page,
            height=210,
            bg=COLOR_PANEL,
            highlightthickness=1,
            highlightbackground=LINE_COLOR,
            bd=0,
        )
        self._shape_canvas.grid(row=9, column=0, sticky="ew", pady=(0, 25))
        self._shape_canvas.bind("<Configure>", self._redraw_shape)

        footer = ctk.CTkFrame(
            page,
            fg_color=COLOR_PANEL,
            border_width=1,
            border_color=LINE_COLOR,
            corner_radius=14,
        )
        footer.grid(row=10, column=0, sticky="ew", pady=(0, 24))
        footer.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            footer,
            text="HANDSHAPE AI     /     COMPUTER VISION RESEARCH",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=10)
        ctk.CTkLabel(
            footer,
            text="BUILT FOR THE HUMAN HAND.     ↗",
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        ).grid(row=0, column=1, sticky="e", padx=12, pady=10)

    def _creator_name(self, parent, role: str, name: str) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=6)
        row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            row,
            text=role,
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            row,
            text=name,
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=15, weight="bold"),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

    def _feature_card(
        self,
        parent,
        column: int,
        number: str,
        title: str,
        description: str,
        shape: str,
    ) -> None:
        card = ctk.CTkFrame(
            parent,
            height=190,
            fg_color=COLOR_PANEL,
            corner_radius=20,
            border_width=1,
            border_color=COLOR_PANEL_BORDER,
        )
        card.grid(row=0, column=column, sticky="nsew", padx=5)
        card.grid_propagate(False)
        ctk.CTkLabel(
            card,
            text=number,
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        ).pack(anchor="w", padx=15, pady=(14, 5))
        ctk.CTkLabel(
            card,
            text=title,
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=14, weight="bold"),
        ).pack(anchor="w", padx=15)
        ctk.CTkLabel(
            card,
            text=description,
            text_color=COLOR_MUTED,
            justify="left",
            wraplength=155,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=10),
        ).pack(anchor="w", padx=15, pady=(7, 0))

        ornament = Canvas(
            card,
            width=78,
            height=54,
            bg=COLOR_PANEL,
            highlightthickness=0,
            bd=0,
        )
        ornament.pack(side="bottom", anchor="e", padx=12, pady=(0, 7))
        self._draw_feature_geometry(ornament, shape)
        ctk.CTkLabel(
            card,
            text="OPEN LIVE DEMO  ↗",
            text_color=COLOR_ACCENT,
            font=ctk.CTkFont(family=FONT_MONO, size=7),
        ).pack(side="bottom", anchor="w", padx=15, pady=(0, 7))
        self._bind_card_interaction(card, self.scroll_to_live)

    def _bind_card_interaction(
        self,
        card: ctk.CTkFrame,
        command: Callable[[], None] | None = None,
    ) -> None:
        def update_hover(_event=None) -> None:
            def apply_hover() -> None:
                if not card.winfo_exists():
                    return
                pointer_x = card.winfo_pointerx()
                pointer_y = card.winfo_pointery()
                inside = (
                    card.winfo_rootx() <= pointer_x < card.winfo_rootx() + card.winfo_width()
                    and card.winfo_rooty() <= pointer_y < card.winfo_rooty() + card.winfo_height()
                )
                card.configure(
                    fg_color=COLOR_PANEL_HOVER if inside else COLOR_PANEL,
                    border_color=COLOR_ACCENT if inside else COLOR_PANEL_BORDER,
                )

            self.after_idle(apply_hover)

        def bind_widget(widget) -> None:
            widget.bind("<Enter>", update_hover, add="+")
            widget.bind("<Leave>", update_hover, add="+")
            if command is not None:
                widget.bind("<Button-1>", lambda _event: command(), add="+")
            for child in widget.winfo_children():
                bind_widget(child)

        bind_widget(card)

    @staticmethod
    def _draw_feature_geometry(canvas: Canvas, shape: str) -> None:
        colors = {
            "circle": ("#324239", "#8FB99A", "#DBF4E0"),
            "v": ("#37413A", "#A6C9AE", "#F0FFF2"),
            "triangle": ("#303B34", "#91B99B", "#E1F7E5"),
            "square": ("#3C453D", "#B0D1B6", "#F3FFF4"),
        }
        shade, mid, light = colors[shape]
        if shape == "circle":
            canvas.create_oval(17, 5, 61, 49, fill=shade, outline=light, width=1)
            canvas.create_arc(
                21,
                9,
                57,
                45,
                start=205,
                extent=145,
                style="arc",
                outline=mid,
                width=3,
            )
        elif shape == "v":
            canvas.create_line(15, 9, 39, 49, 63, 9, fill=mid, width=5, joinstyle="round")
            canvas.create_oval(11, 5, 19, 13, fill=light, outline="")
            canvas.create_oval(59, 5, 67, 13, fill=light, outline="")
        elif shape == "triangle":
            canvas.create_polygon(39, 5, 68, 46, 10, 46, fill=shade, outline=light, width=1)
            canvas.create_polygon(39, 5, 40, 45, 68, 46, fill=mid, outline="")
            canvas.create_line(39, 5, 40, 45, 10, 46, fill=light, width=1)
        elif shape == "square":
            canvas.create_polygon(15, 8, 63, 8, 63, 48, 15, 48, fill=shade, outline=light, width=1)
            canvas.create_polygon(15, 8, 39, 26, 63, 8, fill=light, outline="")
            canvas.create_polygon(39, 26, 63, 8, 63, 48, fill=mid, outline="")
            canvas.create_line(15, 8, 39, 26, 15, 48, fill=light, width=1)
        else:
            canvas.create_polygon(39, 4, 67, 26, 39, 49, 11, 26, fill=shade, outline=light, width=1)
            canvas.create_polygon(39, 4, 39, 26, 67, 26, fill=light, outline="")
            canvas.create_polygon(39, 26, 39, 49, 11, 26, fill=mid, outline="")
            canvas.create_line(39, 4, 39, 49, fill=light, width=1)

    def _render_hero(self, event=None) -> None:
        if not hasattr(self, "_hero_canvas"):
            return
        width = max(self._hero_canvas.winfo_width(), 760)
        height = max(self._hero_canvas.winfo_height(), 420)
        canvas = self._hero_canvas
        canvas.delete("all")
        canvas.create_rectangle(
            0,
            0,
            width,
            height,
            fill="#08090B",
            outline="",
            tags=("hero_ambient",),
        )
        self._hero_ribbons = []
        line_widths = (42, 30, 18, 10, 5, 2, 1)
        paths = (
            (
                0.38,
                42.0,
                0.25,
                0.0,
                ("#171A1B", "#292D2F", "#484640", "#8A7F6B", "#CDD0C8", "#F3F4F2", "#FFFFFF"),
            ),
            (
                0.58,
                34.0,
                -0.21,
                math.pi,
                ("#17191C", "#292C32", "#464B55", "#838D9A", "#CFDDED", "#F1F6FC", "#FFFFFF"),
            ),
        )
        for base, amplitude, slope, phase_offset, colors in paths:
            for line_width, color in zip(line_widths, colors, strict=True):
                line = canvas.create_line(
                    0,
                    0,
                    1,
                    1,
                    fill=color,
                    width=line_width,
                    smooth=True,
                    splinesteps=16,
                    capstyle="round",
                    tags=("hero_ambient",),
                )
                self._hero_ribbons.append(
                    (line, base, amplitude, slope, phase_offset, 0.0)
                )
        glass = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        glass_draw = ImageDraw.Draw(glass)
        glass_draw.rounded_rectangle(
            (
                int(width * 0.17),
                int(height * 0.19),
                int(width * 0.83),
                int(height * 0.79),
            ),
            radius=24,
            fill=(8, 11, 12, 118),
            outline=(224, 235, 230, 72),
            width=1,
        )
        self._hero_glass_image = ImageTk.PhotoImage(glass)
        canvas.create_image(
            0,
            0,
            image=self._hero_glass_image,
            anchor="nw",
            tags=("hero_glass",),
        )
        canvas.create_arc(
            width * 0.12,
            height * 0.45,
            width * 0.88,
            height * 1.5,
            start=180,
            extent=180,
            style="arc",
            outline="#202326",
            width=2,
            tags=("hero_ambient",),
        )
        self._update_hero_ribbons()

        center = width / 2
        canvas.create_text(
            28,
            28,
            text="HSAI    /    COMPUTER VISION",
            anchor="nw",
            fill="#D9DDDA",
            font=(FONT_MONO, 9),
            tags=("hero_content",),
        )
        canvas.create_oval(
            width - 190,
            24,
            width - 182,
            32,
            fill=COLOR_ACCENT,
            outline="",
            tags=("hero_content",),
        )
        canvas.create_text(
            width - 28,
            28,
            text="21 LANDMARKS  ·  04 GESTURES",
            anchor="ne",
            fill="#B5BAB7",
            font=(FONT_MONO, 8),
            tags=("hero_content",),
        )
        canvas.create_text(
            center,
            height * 0.29,
            text="REAL-TIME HAND GESTURE RECOGNITION",
            fill=COLOR_ACCENT,
            font=(FONT_MONO, 9),
            tags=("hero_content",),
        )
        canvas.create_text(
            center,
            height * 0.41,
            text="Dari gestur,",
            fill=COLOR_TEXT,
            font=(FONT_DISPLAY, 43, "bold"),
            tags=("hero_content",),
        )
        canvas.create_text(
            center,
            height * 0.51,
            text="menjadi geometri.",
            fill=COLOR_TEXT,
            font=(FONT_DISPLAY, 43, "bold"),
            tags=("hero_content",),
        )
        canvas.create_text(
            center,
            height * 0.60,
            text="Ubah gerakan tangan menjadi bentuk secara langsung.",
            fill="#CFD4D0",
            font=(FONT_DISPLAY, 12),
            tags=("hero_content",),
        )

        button_width, button_height = 206, 38
        button_top = height * 0.68
        canvas.create_polygon(
            center - button_width / 2,
            button_top,
            center + button_width / 2,
            button_top,
            center + button_width / 2,
            button_top + button_height / 2,
            center + button_width / 2,
            button_top + button_height,
            center - button_width / 2,
            button_top + button_height,
            center - button_width / 2,
            button_top + button_height / 2,
            smooth=True,
            outline=COLOR_ACCENT,
            fill="#131816",
            width=1,
            tags=("hero_content", "hero_cta", "hero_cta_bg"),
        )
        canvas.create_text(
            center,
            button_top + button_height / 2,
            text="MULAI LIVE DEMO   ↗",
            fill=COLOR_ACCENT,
            font=(FONT_MONO, 8),
            tags=("hero_content", "hero_cta", "hero_cta_text"),
        )
        canvas.tag_bind("hero_cta", "<Button-1>", lambda _event: self.scroll_to_live())
        canvas.tag_bind(
            "hero_cta",
            "<Enter>",
            lambda _event: self._set_hero_cta_hover(True),
        )
        canvas.tag_bind(
            "hero_cta",
            "<Leave>",
            lambda _event: self._set_hero_cta_hover(False),
        )
        canvas.tag_bind(
            "hero_cta",
            "<Enter>",
            lambda _event: canvas.configure(cursor="hand2"),
            add="+",
        )
        canvas.tag_bind(
            "hero_cta",
            "<Leave>",
            lambda _event: canvas.configure(cursor=""),
            add="+",
        )

        divider_y = height - 57
        canvas.create_line(
            28,
            divider_y,
            width - 28,
            divider_y,
            fill="#45494A",
            tags=("hero_content",),
        )
        stats = (
            ("21", "HAND LANDMARKS"),
            ("04", "GEOMETRY SHAPES"),
            ("02", "HANDS TRACKED"),
            ("LIVE", "CAMERA ANALYSIS"),
        )
        for index, (value, label) in enumerate(stats):
            x = width * (index + 0.5) / len(stats)
            canvas.create_text(
                x,
                divider_y + 17,
                text=value,
                fill=COLOR_TEXT,
                font=(FONT_DISPLAY, 12, "bold"),
                tags=("hero_content",),
            )
            canvas.create_text(
                x,
                divider_y + 37,
                text=label,
                fill="#B3B9B4",
                font=(FONT_MONO, 7),
                tags=("hero_content",),
            )
            if index:
                separator_x = width * index / len(stats)
                canvas.create_line(
                    separator_x,
                    divider_y + 11,
                    separator_x,
                    height - 9,
                    fill="#505752",
                    tags=("hero_content",),
                )

        if self._hero_animation_id is None:
            self._hero_animation_id = self.after(60, self._animate_hero)

    def _update_hero_ribbons(self) -> None:
        width = max(self._hero_canvas.winfo_width(), 760)
        height = max(self._hero_canvas.winfo_height(), 420)
        for line, base, amplitude, slope, phase_offset, vertical_offset in self._hero_ribbons:
            coordinates = []
            for index in range(41):
                position = index / 40
                x = position * width
                y = (
                    base * height
                    + slope * height * (position - 0.5)
                    + amplitude * math.sin(position * math.tau + self._hero_phase + phase_offset)
                    + vertical_offset
                )
                coordinates.extend((x, y))
            self._hero_canvas.coords(line, *coordinates)
        self._hero_canvas.tag_lower("hero_ambient")

    def _animate_hero(self) -> None:
        self._hero_animation_id = None
        if not self.winfo_exists() or not hasattr(self, "_hero_canvas"):
            return
        self._hero_phase = (self._hero_phase + 0.045) % math.tau
        self._update_hero_ribbons()
        self._hero_animation_id = self.after(60, self._animate_hero)

    def _set_hero_cta_hover(self, hovering: bool) -> None:
        if not hasattr(self, "_hero_canvas"):
            return
        self._hero_canvas.itemconfigure(
            "hero_cta_bg",
            fill=COLOR_ACCENT if hovering else "#131816",
        )
        self._hero_canvas.itemconfigure(
            "hero_cta_text",
            fill=COLOR_BACKGROUND if hovering else COLOR_ACCENT,
        )

    def scroll_to_live(self) -> None:
        self.show_page("Live Detection")

    def _build_history_page(self, parent: ctk.CTkFrame) -> None:
        page = ctk.CTkScrollableFrame(
            parent,
            fg_color=COLOR_BACKGROUND,
            corner_radius=0,
            scrollbar_button_color="#34383B",
            scrollbar_button_hover_color="#555B5F",
        )
        page.grid_columnconfigure(0, weight=1)
        self._pages["History"] = page
        self._page_heading(page, "CAPTURE ARCHIVE", "Saved frames from your live sessions.")
        self._history_list = ctk.CTkFrame(page, fg_color="transparent")
        self._history_list.grid(row=2, column=0, sticky="ew", pady=(24, 0))
        self._history_list.grid_columnconfigure(0, weight=1)
        self.show_history([])

    def _build_about_page(self, parent: ctk.CTkFrame) -> None:
        page = ctk.CTkScrollableFrame(
            parent,
            fg_color=COLOR_BACKGROUND,
            corner_radius=0,
            scrollbar_button_color="#34383B",
            scrollbar_button_hover_color="#555B5F",
        )
        page.grid_columnconfigure(0, weight=1)
        self._pages["About Project"] = page
        self._page_heading(page, "ABOUT HANDSHAPE AI", "A new way to read movement.")

        details = ctk.CTkFrame(
            page,
            fg_color=COLOR_PANEL,
            border_width=1,
            border_color=COLOR_PANEL_BORDER,
            corner_radius=20,
        )
        details.grid(row=4, column=0, sticky="ew", pady=(28, 0))
        details.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            details,
            text=PROJECT_TITLE,
            text_color=COLOR_TEXT,
            wraplength=800,
            justify="left",
            font=ctk.CTkFont(family=FONT_DISPLAY, size=20),
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(22, 18))
        self._rule(details, row=1)
        ctk.CTkLabel(
            details,
            text=(
                "HandShape AI interprets hand landmarks in real time and translates gestures "
                "into geometric forms. OpenCV DNN and OpenCV Zoo ONNX models estimate 21 "
                "landmarks per hand; normalized coordinates, geometry checks, and temporal "
                "smoothing help keep the result stable."
            ),
            text_color=COLOR_MUTED,
            wraplength=760,
            justify="left",
            font=ctk.CTkFont(family=FONT_DISPLAY, size=14),
        ).grid(row=2, column=0, sticky="w", padx=22, pady=(18, 8))
        ctk.CTkLabel(
            details,
            text="CIRCLE      /      V SHAPE      /      TRIANGLE      /      SQUARE",
            text_color=COLOR_ACCENT,
            font=ctk.CTkFont(family=FONT_MONO, size=10),
        ).grid(row=3, column=0, sticky="w", padx=22, pady=(20, 22))
        self._bind_card_interaction(details)

    def _page_heading(self, parent, title: str, subtitle: str) -> None:
        ctk.CTkLabel(
            parent,
            text="Nº 02    /    HANDSHAPE AI",
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=9),
        ).grid(row=0, column=0, sticky="w", pady=(28, 14))
        self._rule(parent, row=1)
        ctk.CTkLabel(
            parent,
            text=title,
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=36),
        ).grid(row=2, column=0, sticky="w", pady=(28, 5))
        ctk.CTkLabel(
            parent,
            text=subtitle,
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=13),
        ).grid(row=3, column=0, sticky="w")

    def _analysis_tile(self, parent, column: int, title: str, value: str, caption: str):
        tile = ctk.CTkFrame(
            parent,
            fg_color=COLOR_PANEL,
            border_width=1,
            border_color=COLOR_PANEL_BORDER,
            corner_radius=18,
        )
        tile.grid(row=0, column=column, sticky="nsew", padx=(0, 12))
        tile.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            tile,
            text=title,
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=8),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 5))
        value_label = ctk.CTkLabel(
            tile,
            text=value,
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_DISPLAY, size=18),
        )
        value_label.grid(row=1, column=0, sticky="w", padx=12)
        ctk.CTkLabel(
            tile,
            text=caption,
            text_color=COLOR_MUTED,
            font=ctk.CTkFont(family=FONT_MONO, size=7),
        ).grid(row=2, column=0, sticky="w", padx=12, pady=(4, 10))
        self._bind_card_interaction(tile)
        return value_label

    def _action_button(
        self,
        parent,
        label: str,
        command: Callable[[], None],
        *,
        primary: bool = False,
    ) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=label,
            height=31,
            corner_radius=18,
            border_width=1,
            border_color=COLOR_ACCENT if primary else LINE_COLOR,
            fg_color=COLOR_ACCENT if primary else COLOR_PANEL,
            hover_color="#D4F1DB" if primary else COLOR_PANEL_LIGHT,
            text_color=COLOR_BACKGROUND if primary else COLOR_TEXT,
            font=ctk.CTkFont(family=FONT_MONO, size=8, weight="bold"),
            command=command,
        )

    @staticmethod
    def _rule(parent, *, row: int) -> None:
        ctk.CTkFrame(parent, height=1, fg_color=LINE_COLOR, corner_radius=0).grid(
            row=row,
            column=0,
            columnspan=8,
            sticky="ew",
        )

    def show_page(self, name: str) -> None:
        target = "Dashboard" if name == "Live Detection" else name
        for page in set(self._pages.values()):
            page.grid_forget()
        self._pages[target].grid(row=0, column=0, sticky="nsew")
        for button_name, button in self._nav_buttons.items():
            active = button_name == name
            button.configure(
                text_color=COLOR_TEXT if active else COLOR_MUTED,
                fg_color=COLOR_PANEL_LIGHT if active else "transparent",
                border_width=1 if active else 0,
                border_color=LINE_COLOR,
            )
        if target == "Dashboard":
            self.after_idle(lambda: self._scroll_landing_to(name == "Live Detection"))
        if target == "History":
            self.event_generate("<<RefreshHistory>>", when="tail")

    def _scroll_landing_to(self, live: bool) -> None:
        page = self._landing_page
        canvas = page._parent_canvas
        canvas.update_idletasks()
        bounds = canvas.bbox("all")
        if bounds is None:
            return
        target_y = self._live_section.winfo_y() if live else 0
        content_height = max(bounds[3] - bounds[1], 1)
        fraction = min(max((target_y - bounds[1]) / content_height, 0), 1)
        canvas.yview_moveto(fraction)

    def set_camera_status(self, status: str, *, error: bool = False) -> None:
        running = status == "Running"
        label = "●  LIVE FEED" if running else f"●  {status.upper()}"
        color = COLOR_DANGER if error else COLOR_SUCCESS if running else COLOR_MUTED
        self._camera_status_badge.configure(text=label, text_color=color)
        self._header_status.configure(
            text="●  CAMERA CONNECTED" if running else "●  SYSTEM STANDBY",
            text_color=color,
        )
        if not running:
            overlay_message = {
                "Connecting": ("CONNECTING", "OPENING CAMERA FEED"),
                "Stopping": ("STOPPING", "RELEASING CAMERA"),
                "Unavailable": ("CAMERA ERROR", "CHECK CAMERA CONNECTION"),
                "Stopped": ("STANDBY", "PRESS START TO BEGIN"),
            }.get(status, (status.upper(), "WAITING FOR CAMERA"))
            self._set_recognition_overlay(*overlay_message, color=color)

    def set_detection_status(self, status: str, *, warning: bool = False) -> None:
        color = COLOR_WARNING if warning else COLOR_SUCCESS if status == "Detected" else COLOR_MUTED
        if hasattr(self, "_gesture_value") and status not in {"Detected", "No gesture"}:
            self._gesture_value.configure(text=status.upper(), text_color=color)

    def set_controls_enabled(self, *, start: bool, stop: bool) -> None:
        self._start_button.configure(state="normal" if start else "disabled")
        self._stop_button.configure(state="normal" if stop else "disabled")
        self._capture_button.configure(state="normal" if stop else "disabled")

    def show_camera_frame(
        self,
        frame_rgb: np.ndarray,
        hand_count: int,
        result: RecognitionResult | None,
        fps: float,
        total_detections: int,
    ) -> None:
        self._camera_image = None
        self._last_frame_rgb = frame_rgb.copy()
        self._render_camera_image()
        self._hands_value.configure(
            text=f"{hand_count:02d}",
            text_color=COLOR_TEXT if hand_count else COLOR_MUTED,
        )
        self._metric_values["fps"].configure(text=f"{fps:.1f} FPS")
        self._metric_values["total"].configure(text=str(total_detections))
        self._current_result = result
        self._gesture_value.configure(
            text=result.label if result else "—",
            text_color=COLOR_TEXT if result else COLOR_MUTED,
        )
        self._shape_value.configure(
            text=result.shape_name.upper() if result else "—",
            text_color=COLOR_TEXT if result else COLOR_MUTED,
        )
        if result is not None:
            self._set_recognition_overlay(
                result.label,
                result.shape_name.upper(),
                color=COLOR_TEXT,
            )
        elif hand_count:
            self._set_recognition_overlay(
                "SCANNING",
                "HAND DETECTED · FORM A GESTURE",
                color=COLOR_MUTED,
            )
        else:
            self._set_recognition_overlay(
                "NO HAND",
                "MOVE YOUR HAND INTO FRAME",
                color=COLOR_MUTED,
            )
        self._shape_canvas.delete("all")
        draw_geometry(
            self._shape_canvas,
            result.label if result else None,
            result.points if result else (),
        )

    def _resize_camera_image(self, _event=None) -> None:
        if hasattr(self, "_last_frame_rgb"):
            self._render_camera_image()

    def _render_camera_image(self) -> None:
        frame = getattr(self, "_last_frame_rgb", None)
        if frame is None:
            return
        available_width = max(self._camera_preview.winfo_width() - 4, 320)
        available_height = max(self._camera_preview.winfo_height() - 4, 240)
        image = Image.fromarray(frame)
        image.thumbnail((available_width, available_height), Image.Resampling.LANCZOS)
        self._camera_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size,
        )
        self._camera_preview.configure(image=self._camera_image, text="")

    def show_camera_message(self, text: str) -> None:
        self._camera_image = None
        if hasattr(self, "_last_frame_rgb"):
            del self._last_frame_rgb
        self._camera_preview.configure(image=None, text=text)
        lowered = text.lower()
        if "connect" in lowered:
            overlay = ("CONNECTING", "OPENING CAMERA FEED")
        elif "stop" in lowered:
            overlay = ("STANDBY", "PRESS START TO BEGIN")
        elif "camera" in lowered or "webcam" in lowered:
            overlay = ("CAMERA ERROR", "CHECK CAMERA CONNECTION")
        else:
            overlay = ("STANDBY", "PRESS START TO BEGIN")
        self._set_recognition_overlay(*overlay, color=COLOR_MUTED)

    def _set_recognition_overlay(self, gesture: str, shape: str, *, color: str) -> None:
        self._overlay_gesture.configure(text=gesture, text_color=color)
        self._overlay_shape.configure(text=shape)

    def show_history(self, screenshots: Sequence[Path]) -> None:
        if not hasattr(self, "_history_list"):
            return
        for child in self._history_list.winfo_children():
            child.destroy()

        if not screenshots:
            ctk.CTkLabel(
                self._history_list,
                text="NO CAPTURES YET     /     Start a session and save a frame.",
                text_color=COLOR_MUTED,
                font=ctk.CTkFont(family=FONT_MONO, size=9),
            ).grid(row=0, column=0, sticky="w", pady=15)
            return

        for row_index, path in enumerate(screenshots):
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y.%m.%d   %H:%M:%S")
            row = ctk.CTkFrame(
                self._history_list,
                fg_color=COLOR_PANEL,
                border_width=1,
                border_color=COLOR_PANEL_BORDER,
                corner_radius=18,
            )
            row.grid(row=row_index, column=0, sticky="ew", pady=5)
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                row,
                text=path.name.upper(),
                text_color=COLOR_TEXT,
                font=ctk.CTkFont(family=FONT_MONO, size=9),
            ).grid(row=0, column=0, sticky="w", padx=14, pady=13)
            ctk.CTkLabel(
                row,
                text=modified,
                text_color=COLOR_MUTED,
                font=ctk.CTkFont(family=FONT_MONO, size=8),
            ).grid(row=0, column=1, sticky="e", padx=14, pady=13)
            self._bind_card_interaction(row)

    def refresh_history(self, screenshots: Sequence[Path]) -> None:
        self.show_history(screenshots)

    def show_notice(self, title: str, message: str, *, error: bool = False) -> None:
        if error:
            messagebox.showerror(title, message, parent=self)
        else:
            messagebox.showinfo(title, message, parent=self)

    def _redraw_shape(self, _event=None) -> None:
        if not hasattr(self, "_shape_canvas"):
            return
        result = self._current_result
        draw_geometry(
            self._shape_canvas,
            result.label if result else None,
            result.points if result else (),
        )
