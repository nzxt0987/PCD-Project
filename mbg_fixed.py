import os
import tkinter as tk

from tkinter import filedialog, messagebox
from datetime import datetime

import customtkinter as ctk

import cv2
import numpy as np

from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# SCI-FI THEME
ctk.set_appearance_mode("dark")

BG_COLOR = "#05080C"
SIDEBAR_COLOR = "#071017"

# ACCENT
CYAN = "#00E5FF"
CYAN_BRIGHT = "#18F5FF"
CYAN_DIM = "#0A5964"

# STATUS
GREEN = "#00F5A0"
RED = "#FF4D5A"
YELLOW = "#FFD166"

TEXT = "#E8F7FA"
TEXT_MUTED = "#76919A"
TEXT_DARK = "#3F5C65"

BORDER = "#12343B"

# PANEL COLOR
PANEL_BG = "#080D12"
PANEL_INNER = "#050A0E"

class ColorSpaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "PRD Image Processing v1.0.0"
        )

        self.root.geometry(
            "1280x800"
        )

        self.root.minsize(
            1100,
            700
        )

        self.root.configure(
            fg_color=BG_COLOR
        )

        # IMAGE STATE
        self.cv_img = None
        self.image_path = None
        self.result_img = None
        self.base_result_img = None
        self.adjusted_img = None

        # IMAGE 2 STATE (for logic/bitwise operations)
        self.cv_img_2 = None
        self.image_path_2 = None

        # UI STATE
        self.nav_buttons = {}
        self.pages = {}
        self.hist_canvas = None
        self.analysis_hist_canvas = None
        self.status_var = tk.StringVar(value="SYSTEM READY")
        self.current_mode = "Grayscale"

        # All available transformation modes
        self.transform_modes = [
            "Grayscale",
            "HSV • Hue",
            "HSV • Saturation",
            "HSV • Value",
            "YCrCb • Y (Luma)",
            "YCrCb • Cr",
            "YCrCb • Cb",
            "Channel Red",
            "Channel Green",
            "Channel Blue",
            "CIE-LAB • L",
            "CIE-LAB • a",
            "CIE-LAB • b",
        ]

        self.setup_ui()
        self.start_clock()

    def setup_ui(self):
        self.root.grid_rowconfigure(
            0,
            weight=1
        )

        self.root.grid_columnconfigure(
            1,
            weight=1
        )

        self.create_sidebar()

        self.main_area = ctk.CTkFrame(
            self.root,
            fg_color=BG_COLOR,
            corner_radius=0
        )

        self.main_area.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.main_area.grid_rowconfigure(
            1,
            weight=1
        )

        self.main_area.grid_rowconfigure(
            2,
            weight=0
        )

        self.main_area.grid_columnconfigure(
            0,
            weight=1
        )

        # HEADER
        self.create_header()

        # CONTAINER
        self.content_container = ctk.CTkFrame(
            self.main_area,
            fg_color="transparent"
        )

        self.content_container.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=0,
            pady=(0, 0)
        )

        self.content_container.grid_rowconfigure(
            0,
            weight=1
        )

        self.content_container.grid_columnconfigure(
            0,
            weight=1
        )

        # PAGES
        self.create_overview_page()
        self.create_analysis_page()

        self.create_footer()
        self.show_overview()

    # HEADER
    def create_header(self):
        """Membuat Header Utama Pada Area Konten."""

        self.header_frame = ctk.CTkFrame(
            self.main_area,
            fg_color=PANEL_BG,
            corner_radius=0,
            border_width=1,
            border_color=BORDER,
            height=72
        )

        self.header_frame.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=0,
            pady=0
        )
        self.header_frame.grid_propagate(False)

        # Left side
        left_header = ctk.CTkFrame(
            self.header_frame,
            fg_color="transparent"
        )
        left_header.pack(side="left", fill="y", padx=(20, 0), pady=10)

        tag_label = ctk.CTkLabel(
            left_header,
            text="// DIGITAL IMAGE TRANSFORMATION & ANALYSIS",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_MUTED
        )
        tag_label.pack(anchor="w")

        self.page_title = ctk.CTkLabel(
            left_header,
            text="PRD // IMAGE PROCESSING",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=22,
                weight="bold"
            ),
            text_color=TEXT
        )
        self.page_title.pack(anchor="w", pady=(2, 0))

        self.page_subtitle = ctk.CTkLabel(
            left_header,
            text="Load  –  Transform  –  Analyze  –  Visualize",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_MUTED
        )
        self.page_subtitle.pack(anchor="w", pady=(2, 0))

        # Right side
        right_header = ctk.CTkFrame(
            self.header_frame,
            fg_color="transparent"
        )
        right_header.pack(side="right", fill="y", padx=(0, 20), pady=10)

        self.header_date = ctk.CTkLabel(
            right_header,
            text=datetime.now().strftime("%a, %d %b %Y"),
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_MUTED
        )
        self.header_date.pack(anchor="e")

        self.header_time = ctk.CTkLabel(
            right_header,
            text=datetime.now().strftime("%H:%M:%S"),
            font=ctk.CTkFont(
                family="Consolas",
                size=22,
                weight="bold"
            ),
            text_color=TEXT
        )
        self.header_time.pack(anchor="e")

        self.header_status = ctk.CTkLabel(
            right_header,
            text="● SYSTEM ONLINE",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=GREEN
        )
        self.header_status.pack(anchor="e", pady=(2, 0))

    def start_clock(self):
        """Memperbarui waktu di header setiap detik."""
        now = datetime.now()
        self.header_time.configure(
            text=now.strftime("%H:%M:%S")
        )
        self.header_date.configure(
            text=now.strftime("%a, %d %b %Y")
        )
        self.root.after(1000, self.start_clock)

    # SIDEBAR
    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self.root,
            width=200,
            fg_color=SIDEBAR_COLOR,
            corner_radius=0,
            border_width=1,
            border_color=BORDER
        )
        self.sidebar.grid(
            row=0,
            column=0,
            sticky="nsew"
        )
        self.sidebar.grid_propagate(
            False
        )

        # BRAND
        brand_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )

        brand_frame.pack(
            fill=tk.X,
            padx=18,
            pady=(20, 4)
        )

        brand_title = ctk.CTkLabel(
            brand_frame,
            text="PRD",
            text_color=CYAN_BRIGHT,
            font=ctk.CTkFont(
                family="Consolas",
                size=28,
                weight="bold"
            )
        )

        brand_title.pack(
            anchor="w"
        )

        brand_subtitle = ctk.CTkLabel(
            brand_frame,
            text="IMAGE PROCESSING",
            text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            )
        )

        brand_subtitle.pack(
            anchor="w"
        )

        version_label = ctk.CTkLabel(
            brand_frame,
            text="Digital Vision, Real Insight",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            )
        )

        version_label.pack(
            anchor="w",
            pady=(2, 0)
        )

        # Separator
        ctk.CTkFrame(
            self.sidebar,
            height=1,
            fg_color=BORDER
        ).pack(fill=tk.X, padx=18, pady=(12, 8))

        # NAVIGATION LABEL
        nav_label = ctk.CTkLabel(
            self.sidebar,
            text="NAVIGATION",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(
                family="Consolas",
                size=8,
                weight="bold"
            )
        )

        nav_label.pack(
            anchor="w",
            padx=20,
            pady=(4, 4)
        )

        self.create_nav_button(
            "◉  Overview",
            "overview",
            self.show_overview
        )

        self.create_nav_button(
            "◈  Analysis",
            "analysis",
            self.show_analysis
        )

        # SPACER
        spacer = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )

        spacer.pack(
            fill=tk.BOTH,
            expand=True
        )

        # SESSION INFORMATION
        separator = ctk.CTkFrame(
            self.sidebar,
            height=1,
            fg_color=BORDER
        )

        separator.pack(
            fill=tk.X,
            padx=18,
            pady=(8, 8)
        )

        session_title = ctk.CTkLabel(
            self.sidebar,
            text="SESSION",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(
                family="Consolas",
                size=8,
                weight="bold"
            )
        )

        session_title.pack(
            anchor="w",
            padx=20,
            pady=(0, 4)
        )

        # Group
        self._sidebar_info_row("Group", "GOJEK")
        self._sidebar_info_row("Session", "PCD-2026")

        # Separator
        ctk.CTkFrame(
            self.sidebar,
            height=1,
            fg_color=BORDER
        ).pack(fill=tk.X, padx=18, pady=(8, 4))

        env_title = ctk.CTkLabel(
            self.sidebar,
            text="ENVIRONMENT",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(
                family="Consolas",
                size=8,
                weight="bold"
            )
        )
        env_title.pack(anchor="w", padx=20, pady=(4, 2))

        env_label = ctk.CTkLabel(
            self.sidebar,
            text="Python | OpenCV | CTk",
            text_color=CYAN_DIM,
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            )
        )
        env_label.pack(anchor="w", padx=20, pady=(0, 16))

    def _sidebar_info_row(self, label, value):
        """Helper: buat baris informasi di sidebar."""
        frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame.pack(fill=tk.X, padx=20, pady=1)

        ctk.CTkLabel(
            frame,
            text=label,
            text_color=TEXT_DARK,
            font=ctk.CTkFont(family="Consolas", size=8)
        ).pack(anchor="w")

        ctk.CTkLabel(
            frame,
            text=value,
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(family="Consolas", size=9, weight="bold")
        ).pack(anchor="w")

    def create_nav_button(
        self,
        text,
        page_name,
        command
    ):
        button = ctk.CTkButton(
            self.sidebar,
            text=text,
            command=command,
            anchor="w",
            height=38,
            corner_radius=6,
            fg_color="transparent",
            hover_color="#0C252C",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            )
        )
        button.pack(
            fill=tk.X,
            padx=8,
            pady=2
        )

        self.nav_buttons[
            page_name
        ] = button

    # OVERVIEW PAGE
    def create_overview_page(self):
        """Membuat halaman Overview sebagai workspace utama."""

        self.pages["overview"] = ctk.CTkFrame(
            self.content_container,
            fg_color="transparent"
        )

        page = self.pages["overview"]

        # Layout: left workspace (images + controls) | right info panel
        page.grid_columnconfigure(0, weight=1)
        page.grid_columnconfigure(1, weight=0)
        page.grid_rowconfigure(0, weight=1)

        # ---- LEFT COLUMN (images + controls) ----
        left_col = ctk.CTkFrame(page, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(14, 6), pady=(8, 8))

        left_col.grid_rowconfigure(0, weight=0)
        left_col.grid_rowconfigure(1, weight=1)
        left_col.grid_rowconfigure(2, weight=0)
        left_col.grid_rowconfigure(3, weight=0)
        left_col.grid_rowconfigure(4, weight=0)
        left_col.grid_rowconfigure(5, weight=0)
        left_col.grid_columnconfigure(0, weight=1)

        # Section title bar
        title_bar = ctk.CTkFrame(left_col, fg_color="transparent")
        title_bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 4))

        ctk.CTkLabel(
            title_bar,
            text="OVERVIEW",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=16,
                weight="bold"
            ),
            text_color=TEXT
        ).pack(side="left")

        # Action buttons in title bar
        btn_frame = ctk.CTkFrame(title_bar, fg_color="transparent")
        btn_frame.pack(side="right")

        self.open_button = ctk.CTkButton(
            btn_frame,
            text="⬜ Open Image",
            width=110,
            height=30,
            corner_radius=6,
            fg_color="#0A2027",
            hover_color="#10333C",
            border_width=1,
            border_color=CYAN_DIM,
            text_color=CYAN_BRIGHT,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.load_image
        )
        self.open_button.pack(side="left", padx=(0, 6))

        self.remove_button = ctk.CTkButton(
            btn_frame,
            text="✕ Remove",
            width=90,
            height=30,
            corner_radius=6,
            fg_color="#1A0A10",
            hover_color="#2A1218",
            border_width=1,
            border_color="#4A1525",
            text_color=RED,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.clear_image
        )
        self.remove_button.pack(side="left", padx=(0, 6))

        self.save_button = ctk.CTkButton(
            btn_frame,
            text="✦ Save Result",
            width=110,
            height=30,
            corner_radius=6,
            fg_color="#0A2027",
            hover_color="#10333C",
            border_width=1,
            border_color=CYAN_DIM,
            text_color=CYAN_BRIGHT,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.save_result
        )
        self.save_button.pack(side="left", padx=(0, 6))

        # Image 2 buttons
        self.open_button_2 = ctk.CTkButton(
            btn_frame,
            text="⬜ Open Img 2",
            width=110,
            height=30,
            corner_radius=6,
            fg_color="#0A2027",
            hover_color="#10333C",
            border_width=1,
            border_color=CYAN_DIM,
            text_color=YELLOW,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.load_image_2
        )
        self.open_button_2.pack(side="left", padx=(0, 6))

        self.remove_button_2 = ctk.CTkButton(
            btn_frame,
            text="✕ Rm Img 2",
            width=90,
            height=30,
            corner_radius=6,
            fg_color="#1A0A10",
            hover_color="#2A1218",
            border_width=1,
            border_color="#4A1525",
            text_color=RED,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.clear_image_2
        )
        self.remove_button_2.pack(side="left")

        # ---- IMAGE ROW ----
        image_row = ctk.CTkFrame(left_col, fg_color="transparent")
        image_row.grid(row=1, column=0, sticky="nsew", padx=0, pady=(0, 6))
        image_row.grid_columnconfigure(0, weight=1)
        image_row.grid_columnconfigure(1, weight=1)
        image_row.grid_columnconfigure(2, weight=1)
        image_row.grid_rowconfigure(0, weight=1)

        # SOURCE IMAGE CARD (Image 1)
        source_panel = ctk.CTkFrame(
            image_row,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8
        )
        source_panel.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(6, 3)
        )

        source_header = ctk.CTkFrame(source_panel, fg_color="transparent")
        source_header.pack(fill="x", padx=10, pady=(8, 3))

        ctk.CTkLabel(
            source_header,
            text="◼ SOURCE IMAGE",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(side="left")

        ctk.CTkLabel(
            source_header,
            text="Image 1",
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT_DARK
        ).pack(side="right")

        self.lbl_orig_overview = ctk.CTkLabel(
            source_panel,
            text="No image loaded",
            fg_color=PANEL_INNER,
            corner_radius=6,
            text_color=TEXT_DARK
        )
        self.lbl_orig_overview.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=(0, 4)
        )

        self.source_info_label = ctk.CTkLabel(
            source_panel,
            text="",
            font=ctk.CTkFont(family="Consolas", size=8),
            text_color=TEXT_DARK
        )
        self.source_info_label.pack(anchor="w", padx=10, pady=(0, 6))

        # IMAGE 2 CARD
        img2_panel = ctk.CTkFrame(
            image_row,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8
        )
        img2_panel.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(3, 3)
        )

        img2_header = ctk.CTkFrame(img2_panel, fg_color="transparent")
        img2_header.pack(fill="x", padx=10, pady=(8, 3))

        ctk.CTkLabel(
            img2_header,
            text="◼ IMAGE 2",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=YELLOW
        ).pack(side="left")

        ctk.CTkLabel(
            img2_header,
            text="Logic Input",
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT_DARK
        ).pack(side="right")

        self.lbl_img2_overview = ctk.CTkLabel(
            img2_panel,
            text="No image loaded",
            fg_color=PANEL_INNER,
            corner_radius=6,
            text_color=TEXT_DARK
        )
        self.lbl_img2_overview.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=(0, 4)
        )

        self.img2_info_label = ctk.CTkLabel(
            img2_panel,
            text="",
            font=ctk.CTkFont(family="Consolas", size=8),
            text_color=TEXT_DARK
        )
        self.img2_info_label.pack(anchor="w", padx=10, pady=(0, 6))

        # RESULT IMAGE CARD
        result_panel = ctk.CTkFrame(
            image_row,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8
        )
        result_panel.grid(
            row=0,
            column=2,
            sticky="nsew",
            padx=(3, 6)
        )

        result_header = ctk.CTkFrame(result_panel, fg_color="transparent")
        result_header.pack(fill="x", padx=10, pady=(8, 3))

        ctk.CTkLabel(
            result_header,
            text="◼ RESULT IMAGE",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=GREEN
        ).pack(side="left")

        self.result_mode_label = ctk.CTkLabel(
            result_header,
            text="Processed",
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT_DARK
        )
        self.result_mode_label.pack(side="right")

        self.lbl_result_overview = ctk.CTkLabel(
            result_panel,
            text="Apply a transformation",
            fg_color=PANEL_INNER,
            corner_radius=6,
            text_color=TEXT_DARK
        )
        self.lbl_result_overview.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=(0, 4)
        )

        self.result_info_label = ctk.CTkLabel(
            result_panel,
            text="",
            font=ctk.CTkFont(family="Consolas", size=8),
            text_color=TEXT_DARK
        )
        self.result_info_label.pack(anchor="w", padx=10, pady=(0, 6))

        # ---- QUICK TRANSFORM PANEL ----
        transform_panel = ctk.CTkFrame(
            left_col,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8,
            height=44
        )
        transform_panel.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=6,
            pady=(0, 4)
        )
        transform_panel.pack_propagate(False)

        ctk.CTkLabel(
            transform_panel,
            text="QUICK TRANSFORM",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            side="left",
            padx=(12, 8),
            pady=6
        )

        self.overview_mode_combo = ctk.CTkComboBox(
            transform_panel,
            values=self.transform_modes,
            width=180,
            height=28,
            corner_radius=5,
            border_width=1,
            border_color=BORDER,
            button_color="#0A2027",
            button_hover_color="#10333C",
            fg_color=PANEL_INNER,
            text_color=TEXT,
            dropdown_fg_color=PANEL_BG,
            dropdown_hover_color="#0C252C",
            dropdown_text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            state="readonly"
        )
        self.overview_mode_combo.pack(
            side="left",
            padx=(0, 6),
            pady=6
        )
        self.overview_mode_combo.set("Grayscale")

        ctk.CTkButton(
            transform_panel,
            text="▶ APPLY",
            width=80,
            height=28,
            corner_radius=5,
            fg_color=CYAN_DIM,
            hover_color="#0D6E7B",
            text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.convert_color
        ).pack(
            side="left",
            pady=6
        )

        # ---- LOGIC OPERATIONS PANEL ----
        logic_panel = ctk.CTkFrame(
            left_col,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8,
            height=44
        )
        logic_panel.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=6,
            pady=(0, 4)
        )
        logic_panel.pack_propagate(False)

        ctk.CTkLabel(
            logic_panel,
            text="LOGIC OPERATIONS",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=YELLOW
        ).pack(
            side="left",
            padx=(12, 8),
            pady=6
        )

        self.logic_mode_combo = ctk.CTkComboBox(
            logic_panel,
            values=["AND", "OR", "NOT", "XOR"],
            width=100,
            height=28,
            corner_radius=5,
            border_width=1,
            border_color=BORDER,
            button_color="#0A2027",
            button_hover_color="#10333C",
            fg_color=PANEL_INNER,
            text_color=TEXT,
            dropdown_fg_color=PANEL_BG,
            dropdown_hover_color="#0C252C",
            dropdown_text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            state="readonly"
        )
        self.logic_mode_combo.pack(
            side="left",
            padx=(0, 6),
            pady=6
        )
        self.logic_mode_combo.set("AND")

        ctk.CTkButton(
            logic_panel,
            text="▶ APPLY",
            width=80,
            height=28,
            corner_radius=5,
            fg_color="#3D3310",
            hover_color="#5A4D18",
            text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.apply_logic_operation
        ).pack(
            side="left",
            pady=6
        )

        self.logic_status_label = ctk.CTkLabel(
            logic_panel,
            text="Img1 ⊕ Img2 → Result",
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT_DARK
        )
        self.logic_status_label.pack(
            side="left",
            padx=(8, 0),
            pady=6
        )

        # Arithmetic operations panel
        arithmetic_panel = ctk.CTkFrame(
            left_col,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8,
            height=44
        )
        arithmetic_panel.grid(
            row=4,
            column=0,
            sticky="ew",
            padx=6,
            pady=(0, 4)
        )
        arithmetic_panel.pack_propagate(False)

        ctk.CTkLabel(
            arithmetic_panel,
            text="ARITHMETIC",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            side="left",
            padx=(12, 8),
            pady=6
        )

        self.arithmetic_mode_combo = ctk.CTkComboBox(
            arithmetic_panel,
            values=["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "BLEND"],
            width=105,
            height=28,
            corner_radius=5,
            border_width=1,
            border_color=BORDER,
            button_color="#0A2027",
            button_hover_color="#10333C",
            fg_color=PANEL_INNER,
            text_color=TEXT,
            dropdown_fg_color=PANEL_BG,
            dropdown_hover_color="#0C252C",
            dropdown_text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            state="readonly"
        )
        self.arithmetic_mode_combo.pack(
            side="left",
            padx=(0, 6),
            pady=6
        )
        self.arithmetic_mode_combo.set("ADD")

        ctk.CTkLabel(
            arithmetic_panel,
            text="α",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=TEXT_MUTED
        ).pack(
            side="left",
            pady=6
        )

        self.blend_alpha_value = ctk.CTkLabel(
            arithmetic_panel,
            text="0.50",
            width=32,
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT
        )
        self.blend_alpha_value.pack(
            side="left",
            pady=6
        )

        self.arithmetic_alpha_slider = ctk.CTkSlider(
            arithmetic_panel,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            width=80,
            height=14,
            button_length=8,
            fg_color="#0A1A1F",
            progress_color=CYAN_DIM,
            button_color=CYAN,
            button_hover_color=CYAN_BRIGHT,
            command=self.update_alpha_label
        )
        self.arithmetic_alpha_slider.set(0.5)
        self.arithmetic_alpha_slider.pack(
            side="left",
            padx=(0, 8),
            pady=6
        )

        ctk.CTkButton(
            arithmetic_panel,
            text="▶ APPLY",
            width=75,
            height=28,
            corner_radius=5,
            fg_color="#0A333C",
            hover_color="#104D5B",
            text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.apply_arithmetic_operation
        ).pack(
            side="left",
            pady=6
        )

        self.arithmetic_status_label = ctk.CTkLabel(
            arithmetic_panel,
            text="Img1 op Img2 → Result",
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT_DARK
        )
        self.arithmetic_status_label.pack(
            side="left",
            padx=(8, 0),
            pady=6
        )

        # Adjustment panel
        adjustment_panel = ctk.CTkFrame(
            left_col,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8,
            height=44
        )
        adjustment_panel.grid(
            row=5,
            column=0,
            sticky="ew",
            padx=6,
            pady=(0, 4)
        )
        adjustment_panel.pack_propagate(False)

        ctk.CTkLabel(
            adjustment_panel,
            text="ADJUSTMENTS",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=GREEN
        ).pack(
            side="left",
            padx=(12, 8),
            pady=6
        )

        ctk.CTkLabel(
            adjustment_panel,
            text="Brightness",
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT_MUTED
        ).pack(side="left", pady=6)

        self.brightness_value = ctk.CTkLabel(
            adjustment_panel,
            text="0",
            width=30,
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT
        )
        self.brightness_value.pack(side="left", pady=6)

        self.brightness_slider = ctk.CTkSlider(
            adjustment_panel,
            from_=-100,
            to=100,
            number_of_steps=200,
            width=120,
            height=14,
            button_length=8,
            fg_color="#0A1A1F",
            progress_color=CYAN_DIM,
            button_color=CYAN,
            button_hover_color=CYAN_BRIGHT,
            command=self.preview_adjustment
        )
        self.brightness_slider.set(0)
        self.brightness_slider.pack(
            side="left",
            padx=(0, 10),
            pady=6
        )

        ctk.CTkLabel(
            adjustment_panel,
            text="Contrast",
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT_MUTED
        ).pack(side="left", pady=6)

        self.contrast_value = ctk.CTkLabel(
            adjustment_panel,
            text="1.0",
            width=30,
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT
        )
        self.contrast_value.pack(side="left", pady=6)

        self.contrast_slider = ctk.CTkSlider(
            adjustment_panel,
            from_=0.2,
            to=2.0,
            number_of_steps=180,
            width=120,
            height=14,
            button_length=8,
            fg_color="#0A1A1F",
            progress_color=CYAN_DIM,
            button_color=CYAN,
            button_hover_color=CYAN_BRIGHT,
            command=self.preview_adjustment
        )
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(
            side="left",
            padx=(0, 8),
            pady=6
        )

        # RESET button (right side)
        ctk.CTkButton(
            adjustment_panel,
            text="RESET",
            width=60,
            height=26,
            corner_radius=5,
            fg_color="#211D0B",
            hover_color="#38300F",
            border_width=1,
            border_color="#66551A",
            text_color=YELLOW,
            font=ctk.CTkFont(
                family="Consolas",
                size=8,
                weight="bold"
            ),
            command=self.reset_adjustment
        ).pack(
            side="right",
            padx=(4, 12),
            pady=6
        )

        # APPLY button
        ctk.CTkButton(
            adjustment_panel,
            text="APPLY",
            width=60,
            height=26,
            corner_radius=5,
            fg_color=CYAN_DIM,
            hover_color="#0D6E7B",
            text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=8,
                weight="bold"
            ),
            command=self.apply_adjustment
        ).pack(
            side="right",
            padx=2,
            pady=6
        )

        # ---- RIGHT COLUMN (info + histogram) ----
        right_col = ctk.CTkFrame(page, fg_color="transparent", width=280)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(0, 14), pady=(8, 8))
        right_col.grid_propagate(False)

        right_col.grid_rowconfigure(0, weight=0)
        right_col.grid_rowconfigure(1, weight=0)
        right_col.grid_rowconfigure(2, weight=1)
        right_col.grid_columnconfigure(0, weight=1)

        # IMAGE INFORMATION
        info_panel = ctk.CTkFrame(
            right_col,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8
        )
        info_panel.grid(
            row=0,
            column=0,
            sticky="new",
            pady=(0, 6)
        )

        ctk.CTkLabel(
            info_panel,
            text="◼ IMAGE INFORMATION",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            anchor="w",
            padx=12,
            pady=(10, 6)
        )

        self.overview_info = ctk.CTkLabel(
            info_panel,
            text="No image loaded.",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_MUTED,
            justify="left",
            anchor="nw"
        )
        self.overview_info.pack(
            fill="x",
            padx=12,
            pady=(0, 6)
        )

        # IMAGE 2 INFORMATION
        ctk.CTkFrame(
            info_panel,
            height=1,
            fg_color=BORDER
        ).pack(fill="x", padx=12, pady=2)

        ctk.CTkLabel(
            info_panel,
            text="IMAGE 2 INFO",
            font=ctk.CTkFont(
                family="Consolas",
                size=8,
                weight="bold"
            ),
            text_color=YELLOW
        ).pack(
            anchor="w",
            padx=12,
            pady=(4, 2)
        )

        self.overview_info_2 = ctk.CTkLabel(
            info_panel,
            text="No image loaded.",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_MUTED,
            justify="left",
            anchor="nw"
        )
        self.overview_info_2.pack(
            fill="x",
            padx=12,
            pady=(0, 6)
        )

        # WORKFLOW STATUS
        ctk.CTkFrame(
            info_panel,
            height=1,
            fg_color=BORDER
        ).pack(fill="x", padx=12, pady=2)

        ctk.CTkLabel(
            info_panel,
            text="WORKFLOW STATUS",
            font=ctk.CTkFont(
                family="Consolas",
                size=8,
                weight="bold"
            ),
            text_color=TEXT_MUTED
        ).pack(
            anchor="w",
            padx=12,
            pady=(6, 2)
        )

        self.overview_status = ctk.CTkLabel(
            info_panel,
            text="Ready for image processing.",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=GREEN,
            justify="left",
            anchor="nw"
        )
        self.overview_status.pack(
            fill="x",
            padx=12,
            pady=(0, 10)
        )

        # HISTOGRAM (compact, in right column)
        hist_panel = ctk.CTkFrame(
            right_col,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8
        )
        hist_panel.grid(
            row=1,
            column=0,
            sticky="new",
            pady=(0, 6)
        )

        hist_header = ctk.CTkFrame(hist_panel, fg_color="transparent")
        hist_header.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            hist_header,
            text="◼ HISTOGRAM",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=GREEN
        ).pack(side="left")

        self.hist_mode_combo = ctk.CTkComboBox(
            hist_header,
            values=["RGB", "Red", "Green", "Blue", "Grayscale"],
            width=90,
            height=22,
            corner_radius=4,
            border_width=1,
            border_color=BORDER,
            button_color="#0A2027",
            button_hover_color="#10333C",
            fg_color=PANEL_INNER,
            text_color=TEXT,
            dropdown_fg_color=PANEL_BG,
            dropdown_hover_color="#0C252C",
            dropdown_text_color=TEXT,
            font=ctk.CTkFont(family="Consolas", size=8),
            command=self.update_overview_histogram,
            state="readonly"
        )
        self.hist_mode_combo.pack(side="right")
        self.hist_mode_combo.set("RGB")

        self.overview_histogram_area = ctk.CTkFrame(
            hist_panel,
            fg_color=PANEL_INNER,
            corner_radius=6,
            height=160
        )
        self.overview_histogram_area.pack(
            fill="x",
            padx=12,
            pady=(0, 12)
        )
        self.overview_histogram_area.pack_propagate(False)

        ctk.CTkLabel(
            self.overview_histogram_area,
            text="Load an image to generate histogram.",
            font=ctk.CTkFont(family="Consolas", size=8),
            text_color=TEXT_DARK
        ).pack(expand=True)

        self.update_action_buttons()

    # ANALYSIS PAGE
    def create_analysis_page(self):
        """Membuat Halaman Analysis."""

        self.pages["analysis"] = ctk.CTkFrame(
            self.content_container,
            fg_color="transparent"
        )

        page = self.pages["analysis"]

        page.grid_columnconfigure(0, weight=1)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(1, weight=1)

        # Title
        ctk.CTkLabel(
            page,
            text="IMAGE ANALYSIS",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=16,
                weight="bold"
            ),
            text_color=TEXT
        ).grid(
            row=0, column=0, columnspan=2,
            sticky="w", padx=20, pady=(10, 4)
        )

        # LEFT: Stats + Pixel Inspector
        left_panel = ctk.CTkFrame(
            page,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8
        )
        left_panel.grid(
            row=1, column=0,
            sticky="nsew",
            padx=(20, 6),
            pady=(0, 14)
        )

        ctk.CTkLabel(
            left_panel,
            text="◼ IMAGE STATISTICS",
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            anchor="w",
            padx=14,
            pady=(12, 6)
        )

        self.analysis_info = ctk.CTkLabel(
            left_panel,
            text="No image loaded.",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_MUTED,
            justify="left",
            anchor="nw"
        )
        self.analysis_info.pack(
            anchor="w",
            fill="x",
            padx=14,
            pady=(0, 10)
        )

        # Pixel Inspector
        ctk.CTkFrame(
            left_panel,
            height=1,
            fg_color=BORDER
        ).pack(fill="x", padx=14, pady=4)

        ctk.CTkLabel(
            left_panel,
            text="◼ PIXEL INSPECTOR",
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(anchor="w", padx=14, pady=(8, 6))

        # Coordinate inputs
        coord_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        coord_frame.pack(fill="x", padx=14, pady=(0, 6))

        ctk.CTkLabel(
            coord_frame, text="X:", width=20,
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color=TEXT_MUTED
        ).pack(side="left")

        self.pixel_x_entry = ctk.CTkEntry(
            coord_frame, width=60, height=26,
            corner_radius=4,
            fg_color=PANEL_INNER,
            border_color=BORDER,
            text_color=TEXT,
            font=ctk.CTkFont(family="Consolas", size=9)
        )
        self.pixel_x_entry.pack(side="left", padx=(2, 8))
        self.pixel_x_entry.insert(0, "0")

        ctk.CTkLabel(
            coord_frame, text="Y:", width=20,
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color=TEXT_MUTED
        ).pack(side="left")

        self.pixel_y_entry = ctk.CTkEntry(
            coord_frame, width=60, height=26,
            corner_radius=4,
            fg_color=PANEL_INNER,
            border_color=BORDER,
            text_color=TEXT,
            font=ctk.CTkFont(family="Consolas", size=9)
        )
        self.pixel_y_entry.pack(side="left", padx=(2, 8))
        self.pixel_y_entry.insert(0, "0")

        ctk.CTkButton(
            coord_frame,
            text="Inspect",
            width=70, height=26,
            corner_radius=4,
            fg_color=CYAN_DIM,
            hover_color="#0D6E7B",
            text_color=TEXT,
            font=ctk.CTkFont(family="Consolas", size=9, weight="bold"),
            command=self.inspect_pixel
        ).pack(side="left", padx=(4, 0))

        self.pixel_info_label = ctk.CTkLabel(
            left_panel,
            text="Enter coordinates and click Inspect.",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color=TEXT_MUTED,
            justify="left",
            anchor="nw"
        )
        self.pixel_info_label.pack(
            fill="x", padx=14, pady=(0, 14)
        )

        # RIGHT: Histogram
        right_panel = ctk.CTkFrame(
            page,
            fg_color=PANEL_BG,
            border_width=1,
            border_color=BORDER,
            corner_radius=8
        )
        right_panel.grid(
            row=1, column=1,
            sticky="nsew",
            padx=(6, 20),
            pady=(0, 14)
        )

        hist_header = ctk.CTkFrame(right_panel, fg_color="transparent")
        hist_header.pack(fill="x", padx=14, pady=(12, 6))

        ctk.CTkLabel(
            hist_header,
            text="◼ HISTOGRAM",
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            text_color=GREEN
        ).pack(side="left")

        self.analysis_hist_mode = ctk.CTkComboBox(
            hist_header,
            values=["RGB", "Red", "Green", "Blue", "Grayscale"],
            width=100,
            height=24,
            corner_radius=4,
            border_width=1,
            border_color=BORDER,
            button_color="#0A2027",
            button_hover_color="#10333C",
            fg_color=PANEL_INNER,
            text_color=TEXT,
            dropdown_fg_color=PANEL_BG,
            dropdown_hover_color="#0C252C",
            dropdown_text_color=TEXT,
            font=ctk.CTkFont(family="Consolas", size=9),
            command=self.update_analysis_histogram,
            state="readonly"
        )
        self.analysis_hist_mode.pack(side="right")
        self.analysis_hist_mode.set("RGB")

        self.histogram_area = ctk.CTkFrame(
            right_panel,
            fg_color=PANEL_INNER,
            corner_radius=6
        )
        self.histogram_area.pack(
            fill="both",
            expand=True,
            padx=14,
            pady=(0, 14)
        )

        ctk.CTkLabel(
            self.histogram_area,
            text="Load an image to generate histogram.",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_DARK
        ).pack(
            expand=True
        )

    # FOOTER
    def create_footer(self):
        """Membuat Footer."""

        self.footer_frame = ctk.CTkFrame(
            self.main_area,
            fg_color=PANEL_INNER,
            height=28,
            corner_radius=0,
            border_width=1,
            border_color=BORDER
        )

        self.footer_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=0,
            pady=0
        )

        self.footer_frame.grid_propagate(False)

        ctk.CTkLabel(
            self.footer_frame,
            text="PRD CORE",
            font=ctk.CTkFont(
                family="Consolas",
                size=8,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            side="left",
            padx=(14, 6)
        )

        ctk.CTkLabel(
            self.footer_frame,
            textvariable=self.status_var,
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT_MUTED
        ).pack(
            side="left",
            padx=4
        )

        ctk.CTkLabel(
            self.footer_frame,
            text="PRD IMAGE PROCESSING  •  PCD-2026  •  v1.0.0",
            font=ctk.CTkFont(
                family="Consolas",
                size=8
            ),
            text_color=TEXT_DARK
        ).pack(
            side="right",
            padx=14
        )

    # HISTOGRAM RENDERING
    def _render_histogram(self, target_area, mode="RGB", figsize=(3.0, 1.8)):
        """Render histogram ke target area yang diberikan."""
        img = self.result_img if self.result_img is not None else self.cv_img
        if img is None:
            return None

        for widget in target_area.winfo_children():
            widget.destroy()

        figure = Figure(
            figsize=figsize,
            dpi=90,
            facecolor=PANEL_INNER
        )

        axis = figure.add_subplot(111)
        axis.set_facecolor(PANEL_INNER)

        channels = img.shape[2] if len(img.shape) == 3 else 1

        if mode == "Grayscale" or channels == 1:
            # Convert to grayscale if needed
            if channels == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
            axis.fill_between(
                range(256), histogram.flatten(),
                alpha=0.5, color="#AAAAAA"
            )
            axis.plot(histogram, color="#CCCCCC", linewidth=0.8)

        elif mode == "Red":
            if channels == 3:
                histogram = cv2.calcHist([img], [2], None, [256], [0, 256])
                axis.fill_between(
                    range(256), histogram.flatten(),
                    alpha=0.4, color="#FF4D5A"
                )
                axis.plot(histogram, color="#FF4D5A", linewidth=0.8)

        elif mode == "Green":
            if channels == 3:
                histogram = cv2.calcHist([img], [1], None, [256], [0, 256])
                axis.fill_between(
                    range(256), histogram.flatten(),
                    alpha=0.4, color="#00F5A0"
                )
                axis.plot(histogram, color="#00F5A0", linewidth=0.8)

        elif mode == "Blue":
            if channels == 3:
                histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
                axis.fill_between(
                    range(256), histogram.flatten(),
                    alpha=0.4, color="#00A5FF"
                )
                axis.plot(histogram, color="#00A5FF", linewidth=0.8)

        else:
            # RGB mode
            if channels == 3:
                colors_map = [
                    (0, "#00A5FF", "B"),
                    (1, "#00F5A0", "G"),
                    (2, "#FF4D5A", "R"),
                ]
                for idx, color, label in colors_map:
                    histogram = cv2.calcHist([img], [idx], None, [256], [0, 256])
                    axis.plot(histogram, color=color, linewidth=0.8, label=label, alpha=0.8)
                    axis.fill_between(
                        range(256), histogram.flatten(),
                        alpha=0.15, color=color
                    )
            else:
                histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
                axis.plot(histogram, color="#CCCCCC", linewidth=0.8)

        axis.set_xlim([0, 256])
        axis.tick_params(
            colors=TEXT_MUTED,
            labelsize=6
        )

        for spine in axis.spines.values():
            spine.set_color(BORDER)

        axis.grid(alpha=0.1, color=TEXT_DARK)

        figure.tight_layout(pad=0.5)

        canvas = FigureCanvasTkAgg(
            figure,
            master=target_area
        )
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=2,
            pady=2
        )

        return canvas

    def update_overview_histogram(self, mode=None):
        """Perbarui histogram di Overview."""
        if mode is None:
            mode = self.hist_mode_combo.get()
        self.hist_canvas = self._render_histogram(
            self.overview_histogram_area,
            mode=mode,
            figsize=(3.0, 1.6)
        )

    def update_analysis_histogram(self, mode=None):
        """Perbarui histogram di Analysis."""
        if mode is None:
            mode = self.analysis_hist_mode.get()
        self.analysis_hist_canvas = self._render_histogram(
            self.histogram_area,
            mode=mode,
            figsize=(5.0, 3.5)
        )

    def update_all_histograms(self):
        """Perbarui histogram di semua halaman."""
        self.update_overview_histogram()
        self.update_analysis_histogram()

    # ANALYSIS FUNCTIONS
    def update_analysis(self):
        """Memperbarui informasi dan histogram gambar."""

        if self.cv_img is None:
            return

        h, w = self.cv_img.shape[:2]
        channels = self.cv_img.shape[2] if len(self.cv_img.shape) == 3 else 1

        file_name = (
            os.path.basename(self.image_path)
            if self.image_path
            else "Unknown"
        )

        # File size
        file_size_str = "N/A"
        if self.image_path and os.path.exists(self.image_path):
            size_bytes = os.path.getsize(self.image_path)
            if size_bytes >= 1024 * 1024:
                file_size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            elif size_bytes >= 1024:
                file_size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                file_size_str = f"{size_bytes} B"

        # Image statistics
        mean_val = np.mean(self.cv_img, axis=(0, 1))
        min_val = int(np.min(self.cv_img))
        max_val = int(np.max(self.cv_img))

        if channels == 3:
            # BGR order → display as RGB
            mean_str = f"{mean_val[2]:.1f}, {mean_val[1]:.1f}, {mean_val[0]:.1f}"
            ch_label = f"{channels} (BGR)"
        else:
            mean_str = f"{mean_val:.1f}"
            ch_label = "1 (Gray)"

        # File extension as format
        _, ext = os.path.splitext(file_name)
        fmt = ext.upper().replace(".", "") if ext else "N/A"

        info_text = (
            f"File       : {file_name}\n"
            f"Format     : {fmt}\n"
            f"Resolution : {w} × {h}\n"
            f"Channels   : {ch_label}\n"
            f"File Size  : {file_size_str}\n"
            f"Data Type  : {self.cv_img.dtype}\n"
            f"\n"
            f"Mean (RGB) : {mean_str}\n"
            f"Min        : {min_val}\n"
            f"Max        : {max_val}"
        )

        self.analysis_info.configure(text=info_text)
        self.update_analysis_histogram()

    def inspect_pixel(self):
        """Inspeksi nilai pixel pada koordinat tertentu."""
        if self.cv_img is None:
            self.pixel_info_label.configure(
                text="No image loaded.",
                text_color=RED
            )
            return

        try:
            x = int(self.pixel_x_entry.get())
            y = int(self.pixel_y_entry.get())
        except ValueError:
            self.pixel_info_label.configure(
                text="Invalid coordinates.",
                text_color=RED
            )
            return

        h, w = self.cv_img.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            self.pixel_info_label.configure(
                text=f"Out of bounds. Image: {w}×{h}",
                text_color=RED
            )
            return

        channels = self.cv_img.shape[2] if len(self.cv_img.shape) == 3 else 1

        if channels == 3:
            b, g, r = self.cv_img[y, x]
            # Also compute HSV
            hsv = cv2.cvtColor(
                self.cv_img[y:y+1, x:x+1],
                cv2.COLOR_BGR2HSV
            )
            hv, sv, vv = hsv[0, 0]

            info = (
                f"Pixel ({x}, {y})\n"
                f"─────────────\n"
                f"R : {r}\n"
                f"G : {g}\n"
                f"B : {b}\n"
                f"─────────────\n"
                f"H : {hv}\n"
                f"S : {sv}\n"
                f"V : {vv}"
            )
        else:
            val = self.cv_img[y, x]
            info = (
                f"Pixel ({x}, {y})\n"
                f"─────────────\n"
                f"Gray : {val}"
            )

        self.pixel_info_label.configure(
            text=info,
            text_color=TEXT
        )

    # ADJUSTMENT
    def preview_adjustment(self, value=None):
        """Memperbarui preview brightness dan contrast secara real-time."""

        if self.cv_img is None:
            return

        brightness = int(self.brightness_slider.get())
        contrast = float(self.contrast_slider.get())

        self.brightness_value.configure(
            text=str(brightness)
        )

        self.contrast_value.configure(
            text=f"{contrast:.1f}"
        )

        source = self.base_result_img

        if source is None:
            source = self.cv_img

        # Contrast adjustment (separate from brightness)
        if contrast != 1.0:
            target = cv2.convertScaleAbs(source, alpha=contrast, beta=0)
        else:
            target = source

        # Brightness adjustment using cv2.add() per assignment requirement:
        # cv2.add(self.cv_img, np.array([slider_value]))
        slider_value = float(brightness)
        if len(target.shape) == 3 and target.shape[2] > 1:
            val_arr = np.array([slider_value] * target.shape[2], dtype=np.float64)
        else:
            val_arr = np.array([slider_value])
        adjusted = cv2.add(target, val_arr)

        self.adjusted_img = adjusted
        self.result_img = adjusted
        self.update_action_buttons()

        self.render_label(
            adjusted,
            self.lbl_result_overview
        )

        mode_text = self.current_mode
        self.result_mode_label.configure(
            text=f"Preview: {mode_text} | B{brightness:+d} | C{contrast:.1f}"
        )

        self.overview_status.configure(
            text=f"Live preview • Brightness {brightness:+d} • Contrast {contrast:.1f}"
        )

        self.status_var.set(
            f"Preview Adjustment : Brightness {brightness:+d} | Contrast {contrast:.1f}"
        )

        self.update_overview_histogram()

    def apply_adjustment(self):
        """Menerapkan brightness dan contrast pada citra."""

        if self.cv_img is None:
            messagebox.showwarning(
                "Peringatan",
                "Silahkan Buka Citra Terlebih Dahulu."
            )
            return

        brightness = int(self.brightness_slider.get())
        contrast = float(self.contrast_slider.get())

        source = self.base_result_img

        if source is None:
            source = self.cv_img

        # Contrast adjustment (separate from brightness)
        if contrast != 1.0:
            target = cv2.convertScaleAbs(source, alpha=contrast, beta=0)
        else:
            target = source

        # Brightness adjustment using cv2.add() per assignment requirement:
        # cv2.add(self.cv_img, np.array([slider_value]))
        slider_value = float(brightness)
        if len(target.shape) == 3 and target.shape[2] > 1:
            val_arr = np.array([slider_value] * target.shape[2], dtype=np.float64)
        else:
            val_arr = np.array([slider_value])
        adjusted = cv2.add(target, val_arr)

        self.adjusted_img = adjusted
        self.result_img = adjusted
        self.update_action_buttons()

        self.render_label(
            adjusted,
            self.lbl_result_overview
        )

        mode_text = self.current_mode
        self.result_mode_label.configure(
            text=f"{mode_text} • B{brightness:+d} • C{contrast:.1f}"
        )

        self.overview_status.configure(
            text=f"Adjustment applied • B {brightness:+d} • C {contrast:.1f}"
        )

        self.status_var.set(
            f"Adjustment aktif : Brightness {brightness} | Contrast {contrast:.1f}"
        )

        self.update_all_histograms()

    def reset_adjustment(self):
        """Mengembalikan nilai adjustment ke kondisi awal."""

        self.brightness_slider.set(0)
        self.contrast_slider.set(1.0)

        self.brightness_value.configure(
            text="0"
        )

        self.contrast_value.configure(
            text="1.0"
        )

        self.adjusted_img = None

        source = self.base_result_img

        if source is None:
            source = self.cv_img

        if source is not None:
            self.result_img = source.copy()
            self.update_action_buttons()

            self.render_label(
                self.result_img,
                self.lbl_result_overview
            )

        self.result_mode_label.configure(
            text=f"{self.current_mode} • Adjustment reset"
        )

        self.overview_status.configure(
            text="Adjustment reset. Base transformation restored."
        )

        self.status_var.set(
            "Adjustment di-reset"
        )

        self.update_all_histograms()

    # SAVE
    def save_result(self):
        """Menyimpan hasil transformasi atau adjustment."""

        if self.result_img is None:
            messagebox.showwarning(
                "Peringatan",
                "Belum ada hasil gambar untuk disimpan."
            )
            return

        file_path = filedialog.asksaveasfilename(
            title="Simpan Hasil",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        success = cv2.imwrite(
            file_path,
            self.result_img
        )

        if success:
            self.status_var.set(
                f"Hasil disimpan : {file_path}"
            )
            messagebox.showinfo(
                "Berhasil",
                "Hasil gambar berhasil disimpan."
            )
        else:
            messagebox.showerror(
                "Error",
                "Hasil gambar gagal disimpan."
            )

    # PAGE NAVIGATION
    def show_page(
        self,
        page_name
    ):
        for page in self.pages.values():
            page.grid_remove()

        self.pages[
            page_name
        ].grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        for name, button in self.nav_buttons.items():
            if name == page_name:
                button.configure(
                    fg_color="#0B2D35",
                    text_color=CYAN_BRIGHT
                )

            else:
                button.configure(
                    fg_color="transparent",
                    text_color=TEXT_MUTED
                )

    def show_overview(self):
        self.show_page(
            "overview"
        )

    def show_analysis(self):
        self.show_page(
            "analysis"
        )
        # Refresh analysis data when switching
        if self.cv_img is not None:
            self.update_analysis()

    # BUTTON STATE MANAGEMENT
    def update_action_buttons(self):
        """Mengatur status tombol sesuai kondisi workspace."""

        has_image = self.cv_img is not None
        has_image_2 = self.cv_img_2 is not None
        has_result = self.result_img is not None

        self.remove_button.configure(
            state="normal" if has_image else "disabled"
        )

        self.save_button.configure(
            state="normal" if has_result else "disabled"
        )

        self.remove_button_2.configure(
            state="normal" if has_image_2 else "disabled"
        )

    # LOAD IMAGE
    def load_image(self):

        file_path = filedialog.askopenfilename(
            title="Pilih Citra",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        image = cv2.imread(
            file_path
        )

        if image is None:

            messagebox.showerror(
                "Error",
                "Gambar gagal dibaca"
            )

            return

        self.cv_img = image
        self.image_path = file_path
        self.result_img = None
        self.base_result_img = None
        self.adjusted_img = None

        self.render_label(
            self.cv_img,
            self.lbl_orig_overview
        )

        self.lbl_result_overview.configure(
            image="",
            text="Apply a transformation"
        )
        self.lbl_result_overview.image = None

        self.status_var.set(
            f"Citra dimuat : {os.path.basename(file_path)}"
        )

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        # File size
        file_size_str = "N/A"
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            if size_bytes >= 1024 * 1024:
                file_size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            elif size_bytes >= 1024:
                file_size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                file_size_str = f"{size_bytes} B"

        # Image statistics
        mean_val = np.mean(image, axis=(0, 1))
        min_val = int(np.min(image))
        max_val = int(np.max(image))

        if channels == 3:
            mean_str = f"{mean_val[2]:.1f}, {mean_val[1]:.1f}, {mean_val[0]:.1f}"
            ch_label = f"{channels} (BGR)"
        else:
            mean_str = f"{mean_val:.1f}"
            ch_label = "1 (Gray)"

        _, ext = os.path.splitext(file_path)
        fmt = ext.upper().replace(".", "") if ext else "N/A"

        self.overview_info.configure(
            text=(
                f"File       : {os.path.basename(file_path)}\n"
                f"Format     : {fmt}\n"
                f"Resolution : {width} × {height}\n"
                f"Channels   : {ch_label}\n"
                f"File Size  : {file_size_str}\n"
                f"Data Type  : {image.dtype}\n"
                f"\n"
                f"Mean (RGB) : {mean_str}\n"
                f"Min        : {min_val}\n"
                f"Max        : {max_val}"
            )
        )

        # Source info beneath image
        self.source_info_label.configure(
            text=f"{width} × {height}  |  {fmt}  |  {file_size_str}"
        )
        self.result_info_label.configure(text="")

        self.overview_status.configure(
            text="Image loaded. Ready for transformation."
        )

        self.result_mode_label.configure(
            text="No transformation"
        )

        self.brightness_slider.set(0)
        self.contrast_slider.set(1.0)
        self.brightness_value.configure(text="0")
        self.contrast_value.configure(text="1.0")

        self.update_action_buttons()
        self.update_all_histograms()

    # CLEAR IMAGE
    def clear_image(self):
        """Menghapus gambar yang sedang aktif dari workspace (tidak menghapus file dari disk)."""

        self.cv_img = None
        self.image_path = None
        self.result_img = None
        self.base_result_img = None
        self.adjusted_img = None

        self.lbl_orig_overview.configure(
            image="",
            text="No image loaded"
        )
        self.lbl_orig_overview.image = None

        self.lbl_result_overview.configure(
            image="",
            text="Apply a transformation"
        )
        self.lbl_result_overview.image = None

        self.overview_mode_combo.set("Grayscale")
        self.current_mode = "Grayscale"

        self.brightness_slider.set(0)
        self.contrast_slider.set(1.0)

        self.brightness_value.configure(text="0")
        self.contrast_value.configure(text="1.0")

        self.overview_info.configure(
            text="No image loaded."
        )

        self.overview_status.configure(
            text="Ready for image processing."
        )

        self.result_mode_label.configure(
            text="No transformation"
        )

        self.source_info_label.configure(text="")
        self.result_info_label.configure(text="")

        self.analysis_info.configure(
            text="No image loaded."
        )

        for widget in self.histogram_area.winfo_children():
            widget.destroy()

        for widget in self.overview_histogram_area.winfo_children():
            widget.destroy()

        ctk.CTkLabel(
            self.overview_histogram_area,
            text="Load an image to generate histogram.",
            font=ctk.CTkFont(family="Consolas", size=8),
            text_color=TEXT_DARK
        ).pack(expand=True)

        ctk.CTkLabel(
            self.histogram_area,
            text="Load an image to generate histogram.",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color=TEXT_DARK
        ).pack(expand=True)

        self.hist_canvas = None
        self.analysis_hist_canvas = None

        self.pixel_info_label.configure(
            text="Enter coordinates and click Inspect.",
            text_color=TEXT_MUTED
        )

        self.logic_status_label.configure(
            text="Img1 ⊕ Img2 → Result",
            text_color=TEXT_DARK
        )
        self.arithmetic_status_label.configure(
            text="Img1 op Img2 → Result",
            text_color=TEXT_DARK
        )

        self.update_action_buttons()

        self.status_var.set(
            "Ready - No image loaded"
        )

    # LOAD IMAGE 2
    def load_image_2(self):
        """Memuat citra kedua untuk operasi logika/bitwise."""

        file_path = filedialog.askopenfilename(
            title="Pilih Citra 2",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        image = cv2.imread(file_path)

        if image is None:
            messagebox.showerror(
                "Error",
                "Gambar 2 gagal dibaca."
            )
            return

        self.cv_img_2 = image
        self.image_path_2 = file_path

        self.render_label(
            self.cv_img_2,
            self.lbl_img2_overview
        )

        h2, w2 = image.shape[:2]
        ch2 = image.shape[2] if len(image.shape) == 3 else 1
        _, ext2 = os.path.splitext(file_path)
        fmt2 = ext2.upper().replace(".", "") if ext2 else "N/A"

        file_size_str_2 = "N/A"
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            if size_bytes >= 1024 * 1024:
                file_size_str_2 = f"{size_bytes / (1024 * 1024):.1f} MB"
            elif size_bytes >= 1024:
                file_size_str_2 = f"{size_bytes / 1024:.1f} KB"
            else:
                file_size_str_2 = f"{size_bytes} B"

        self.img2_info_label.configure(
            text=f"{w2} × {h2}  |  {fmt2}  |  {file_size_str_2}"
        )

        ch_label_2 = f"{ch2} (BGR)" if ch2 == 3 else "1 (Gray)"
        self.overview_info_2.configure(
            text=(
                f"File       : {os.path.basename(file_path)}\n"
                f"Resolution : {w2} × {h2}\n"
                f"Channels   : {ch_label_2}\n"
                f"Data Type  : {image.dtype}\n"
                f"File Size  : {file_size_str_2}"
            )
        )

        self.overview_status.configure(
            text="Image 2 loaded. Ready for logic or arithmetic operation."
        )

        self.status_var.set(
            f"Citra 2 dimuat : {os.path.basename(file_path)}"
        )

        self.update_action_buttons()

    # CLEAR IMAGE 2
    def clear_image_2(self):
        """Menghapus citra kedua dari workspace (tidak menghapus file dari disk)."""

        self.cv_img_2 = None
        self.image_path_2 = None

        self.lbl_img2_overview.configure(
            image="",
            text="No image loaded"
        )
        self.lbl_img2_overview.image = None

        self.img2_info_label.configure(text="")

        self.overview_info_2.configure(
            text="No image loaded."
        )

        self.overview_status.configure(
            text="Image 2 removed."
        )

        self.logic_status_label.configure(
            text="Img1 ⊕ Img2 → Result",
            text_color=TEXT_DARK
        )
        self.arithmetic_status_label.configure(
            text="Img1 op Img2 → Result",
            text_color=TEXT_DARK
        )

        self.status_var.set(
            "Image 2 removed"
        )

        self.update_action_buttons()

    # LOGIC / BITWISE OPERATIONS
    def apply_logic_operation(self):
        """Menerapkan operasi logika/bitwise pada citra."""

        operation = self.logic_mode_combo.get()

        if self.cv_img is None:
            messagebox.showwarning(
                "Peringatan",
                "Silahkan buka Image 1 terlebih dahulu."
            )
            return

        if operation == "NOT":
            try:
                result = cv2.bitwise_not(self.cv_img)
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Gagal melakukan operasi NOT: {e}"
                )
                return

            self.base_result_img = result.copy()
            self.adjusted_img = None
            self.result_img = result.copy()

            self.brightness_slider.set(0)
            self.contrast_slider.set(1.0)
            self.brightness_value.configure(text="0")
            self.contrast_value.configure(text="1.0")

            self.render_label(
                result,
                self.lbl_result_overview
            )

            self.current_mode = f"Logic: {operation}"
            self.result_mode_label.configure(
                text=f"Logic: {operation}"
            )

            h, w = result.shape[:2]
            self.result_info_label.configure(
                text=f"{w} × {h}  |  {operation}"
            )

            self.overview_status.configure(
                text=f"Bitwise {operation} applied to Image 1."
            )

            self.logic_status_label.configure(
                text=f"✔ {operation} applied",
                text_color=GREEN
            )
            self.arithmetic_status_label.configure(
                text="Img1 op Img2 → Result",
                text_color=TEXT_DARK
            )

            self.status_var.set(
                f"Logic Operation : {operation}"
            )

            self.update_action_buttons()
            self.update_all_histograms()
            return

        if self.cv_img_2 is None:
            messagebox.showwarning(
                "Peringatan",
                f"Silahkan buka Image 2 sebelum menerapkan {operation}."
            )
            return

        try:
            # --- Make compatible copies (never modify originals) ---
            img1 = self.cv_img.copy()
            img2 = self.cv_img_2.copy()

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            resized_note = ""

            # Resize Image 2 to match Image 1 if dimensions differ
            if (h1, w1) != (h2, w2):
                img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)
                resized_note = f" (Image 2 resized: {w2}×{h2} → {w1}×{h1})"

            # Ensure 3-channel or 1-channel format (handle 4-channel BGRA)
            if len(img1.shape) == 3 and img1.shape[2] == 4:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
            if len(img2.shape) == 3 and img2.shape[2] == 4:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

            # Match channel counts
            ch1 = img1.shape[2] if len(img1.shape) == 3 else 1
            ch2 = img2.shape[2] if len(img2.shape) == 3 else 1

            if ch1 != ch2:
                if ch1 == 1 and ch2 == 3:
                    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                elif ch1 == 3 and ch2 == 1:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

            # Ensure matching data types
            if img1.dtype != img2.dtype:
                img2 = img2.astype(img1.dtype)

            if operation == "AND":
                result = cv2.bitwise_and(img1, img2)
            elif operation == "OR":
                result = cv2.bitwise_or(img1, img2)
            elif operation == "XOR":
                result = cv2.bitwise_xor(img1, img2)
            else:
                messagebox.showerror(
                    "Error",
                    f"Operasi tidak dikenal: {operation}"
                )
                return

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Gagal melakukan operasi {operation}: {e}"
            )
            return

        self.base_result_img = result.copy()
        self.adjusted_img = None
        self.result_img = result.copy()

        self.brightness_slider.set(0)
        self.contrast_slider.set(1.0)
        self.brightness_value.configure(text="0")
        self.contrast_value.configure(text="1.0")

        self.render_label(
            result,
            self.lbl_result_overview
        )

        self.current_mode = f"Logic: {operation}"
        self.result_mode_label.configure(
            text=f"Logic: {operation}"
        )

        h, w = result.shape[:2]
        self.result_info_label.configure(
            text=f"{w} × {h}  |  {operation}"
        )

        self.overview_status.configure(
            text=f"Bitwise {operation} applied.{resized_note}"
        )

        self.logic_status_label.configure(
            text=f"✔ {operation} applied",
            text_color=GREEN
        )
        self.arithmetic_status_label.configure(
            text="Img1 op Img2 → Result",
            text_color=TEXT_DARK
        )

        self.status_var.set(
            f"Logic Operation : {operation}{resized_note}"
        )

        self.update_action_buttons()
        self.update_all_histograms()

    # COLOR TRANSFORMATION

    def update_alpha_label(self, val):
        self.blend_alpha_value.configure(text=f"{float(val):.2f}")

    def apply_arithmetic_operation(self):
        """Menerapkan operasi aritmatika (ADD, SUBTRACT, MULTIPLY, DIVIDE, BLEND) pada citra."""
        operation = self.arithmetic_mode_combo.get()

        if self.cv_img is None:
            messagebox.showwarning(
                "Peringatan",
                "Silahkan buka Image 1 terlebih dahulu."
            )
            return

        if self.cv_img_2 is None:
            messagebox.showwarning(
                "Peringatan",
                f"Silahkan buka Image 2 sebelum menerapkan operasi {operation}."
            )
            return

        try:
            img1 = self.cv_img.copy()
            img2 = self.cv_img_2.copy()

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            resized_note = ""
            if (h1, w1) != (h2, w2):
                img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)
                resized_note = f" (Image 2 resized: {w2}×{h2} → {w1}×{h1})"

            # Ensure 3-channel or 1-channel format (handle 4-channel BGRA)
            if len(img1.shape) == 3 and img1.shape[2] == 4:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
            if len(img2.shape) == 3 and img2.shape[2] == 4:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

            ch1 = img1.shape[2] if len(img1.shape) == 3 else 1
            ch2 = img2.shape[2] if len(img2.shape) == 3 else 1

            if ch1 != ch2:
                if ch1 == 1 and ch2 == 3:
                    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                elif ch1 == 3 and ch2 == 1:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

            if img1.dtype != np.uint8:
                img1 = np.clip(img1, 0, 255).astype(np.uint8)
            if img2.dtype != np.uint8:
                img2 = np.clip(img2, 0, 255).astype(np.uint8)

            alpha = 0.5
            if operation == "ADD":
                result = cv2.add(img1, img2)
            elif operation == "SUBTRACT":
                result = cv2.subtract(img1, img2)
            elif operation == "MULTIPLY":
                # Scale by 1/255 so (255 * 255) / 255 = 255, preserving uint8 range
                result = cv2.multiply(img1, img2, scale=1.0 / 255.0)
            elif operation == "DIVIDE":
                # Handle zero-valued denominator pixels safely: clamp 0 to 1 to avoid division by zero
                img2_safe = np.where(img2 == 0, 1, img2)
                result = cv2.divide(img1, img2_safe)
            elif operation == "BLEND":
                alpha = float(self.arithmetic_alpha_slider.get())
                beta = 1.0 - alpha
                result = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
            else:
                messagebox.showerror(
                    "Error",
                    f"Operasi tidak dikenal: {operation}"
                )
                return

            if result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Gagal melakukan operasi {operation}: {e}"
            )
            return

        self.base_result_img = result.copy()
        self.adjusted_img = None
        self.result_img = result.copy()

        self.brightness_slider.set(0)
        self.contrast_slider.set(1.0)
        self.brightness_value.configure(text="0")
        self.contrast_value.configure(text="1.0")

        self.render_label(
            result,
            self.lbl_result_overview
        )

        self.current_mode = f"Arithmetic: {operation}"
        self.result_mode_label.configure(
            text=f"Arithmetic: {operation}"
        )

        h, w = result.shape[:2]
        info_extra = f"{operation} (α={alpha:.2f})" if operation == "BLEND" else operation
        self.result_info_label.configure(
            text=f"{w} × {h}  |  {info_extra}"
        )

        self.overview_status.configure(
            text=f"Arithmetic {operation} applied.{resized_note}"
        )

        self.arithmetic_status_label.configure(
            text=f"✔ {operation} applied",
            text_color=GREEN
        )
        self.logic_status_label.configure(
            text="Img1 ⊕ Img2 → Result",
            text_color=TEXT_DARK
        )

        self.status_var.set(
            f"Arithmetic Operation : {operation}{resized_note}"
        )

        self.update_action_buttons()
        self.update_all_histograms()

    def convert_color(self, event=None):
        """Menerapkan transformasi warna pada citra."""

        if self.cv_img is None:
            messagebox.showwarning(
                "Peringatan",
                "Silahkan Buka Citra Terlebih Dahulu."
            )
            return

        mode = self.overview_mode_combo.get()
        self.current_mode = mode

        # GRAYSCALE
        if mode == "Grayscale":
            gray = cv2.cvtColor(
                self.cv_img,
                cv2.COLOR_BGR2GRAY
            )
            result = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR
            )

        # HSV CHANNELS
        # Audit fix: HSV tidak lagi BGR→HSV→BGR (round-trip yang tidak bermakna).
        # Sekarang menampilkan channel individual H, S, V.
        elif mode == "HSV • Hue":
            hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]  # Hue channel (0-179 di OpenCV)
            # Normalize to 0-255 for display
            hue_display = cv2.normalize(hue, None, 0, 255, cv2.NORM_MINMAX)
            result = cv2.applyColorMap(hue_display.astype(np.uint8), cv2.COLORMAP_HSV)

        elif mode == "HSV • Saturation":
            hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]  # Saturation channel
            result = cv2.cvtColor(saturation, cv2.COLOR_GRAY2BGR)

        elif mode == "HSV • Value":
            hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
            value = hsv[:, :, 2]  # Value channel (brightness)
            result = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)

        # YCrCb CHANNELS
        # Audit fix: YCrCb tidak lagi BGR→YCrCb→BGR (round-trip).
        # Sekarang menampilkan channel individual Y, Cr, Cb.
        elif mode == "YCrCb • Y (Luma)":
            ycrcb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb[:, :, 0]  # Luma
            result = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)

        elif mode == "YCrCb • Cr":
            ycrcb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2YCrCb)
            cr_channel = ycrcb[:, :, 1]  # Cr (red chrominance)
            result = cv2.cvtColor(cr_channel, cv2.COLOR_GRAY2BGR)

        elif mode == "YCrCb • Cb":
            ycrcb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2YCrCb)
            cb_channel = ycrcb[:, :, 2]  # Cb (blue chrominance)
            result = cv2.cvtColor(cb_channel, cv2.COLOR_GRAY2BGR)

        # RGB CHANNEL ISOLATION
        # OpenCV BGR indexing: 0=Blue, 1=Green, 2=Red
        elif mode == "Channel Red":
            result = self.cv_img.copy()
            result[:, :, 0] = 0  # Zero Blue
            result[:, :, 1] = 0  # Zero Green
            # Keep index 2 = Red

        elif mode == "Channel Green":
            result = self.cv_img.copy()
            result[:, :, 0] = 0  # Zero Blue
            result[:, :, 2] = 0  # Zero Red
            # Keep index 1 = Green

        elif mode == "Channel Blue":
            result = self.cv_img.copy()
            result[:, :, 1] = 0  # Zero Green
            result[:, :, 2] = 0  # Zero Red
            # Keep index 0 = Blue

        # CIE-LAB CHANNELS
        # Audit fix: LAB tidak lagi BGR→LAB→BGR (round-trip).
        # Sekarang menampilkan channel individual L, a, b.
        elif mode == "CIE-LAB • L":
            lab = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]  # Lightness
            result = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2BGR)

        elif mode == "CIE-LAB • a":
            lab = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2LAB)
            a_channel = lab[:, :, 1]  # Green-Red axis
            result = cv2.cvtColor(a_channel, cv2.COLOR_GRAY2BGR)

        elif mode == "CIE-LAB • b":
            lab = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2LAB)
            b_channel = lab[:, :, 2]  # Blue-Yellow axis
            result = cv2.cvtColor(b_channel, cv2.COLOR_GRAY2BGR)

        else:
            result = self.cv_img.copy()

        self.base_result_img = result.copy()
        self.adjusted_img = None
        self.result_img = result.copy()

        self.brightness_slider.set(0)
        self.contrast_slider.set(1.0)
        self.brightness_value.configure(text="0")
        self.contrast_value.configure(text="1.0")

        self.render_label(
            result,
            self.lbl_result_overview
        )

        self.overview_mode_combo.set(mode)

        self.result_mode_label.configure(
            text=f"Mode: {mode}"
        )

        # Update result info label
        h, w = result.shape[:2]
        ch = result.shape[2] if len(result.shape) == 3 else 1
        ch_str = "Grayscale" if ch == 1 else f"{ch}ch"
        self.result_info_label.configure(
            text=f"{w} × {h}  |  {mode}"
        )

        self.overview_status.configure(
            text=f"{mode} transformation applied."
        )

        self.logic_status_label.configure(
            text="Img1 ⊕ Img2 → Result",
            text_color=TEXT_DARK
        )
        self.arithmetic_status_label.configure(
            text="Img1 op Img2 → Result",
            text_color=TEXT_DARK
        )

        self.update_action_buttons()

        self.status_var.set(
            f"Mode Aktif : {mode}"
        )

        self.update_all_histograms()

    # IMAGE RENDER
    def render_label(
        self,
        cv_img,
        target_lbl
    ):
        """Render OpenCV image ke CTkLabel widget."""

        if cv_img is None:
            return

        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(
                cv_img,
                cv2.COLOR_GRAY2BGR
            )

        rgb = cv2.cvtColor(
            cv_img,
            cv2.COLOR_BGR2RGB
        )

        h, w = rgb.shape[:2]

        # Dynamic sizing based on label size
        max_width = max(target_lbl.winfo_width() - 10, 200)
        max_height = max(target_lbl.winfo_height() - 10, 200)

        # Fallback for initial render (sized for 3-column layout)
        if max_width < 50:
            max_width = 250
        if max_height < 50:
            max_height = 280

        scale = min(
            max_width / w,
            max_height / h,
            1.0
        )

        display_width = max(
            1,
            int(w * scale)
        )

        display_height = max(
            1,
            int(h * scale)
        )

        resized = cv2.resize(
            rgb,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA
        )

        pil_image = Image.fromarray(
            resized
        )

        img_tk = ImageTk.PhotoImage(
            image=pil_image
        )

        target_lbl.configure(
            image=img_tk,
            text=""
        )

        target_lbl.image = img_tk

if __name__ == "__main__":

    root = ctk.CTk()

    app = ColorSpaceApp(
        root
    )

    root.mainloop()