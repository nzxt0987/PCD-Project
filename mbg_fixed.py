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


#SCI-FI-THEME 
#THEME 
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


class ColorSpaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "PRD Image Processing"
        )

        self.root.geometry(
            "1280x800"
        )

        self.root.minsize(
            1100,
            700
        )

        self.root.configure(
            fg_color = BG_COLOR
        )

        # IMAGE STATE
        self.cv_img = None
        self.image_path = None
        self.result_img = None
        self.base_result_img = None
        self.adjusted_img = None

        # UI STATE
        self.nav_buttons = {}
        self.pages = {}
        self.recent_actions = []
        self.hist_canvas = None
        self.status_var = tk.StringVar(value="SYSTEM READY")

        self.setup_ui()
    
    def setup_ui(self):

        self.root.grid_rowconfigure(
            0,
            weight = 1
        )

        self.root.grid_columnconfigure(
            1,
            weight = 1
        )

        self.create_sidebar()

        self.main_area = ctk.CTkFrame(
            self.root,
            fg_color = BG_COLOR,
            corner_radius = 0
        )

        self.main_area.grid(
            row = 0,
            column = 1,
            sticky = "nsew"
        )

        self.main_area.grid_rowconfigure(
            1,
            weight = 1
        )

        self.main_area.grid_rowconfigure(
            2,
            weight = 0
        )

        self.main_area.grid_columnconfigure(
            0,
            weight = 1
        )

        # CONTAINER
        self.content_container = ctk.CTkFrame(
            self.main_area,
            fg_color = "transparent"
        )

        self.content_container.grid(
            row = 1,
            column = 0,
            sticky = "nsew",
            padx = 18,
            pady = (0, 10)
        )

        self.content_container.grid_rowconfigure(
            0,
            weight = 1
        )

        self.content_container.grid_columnconfigure(
            0,
            weight = 1
        )
        # HEADER
        self.create_header()

        # PAGES
        self.create_overview_page()
        self.create_analysis_page()

        self.create_footer()
        self.show_overview()
    def create_header(self):
        "Membuat Header Utama Pada Area Konten."

        self.header_frame = ctk.CTkFrame(
            self.main_area,
            fg_color = "transparent"
        )

        self.header_frame.grid(
            row = 0,
            column = 0,
            sticky = "ew",
            padx = 28,
            pady = (24, 10)
        )

        left_header = ctk.CTkFrame(
            self.header_frame,
            fg_color = "transparent"
        )
        left_header.pack(side = "left", fill = "x", expand = True)

        self.page_title = ctk.CTkLabel(
            left_header,
            text = "IMAGE PROCESSING",
            font = ctk.CTkFont(
                family = "Segoe UI",
                size = 26,
                weight = "bold" 
            ),
            text_color = TEXT
        )
        self.page_title.pack (anchor = "w")

        self.page_subtitle = ctk.CTkLabel(
            left_header,
            text = "Digital Image Analysis And Color Transformation",
            font = ctk.CTkFont(
                family = "Segoe UI",
                size = 12
            ),
            text_color = TEXT_MUTED
        )

        self.page_subtitle.pack(
            anchor = "w",
            pady = (4, 0)
        )

        right_header = ctk.CTkFrame(
            self.header_frame,
            fg_color = "transparent"
        )
        right_header.pack(side = "right")
        self.header_status = ctk.CTkLabel(
            right_header,
            text ="● SYSTEM ONLINE",
            font = ctk.CTkFont(
                family = "Consolas",
                size = 11,
                weight = "bold"
            ),
            text_color = GREEN
        )
        self.header_status.pack(
            anchor = "e",
            pady = (0, 4)
        )

        self.header_time = ctk.CTkLabel(
            right_header,
            text = datetime.now().strftime("%d %b %Y  •  %H:%M"),
            font = ctk.CTkFont(
                family = "Consolas",
                size = 10
            ),
            text_color = TEXT_MUTED
        )
        self.header_time.pack (anchor = "e")


    # SIDE BAR
    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self.root,
            width = 225,
            fg_color = SIDEBAR_COLOR,
            corner_radius = 0,
            border_width = 1,
            border_color = BORDER
        )
        self.sidebar.grid(
            row = 0,
            column = 0,
            sticky = "nsew"
        )
        self.sidebar.grid_propagate(
            False
        )

        # BRAND
        brand_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color = "transparent"
        )

        brand_frame.pack(
            fill = tk.X,
            padx = 18,
            pady = (24, 18)
        )

        brand_title = ctk.CTkLabel(
            brand_frame,
            text = "PRD",
            text_color = CYAN_BRIGHT,
            font = ctk.CTkFont(
                family = "consolas",
                size = 30,
                weight = "bold"
            )
        )

        brand_title.pack(
            anchor = "w"
        )

        brand_subtitle = ctk.CTkLabel(
            brand_frame,
            text = "IMAGE PROCESSING",
            text_color = TEXT,
            font = ctk.CTkFont(
                family = "Consolas",
                size = 11,
                weight = "bold"
            )
        )

        brand_subtitle.pack(
            anchor = "w"
        )

        version_label = ctk.CTkLabel(
            brand_frame,
            text = "DIGITAL VISION",
            text_color = TEXT_MUTED,
            font = ctk.CTkFont(
                family = "Consolas",
                size = 9
            )
        )

        version_label.pack(
            anchor = "w",
            pady = (2, 0)
        )

        # NAVIGATION
        nav_label = ctk.CTkLabel(
            self.sidebar,
            text = "NAVIGATION",
            text_color = TEXT_MUTED,
            font = ctk.CTkFont(
                family = "Consolas",
                size = 9,
                weight = "bold"
            )
        )

        nav_label.pack(
            anchor = "w",
            padx = 20,
            pady = (5, 0)
        )
        
        self.create_nav_button(
        "◉  OVERVIEW",
        "overview",
        self.show_overview
        )

        self.create_nav_button(
        "◈  ANALYSIS",
        "analysis",
        self.show_analysis
        )

        # SPACER
        spacer = ctk.CTkFrame(
            self.sidebar,
            fg_color = "transparent"
        )

        spacer.pack(
            fill = tk.BOTH,
            expand = True
        )

        # SESSION INFORMATION
        separator = ctk.CTkFrame(
            self.sidebar,
            height = 1,
            fg_color = BORDER
        )

        separator.pack(
            fill = tk.X,
            padx = 18,
            pady = (8, 8)
        )

        session_title = ctk.CTkLabel(
            self.sidebar,
            text = "SESSION",
            text_color = TEXT_MUTED,
            font = ctk.CTkFont(
                family = "Consolas",
                size = 9,
                weight = "bold"
            )
        )

        session_title.pack(
            anchor = "w",
            padx = 20,
            pady = (0, 4)
        )

        session_info = ctk.CTkLabel(
            self.sidebar,
            text = "Group  GOJEK\nSession  PCD-2026\nPython | OpenCV | CTk",
            text_color = TEXT_MUTED,
            justify = "left",
            anchor = "w",
            font = ctk.CTkFont(
                family = "Consolas",
                size = 9
            )
        )

        session_info.pack(
            fill = tk.X,
            padx = 20,
            pady = (0, 14)
        )
    
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
        height=46,
        corner_radius=0,
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
        fill = tk.X,
        padx = 8,
        pady = 2
        )

        self.nav_buttons[
        page_name
        ] = button

    def create_overview_page(self):
        """Membuat halaman Overview sebagai workspace utama."""

        self.pages["overview"] = ctk.CTkFrame(
            self.content_container,
            fg_color="transparent"
        )

        page = self.pages["overview"]

        page.grid_columnconfigure(0, weight=4)
        page.grid_columnconfigure(1, weight=4)
        page.grid_columnconfigure(2, weight=3)
        page.grid_rowconfigure(1, weight=1)
        page.grid_rowconfigure(2, weight=0)
        page.grid_rowconfigure(3, weight=0)

        # SECTION TITLE
        ctk.CTkLabel(
            page,
            text="IMAGE WORKSPACE",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=20,
                weight="bold"
            ),
            text_color=TEXT
        ).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            padx=28,
            pady=(8, 2)
        )

        ctk.CTkLabel(
            page,
            text="Load • Transform • Adjust • Preview",
            font=ctk.CTkFont(
                family="Consolas",
                size=10
            ),
            text_color=TEXT_MUTED
        ).grid(
            row=0,
            column=2,
            sticky="e",
            padx=28,
            pady=(8, 2)
        )

        # WORKSPACE
        workspace = ctk.CTkFrame(
            page,
            fg_color="transparent"
        )
        workspace.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="nsew",
            padx=28,
            pady=(8, 8)
        )

        workspace.grid_columnconfigure(0, weight=4)
        workspace.grid_columnconfigure(1, weight=4)
        workspace.grid_columnconfigure(2, weight=3)
        workspace.grid_rowconfigure(0, weight=1)

        # SOURCE IMAGE
        source_panel = ctk.CTkFrame(
            workspace,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        source_panel.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 6)
        )

        ctk.CTkLabel(
            source_panel,
            text="SOURCE IMAGE",
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

        self.lbl_orig_overview = ctk.CTkLabel(
            source_panel,
            text="No image loaded",
            fg_color="#050A0E",
            corner_radius=8,
            text_color=TEXT_MUTED
        )
        self.lbl_orig_overview.pack(
            fill="both",
            expand=True,
            padx=14,
            pady=(0, 10)
        )

        # RESULT IMAGE
        result_panel = ctk.CTkFrame(
            workspace,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        result_panel.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=6
        )

        ctk.CTkLabel(
            result_panel,
            text="RESULT / PREVIEW",
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            text_color=GREEN
        ).pack(
            anchor="w",
            padx=14,
            pady=(12, 6)
        )

        self.lbl_result_overview = ctk.CTkLabel(
            result_panel,
            text="Apply a transformation",
            fg_color="#050A0E",
            corner_radius=8,
            text_color=TEXT_MUTED
        )
        self.lbl_result_overview.pack(
            fill="both",
            expand=True,
            padx=14,
            pady=(0, 12)
        )

        self.result_mode_label = ctk.CTkLabel(
            result_panel,
            text="No transformation",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_MUTED
        )
        self.result_mode_label.pack(
            anchor="w",
            padx=14,
            pady=(0, 12)
        )

        # IMAGE INFORMATION
        info_panel = ctk.CTkFrame(
            workspace,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        info_panel.grid(
            row=0,
            column=2,
            sticky="nsew",
            padx=(6, 0)
        )

        ctk.CTkLabel(
            info_panel,
            text="INFORMATION",
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            anchor="w",
            padx=14,
            pady=(12, 8)
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
            padx=14,
            pady=(0, 10)
        )

        ctk.CTkFrame(
            info_panel,
            height=1,
            fg_color=BORDER
        ).pack(
            fill="x",
            padx=14,
            pady=4
        )

        ctk.CTkLabel(
            info_panel,
            text="WORKFLOW STATUS",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=TEXT_MUTED
        ).pack(
            anchor="w",
            padx=14,
            pady=(8, 4)
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
            padx=14,
            pady=(0, 10)
        )

        ctk.CTkLabel(
            info_panel,
            text="IMAGE ACTIONS",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=TEXT_MUTED
        ).pack(
            anchor="w",
            padx=14,
            pady=(2, 6)
        )

        self.open_button = ctk.CTkButton(
            info_panel,
            text="OPEN IMAGE",
            height=32,
            corner_radius=7,
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
        self.open_button.pack(
            fill="x",
            padx=14,
            pady=(0, 6)
        )

        self.remove_button = ctk.CTkButton(
            info_panel,
            text="REMOVE IMAGE",
            height=32,
            corner_radius=7,
            fg_color="#241015",
            hover_color="#3A1820",
            border_width=1,
            border_color=RED,
            text_color=RED,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.clear_image
        )
        self.remove_button.pack(
            fill="x",
            padx=14,
            pady=(0, 6)
        )

        self.save_button = ctk.CTkButton(
            info_panel,
            text="SAVE RESULT",
            height=32,
            corner_radius=7,
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
        self.save_button.pack(
            fill="x",
            padx=14,
            pady=(0, 14)
        )

        self.update_action_buttons()

        # QUICK TRANSFORM
        transform_panel = ctk.CTkFrame(
            page,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=10
        )
        transform_panel.grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=28,
            pady=(0, 6)
        )

        ctk.CTkLabel(
            transform_panel,
            text="QUICK TRANSFORM",
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            side="left",
            padx=(14, 8)
        )

        self.overview_mode_combo = ctk.CTkComboBox(
            transform_panel,
            values=[
                "Grayscale",
                "HSV",
                "HSV Value (V)",
                "YCrCb",
                "Channel Red",
                "Channel Green",
                "Channel Blue",
                "CIE-LAB"
            ],
            width=180,
            height=32,
            corner_radius=7,
            border_width=1,
            border_color=BORDER,
            button_color="#0A2027",
            button_hover_color="#10333C",
            fg_color="#050A0E",
            text_color=TEXT,
            dropdown_fg_color="#080D12",
            dropdown_hover_color="#0C252C",
            dropdown_text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            command=self.convert_color
        )
        self.overview_mode_combo.pack(
            side="left",
            padx=(0, 8)
        )
        self.overview_mode_combo.set("Grayscale")

        ctk.CTkButton(
            transform_panel,
            text="APPLY",
            width=85,
            height=32,
            corner_radius=7,
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
            side="left"
        )

        ctk.CTkLabel(
            transform_panel,
            text="Choose a color space or isolate an RGB channel.",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=10
            ),
            text_color=TEXT_MUTED
        ).pack(
            side="left",
            padx=14
        )

        # ADJUSTMENTS
        adjustment_panel = ctk.CTkFrame(
            page,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=10
        )
        adjustment_panel.grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=28,
            pady=(0, 8)
        )

        ctk.CTkLabel(
            adjustment_panel,
            text="ADJUSTMENTS",
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            text_color=GREEN
        ).pack(
            side="left",
            padx=(14, 12)
        )

        ctk.CTkLabel(
            adjustment_panel,
            text="Brightness",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_MUTED
        ).pack(
            side="left"
        )

        self.brightness_value = ctk.CTkLabel(
            adjustment_panel,
            text="0",
            width=35,
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT
        )
        self.brightness_value.pack(
            side="left"
        )

        self.brightness_slider = ctk.CTkSlider(
            adjustment_panel,
            from_=-100,
            to=100,
            number_of_steps=200,
            width=190,
            command=self.preview_adjustment
        )
        self.brightness_slider.set(0)
        self.brightness_slider.pack(
            side="left",
            padx=(0, 14)
        )

        ctk.CTkLabel(
            adjustment_panel,
            text="Contrast",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_MUTED
        ).pack(
            side="left"
        )

        self.contrast_value = ctk.CTkLabel(
            adjustment_panel,
            text="1.0",
            width=35,
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT
        )
        self.contrast_value.pack(
            side="left"
        )

        self.contrast_slider = ctk.CTkSlider(
            adjustment_panel,
            from_=0.2,
            to=2.0,
            number_of_steps=180,
            width=190,
            command=self.preview_adjustment
        )
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(
            side="left",
            padx=(0, 12)
        )

        ctk.CTkButton(
            adjustment_panel,
            text="RESET",
            width=72,
            height=30,
            corner_radius=7,
            fg_color="#211D0B",
            hover_color="#38300F",
            border_width=1,
            border_color="#66551A",
            text_color=YELLOW,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.reset_adjustment
        ).pack(
            side="right",
            padx=(4, 14)
        )

        ctk.CTkButton(
            adjustment_panel,
            text="APPLY",
            width=72,
            height=30,
            corner_radius=7,
            fg_color=CYAN_DIM,
            hover_color="#0D6E7B",
            text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            command=self.apply_adjustment
        ).pack(
            side="right",
            padx=4
        )


    def create_analysis_page(self):
        """Membuat Halaman Analysis."""

        self.pages["analysis"] = ctk.CTkFrame(
            self.content_container,
            fg_color="transparent"
        )

        page = self.pages["analysis"]

        ctk.CTkLabel(
            page,
            text="IMAGE ANALYSIS",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=20,
                weight="bold"
            ),
            text_color=TEXT
        ).pack(
            anchor="w",
            padx=28,
            pady=(10, 4)
        )

        ctk.CTkLabel(
            page,
            text="Inspect image information and pixel intensity distribution.",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=11
            ),
            text_color=TEXT_MUTED
        ).pack(
            anchor="w",
            padx=28,
            pady=(0, 12)
        )

        analysis_container = ctk.CTkFrame(
            page,
            fg_color="transparent"
        )
        analysis_container.pack(
            fill="both",
            expand=True,
            padx=28,
            pady=(0, 20)
        )

        info_panel = ctk.CTkFrame(
            analysis_container,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        info_panel.pack(
            fill="x",
            pady=(0, 10)
        )

        ctk.CTkLabel(
            info_panel,
            text="IMAGE INFORMATION",
            font=ctk.CTkFont(
                family="Consolas",
                size=11,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            anchor="w",
            padx=18,
            pady=(14, 8)
        )

        self.analysis_info = ctk.CTkLabel(
            info_panel,
            text="No image loaded.",
            font=ctk.CTkFont(
                family="Consolas",
                size=10
            ),
            text_color=TEXT_MUTED,
            justify="left"
        )
        self.analysis_info.pack(
            anchor="w",
            padx=18,
            pady=(0, 14)
        )

        histogram_panel = ctk.CTkFrame(
            analysis_container,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        histogram_panel.pack(
            fill="both",
            expand=True
        )

        ctk.CTkLabel(
            histogram_panel,
            text="HISTOGRAM",
            font=ctk.CTkFont(
                family="Consolas",
                size=11,
                weight="bold"
            ),
            text_color=GREEN
        ).pack(
            anchor="w",
            padx=18,
            pady=(14, 8)
        )

        self.histogram_area = ctk.CTkFrame(
            histogram_panel,
            fg_color="#050A0E",
            corner_radius=8
        )
        self.histogram_area.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(0, 18)
        )

        ctk.CTkLabel(
            self.histogram_area,
            text="Load an image to generate histogram.",
            font=ctk.CTkFont(
                family="Consolas",
                size=10
            ),
            text_color=TEXT_MUTED
        ).pack(
            expand=True
        )



    def create_footer(self):
        """Membuat Footer."""

        self.footer_frame = ctk.CTkFrame(
            self.main_area,
            fg_color="#050A0E",
            height=34,
            corner_radius=0,
            border_width=1,
            border_color=BORDER
        )

        self.footer_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=18,
            pady=(0, 10)
        )

        self.footer_frame.grid_propagate(False)

        ctk.CTkLabel(
            self.footer_frame,
            textvariable=self.status_var,
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_MUTED
        ).pack(
            side="left",
            padx=14
        )

        ctk.CTkLabel(
            self.footer_frame,
            text="PRD IMAGE PROCESSING  •  PCD-2026",
            font=ctk.CTkFont(
                family="Consolas",
                size=9
            ),
            text_color=TEXT_DARK
        ).pack(
            side="right",
            padx=14
        )

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

        self.analysis_info.configure(
            text=(
                f"File      : {file_name}\n"
                f"Resolution: {w} × {h} px\n"
                f"Channels  : {channels}\n"
                f"Data type : {self.cv_img.dtype}"
            )
        )

        for widget in self.histogram_area.winfo_children():
            widget.destroy()

        figure = Figure(
            figsize=(6, 3.2),
            dpi=90,
            facecolor="#050A0E"
        )

        axis = figure.add_subplot(111)
        axis.set_facecolor("#050A0E")

        if channels == 3:
            for index, label in enumerate(["Blue", "Green", "Red"]):
                histogram = cv2.calcHist(
                    [self.cv_img],
                    [index],
                    None,
                    [256],
                    [0, 256]
                )
                axis.plot(
                    histogram,
                    label=label,
                    linewidth=1
                )
        else:
            histogram = cv2.calcHist(
                [self.cv_img],
                [0],
                None,
                [256],
                [0, 256]
            )
            axis.plot(
                histogram,
                label="Gray",
                linewidth=1
            )

        axis.set_xlim([0, 256])
        axis.tick_params(
            colors=TEXT_MUTED,
            labelsize=8
        )
        axis.set_xlabel(
            "Intensity",
            color=TEXT_MUTED,
            fontsize=8
        )
        axis.set_ylabel(
            "Pixels",
            color=TEXT_MUTED,
            fontsize=8
        )

        for spine in axis.spines.values():
            spine.set_color(BORDER)

        axis.grid(
            alpha=0.15
        )

        legend = axis.legend(
            fontsize=8,
            facecolor="#080D12",
            edgecolor=BORDER
        )

        for label in legend.get_texts():
            label.set_color(TEXT_MUTED)

        figure.tight_layout()

        self.hist_canvas = FigureCanvasTkAgg(
            figure,
            master=self.histogram_area
        )
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

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

        adjusted = cv2.convertScaleAbs(
            source,
            alpha=contrast,
            beta=brightness
        )

        self.adjusted_img = adjusted
        self.result_img = adjusted
        self.update_action_buttons()

        if hasattr(self, "lbl_result_overview"):
            self.render_label(
                adjusted,
                self.lbl_result_overview
            )

        if hasattr(self, "result_mode_label"):
            mode_text = self.overview_mode_combo.get()
            self.result_mode_label.configure(
                text=f"Preview: {mode_text} | B {brightness:+d} | C {contrast:.1f}"
            )

        if hasattr(self, "overview_status"):
            self.overview_status.configure(
                text=f"Live preview • Brightness {brightness:+d} • Contrast {contrast:.1f}"
            )

        self.status_var.set(
            f"Preview Adjustment : Brightness {brightness:+d} | Contrast {contrast:.1f}"
        )

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

        adjusted = cv2.convertScaleAbs(
            source,
            alpha=contrast,
            beta=brightness
        )

        self.adjusted_img = adjusted
        self.result_img = adjusted
        self.update_action_buttons()

        if hasattr(self, "lbl_result_overview"):
            self.render_label(
                adjusted,
                self.lbl_result_overview
            )

        if hasattr(self, "result_mode_label"):
            mode_text = self.overview_mode_combo.get()
            self.result_mode_label.configure(
                text=f"{mode_text} • Brightness {brightness:+d} • Contrast {contrast:.1f}"
            )

        if hasattr(self, "overview_status"):
            self.overview_status.configure(
                text=f"Adjustment applied • B {brightness:+d} • C {contrast:.1f}"
            )

        self.status_var.set(
            f"Adjustment aktif : Brightness {brightness} | Contrast {contrast:.1f}"
        )

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
            if hasattr(self, "lbl_result_overview"):
                self.render_label(
                    self.result_img,
                    self.lbl_result_overview
                )

            if hasattr(self, "lbl_result"):
                self.render_label(
                    self.result_img,
                    self.lbl_result
                )

        if hasattr(self, "result_mode_label"):
            mode_text = self.overview_mode_combo.get()
            self.result_mode_label.configure(
                text=f"{mode_text} • Adjustment reset"
            )

        if hasattr(self, "overview_status"):
            self.overview_status.configure(
                text="Adjustment reset. Base transformation restored."
            )

        self.status_var.set(
            "Adjustment di-reset"
        )

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

    #PAGE NAVIGATOR
    def show_page(
        self,
        page_name
    ):
        for page in self.pages.values():
            page.grid_remove()

        self.pages[
            page_name
        ].grid(
            row = 0,
            column = 0,
            sticky = "nsew"
        )

        for name, button in self.nav_buttons.items():
            if name == page_name:
                button.configure(
                    fg_color = "#0B2D35",
                    text_color = CYAN_BRIGHT
                )

            else:
                button.configure(
                    fg_color = "transparent",
                    text_color = TEXT_MUTED
                )

    def show_overview(self):

        self.show_page(
            "overview"
        )
    
    
    def show_analysis(self):
        self.show_page(
            "analysis"
        )

        
    def update_action_buttons(self):
        """Mengatur status tombol sesuai kondisi workspace."""

        has_image = self.cv_img is not None
        has_result = self.result_img is not None

        if hasattr(self, "remove_button"):
            self.remove_button.configure(
                state="normal" if has_image else "disabled"
            )

        if hasattr(self, "save_button"):
            self.save_button.configure(
                state="normal" if has_result else "disabled"
            )

    def load_image(self):

        file_path = filedialog.askopenfilename(
            title = "Pilih Citra",
            filetypes = [
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

        if hasattr(self, "lbl_orig_overview"):
            self.render_label(
                self.cv_img,
                self.lbl_orig_overview
            )

        if hasattr(self, "lbl_result_overview"):
            self.lbl_result_overview.configure(
                image="",
                text="Apply a transformation"
            )
            self.lbl_result_overview.image = None

        self.status_var.set(
            f"Citra dimuat : {file_path}"
        )

        if hasattr(self, "overview_info"):
            height, width = image.shape[:2]
            self.overview_info.configure(
                text=(
                    f"File      : {os.path.basename(file_path)}\n"
                    f"Resolution: {width} × {height} px\n"
                    f"Channels  : {image.shape[2] if len(image.shape) == 3 else 1} (BGR)\n"
                    f"Data type : {image.dtype}"
                )
            )

        if hasattr(self, "overview_status"):
            self.overview_status.configure(
                text="Image loaded. Ready for transformation."
            )

        if hasattr(self, "result_mode_label"):
            self.result_mode_label.configure(
                text="No transformation"
            )

        self.update_analysis()


    def clear_image(self):
        """Menghapus gambar yang sedang aktif dari workspace."""

        self.cv_img = None
        self.image_path = None
        self.result_img = None
        self.base_result_img = None
        self.adjusted_img = None

        if hasattr(self, "lbl_orig_overview"):
            self.lbl_orig_overview.configure(
                image="",
                text="No image loaded"
            )
            self.lbl_orig_overview.image = None

        if hasattr(self, "lbl_result_overview"):
            self.lbl_result_overview.configure(
                image="",
                text="Apply a transformation"
            )
            self.lbl_result_overview.image = None

        if hasattr(self, "overview_mode_combo"):
            self.overview_mode_combo.set("Grayscale")

        if hasattr(self, "brightness_slider"):
            self.brightness_slider.set(0)

        if hasattr(self, "contrast_slider"):
            self.contrast_slider.set(1.0)

        if hasattr(self, "brightness_value"):
            self.brightness_value.configure(text="0")

        if hasattr(self, "contrast_value"):
            self.contrast_value.configure(text="1.0")

        if hasattr(self, "overview_info"):
            self.overview_info.configure(
                text="No image loaded."
            )

        if hasattr(self, "overview_status"):
            self.overview_status.configure(
                text="Ready for image processing."
            )

        if hasattr(self, "result_mode_label"):
            self.result_mode_label.configure(
                text="No transformation"
            )

        if hasattr(self, "analysis_info"):
            self.analysis_info.configure(
                text="No image loaded."
            )

        if hasattr(self, "histogram_area"):
            for widget in self.histogram_area.winfo_children():
                widget.destroy()

        self.hist_canvas = None

        self.update_action_buttons()

        self.status_var.set(
            "Ready - No image loaded"
        )


    def convert_color(self, event=None):

        if self.cv_img is None:
            messagebox.showwarning(  
                "Peringatan",
                "Silahkan Buka Citra Terlebih Dahulu."
            )
            return

        modes = [
            "Grayscale",
            "HSV",
            "HSV Value (V)",
            "YCrCb",
            "Channel Red",
            "Channel Green",
            "Channel Blue",
            "CIE-LAB"
        ]

        if isinstance(event, str) and event in modes:
            mode = event
        else:
            mode = self.overview_mode_combo.get()

        if mode == "Grayscale":

            gray = cv2.cvtColor(
                self.cv_img,
                cv2.COLOR_BGR2GRAY
            )

            result = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR
            )

        elif mode == "HSV":

            hsv = cv2.cvtColor(
                self.cv_img,
                cv2.COLOR_BGR2HSV
            )

            result = cv2.cvtColor(
                hsv,
                cv2.COLOR_HSV2BGR
            )

        elif mode == "HSV Value (V)":
            hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
            value = hsv[:, :, 2]
            result = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)

        elif mode == "YCrCb":

            ycrcb = cv2.cvtColor(
                self.cv_img,
                cv2.COLOR_BGR2YCrCb
            )

            result = cv2.cvtColor(
                ycrcb,
                cv2.COLOR_YCrCb2BGR
            )

        elif mode == "Channel Red":

            result = self.cv_img.copy()

            result[:, :, 0] = 0
            result[:, :, 1] = 0  

        elif mode == "Channel Green":

            result = self.cv_img.copy()

            result[:, :, 0] = 0
            result[:, :, 2] = 0  

        elif mode == "Channel Blue":

            result = self.cv_img.copy()  
            
            result[:, :, 1] = 0
            result[:, :, 2] = 0

        elif mode == "CIE-LAB":

            lab = cv2.cvtColor(
                self.cv_img,
                cv2.COLOR_BGR2LAB
            )

            result = cv2.cvtColor(
                lab,
                cv2.COLOR_LAB2BGR
            )

        else:

            result = self.cv_img.copy()

        self.base_result_img = result.copy()
        self.adjusted_img = None
        self.result_img = result.copy()
        self.update_action_buttons()

        if hasattr(self, "brightness_slider"):
            self.brightness_slider.set(0)
        if hasattr(self, "contrast_slider"):
            self.contrast_slider.set(1.0)
        if hasattr(self, "brightness_value"):
            self.brightness_value.configure(text="0")
        if hasattr(self, "contrast_value"):
            self.contrast_value.configure(text="1.0")

        if hasattr(self, "lbl_result_overview"):
            self.render_label(
                result,
                self.lbl_result_overview
            )

        if hasattr(self, "overview_mode_combo"):
            self.overview_mode_combo.set(mode)

        if hasattr(self, "result_mode_label"):
            self.result_mode_label.configure(
                text=f"Mode: {mode}"
            )

        if hasattr(self, "overview_status"):
            self.overview_status.configure(
                text=f"{mode} transformation applied."
            )

        self.status_var.set(
            f"Mode Aktif : {mode}"
        )


    
    def render_label(
        self,
        cv_img,
        target_lbl
    ):

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

        max_width = 400
        max_height = 450

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