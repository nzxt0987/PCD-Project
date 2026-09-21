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
        self.create_transform_page()
        self.create_analysis_page()
        self.create_adjustments_page()
        self.create_tools_page()

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
        "▣  TRANSFORM",
        "transform",
        self.show_transform
        )

        self.create_nav_button(
        "◈  ANALYSIS",
        "analysis",
        self.show_analysis
        )

        self.create_nav_button(
        "◇  ADJUSTMENTS",
        "adjustments",
        self.show_adjustments
        )

        self.create_nav_button(
        "⚙  TOOLS",
        "tools",
        self.show_tools
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

        #SYSTEM INFORMATION
        separator = ctk.CTkFrame(
            self.sidebar,
            height = 1,
            fg_color = BORDER
        )

        separator.pack(
            fill = tk.X,
            padx = 18,
            pady = 10
        )

        system_label = ctk.CTkLabel(
            self.sidebar,
            text = "SYSTEM",
            text_color = TEXT_MUTED,
            font = ctk.CTkFont(
                family = "Consolas",
                size = 9,
                weight = "bold"
            )
        )

        system_label.pack(
            anchor = "w",
            padx = 20
        )

        online_label = ctk.CTkLabel(
            self.sidebar,
            text = "●  ONLINE",
            text_color = GREEN,
            font = ctk.CTkFont(
                family = "Consolas",
                size = 11,
                weight = "bold"
            )
        )

        online_label.pack(
            anchor = "w",
            padx = 20,
            pady = (4, 5)
        )

        user_label = ctk.CTkLabel(
            self.sidebar,
            text = "Group \n GOJEK",
            text_color = TEXT_MUTED,
            justify = "left",
            anchor = "w",
            font = ctk.CTkFont(
                family = "Consolas",
                size = 9
            )
        )

        user_label.pack(
            fill = tk.X,
            padx = 20,
            pady = (8, 0)
        )

        session_label = ctk.CTkLabel(
        self.sidebar,
        text="Session\nPCD-2026",
        text_color=TEXT_MUTED,
        justify="left",
        anchor="w",
        font=ctk.CTkFont(
            family="Consolas",
            size=9
        )
        )

        session_label.pack(
        fill=tk.X,
        padx=20,
        pady=(8, 0)
        )


        environment_label = ctk.CTkLabel(
        self.sidebar,
        text="Environment\nPython | OpenCV | CTk",
        text_color=TEXT_MUTED,
        justify="left",
        anchor="w",
        font=ctk.CTkFont(
            family="Consolas",
            size=9
        )
        )

        environment_label.pack(
        fill=tk.X,
        padx=20,
        pady=(8, 18)
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
        "Membuat Halaman Overview"

        self.pages["overview"] = ctk.CTkFrame(
            self.content_container,
            fg_color = "transparent"
        )

        page = self.pages["overview"]

        hero = ctk.CTkFrame(
            page,
            fg_color = "#080D12",
            border_width = 1,
            border_color = BORDER,
            corner_radius = 14
        )

        hero.pack(
            fill = "x",
            padx = 28,
            pady = (10, 12)
        )

        hero_left = ctk.CTkFrame(
            hero,
            fg_color = "transparent"
        )

        hero_left.pack(
            side = "left",
            fill = "both",
            expand = True,
            padx = 24,
            pady = 22
        )

        ctk.CTkLabel(
            hero_left,
            text = "Explore the visual structure\nof digital images.",
            font = ctk.CTkFont(
                family = "Segoe UI",
                size = 25,
                weight = "bold"
            ),
            text_color = TEXT,
            justify = "left"
        ).pack(
            anchor = "w",
            pady = (7, 8)
        )

        ctk.CTkLabel(
            hero_left,
            text =(
                "Analyze color spaces, isolate RGB channels, "
                "and inspect image characteristics."
            ),
            font = ctk.CTkFont(
                family = "Segoe UI",
                size = 12
            ),
            text_color = TEXT_MUTED,
            justify = "left"
        ).pack(anchor = "w")

        hero_status = ctk.CTkFrame(
            hero,
            width=210,
            fg_color="#050A0E",
            border_width=1,
            border_color=BORDER,
            corner_radius=10
        )
        hero_status.pack(
            side="right",
            padx=22,
            pady=22,
            fill="y"
        )
        hero_status.pack_propagate(False)

        ctk.CTkLabel(
            hero_status,
            text="SESSION",
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            text_color=TEXT_MUTED
        ).pack(
            anchor="w",
            padx=16,
            pady=(16, 2)
        )

        ctk.CTkLabel(
            hero_status,
            text="PCD-2026",
            font=ctk.CTkFont(
                family="Consolas",
                size=18,
                weight="bold"
            ),
            text_color=CYAN_BRIGHT
        ).pack(
            anchor="w",
            padx=16
        )

        ctk.CTkFrame(
            hero_status,
            height=1,
            fg_color=BORDER
        ).pack(
            fill="x",
            padx=16,
            pady=12
        )

        ctk.CTkLabel(
            hero_status,
            text="ENVIRONMENT",
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            text_color=TEXT_MUTED
        ).pack(
            anchor="w",
            padx=16
        )

        ctk.CTkLabel(
            hero_status,
            text="Python  •  OpenCV  •  CTk",
            font=ctk.CTkFont(
                family="Consolas",
                size=10
            ),
            text_color=TEXT
        ).pack(
            anchor="w",
            padx=16,
            pady=(4, 16)
        )

        # SECTION TITLE
        

        ctk.CTkLabel(
            page,
            text="WORKSPACE",
            font=ctk.CTkFont(
                family="Consolas",
                size=11,
                weight="bold"
            ),
            text_color=TEXT_MUTED
        ).pack(
            anchor="w",
            padx=28,
            pady=(4, 8)
        )

        
        # WORKSPACE CARDS

        cards_frame = ctk.CTkFrame(
            page,
            fg_color="transparent"
        )
        cards_frame.pack(
            fill="x",
            padx=28
        )

        cards_frame.grid_columnconfigure(
            0,
            weight=1
        )
        cards_frame.grid_columnconfigure(
            1,
            weight=1
        )
        cards_frame.grid_columnconfigure(
            2,
            weight=1
        )


        # CARD 1 - TRANSFORM


        transform_card = ctk.CTkFrame(
            cards_frame,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        transform_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 6)
        )

        ctk.CTkLabel(
            transform_card,
            text="01",
            font=ctk.CTkFont(
                family="Consolas",
                size=11,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            anchor="w",
            padx=18,
            pady=(16, 4)
        )

        ctk.CTkLabel(
            transform_card,
            text="TRANSFORM",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=17,
                weight="bold"
            ),
            text_color=TEXT
        ).pack(
            anchor="w",
            padx=18
        )

        ctk.CTkLabel(
            transform_card,
            text=(
                "Convert images into different "
                "color representations."
            ),
            font=ctk.CTkFont(
                family="Segoe UI",
                size=11
            ),
            text_color=TEXT_MUTED,
            justify="left"
        ).pack(
            anchor="w",
            padx=18,
            pady=(5, 16)
        )

        ctk.CTkButton(
            transform_card,
            text="OPEN TRANSFORM",
            height=32,
            corner_radius=7,
            fg_color="#0A2027",
            hover_color="#10333C",
            border_width=1,
            border_color=CYAN_DIM,
            text_color=CYAN_BRIGHT,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            command=lambda: self.show_page("transform")
        ).pack(
            fill="x",
            padx=18,
            pady=(0, 18)
        )

        
        # CARD 2 - ANALYSIS
        

        analysis_card = ctk.CTkFrame(
            cards_frame,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        analysis_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=6
        )

        ctk.CTkLabel(
            analysis_card,
            text="02",
            font=ctk.CTkFont(
                family="Consolas",
                size=11,
                weight="bold"
            ),
            text_color=GREEN
        ).pack(
            anchor="w",
            padx=18,
            pady=(16, 4)
        )

        ctk.CTkLabel(
            analysis_card,
            text="ANALYSIS",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=17,
                weight="bold"
            ),
            text_color=TEXT
        ).pack(
            anchor="w",
            padx=18
        )

        ctk.CTkLabel(
            analysis_card,
            text=(
                "Inspect image channels, histogram "
                "and visual characteristics."
            ),
            font=ctk.CTkFont(
                family="Segoe UI",
                size=11
            ),
            text_color=TEXT_MUTED,
            justify="left"
        ).pack(
            anchor="w",
            padx=18,
            pady=(5, 16)
        )

        ctk.CTkButton(
            analysis_card,
            text="OPEN ANALYSIS",
            height=32,
            corner_radius=7,
            fg_color="#0B211A",
            hover_color="#10382B",
            border_width=1,
            border_color="#17664A",
            text_color=GREEN,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            command=lambda: self.show_page("analysis")
        ).pack(
            fill="x",
            padx=18,
            pady=(0, 18)
        )

        # CARD 3 - TOOLS

        tools_card = ctk.CTkFrame(
            cards_frame,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        tools_card.grid(
            row=0,
            column=2,
            sticky="nsew",
            padx=(6, 0)
        )

        ctk.CTkLabel(
            tools_card,
            text="03",
            font=ctk.CTkFont(
                family="Consolas",
                size=11,
                weight="bold"
            ),
            text_color=YELLOW
        ).pack(
            anchor="w",
            padx=18,
            pady=(16, 4)
        )

        ctk.CTkLabel(
            tools_card,
            text="TOOLS",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=17,
                weight="bold"
            ),
            text_color=TEXT
        ).pack(
            anchor="w",
            padx=18
        )

        ctk.CTkLabel(
            tools_card,
            text=(
                "Additional utilities for image "
                "processing workflow."
            ),
            font=ctk.CTkFont(
                family="Segoe UI",
                size=11
            ),
            text_color=TEXT_MUTED,
            justify="left"
        ).pack(
            anchor="w",
            padx=18,
            pady=(5, 16)
        )

        ctk.CTkButton(
            tools_card,
            text="OPEN TOOLS",
            height=32,
            corner_radius=7,
            fg_color="#211D0B",
            hover_color="#38300F",
            border_width=1,
            border_color="#66551A",
            text_color=YELLOW,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            command=lambda: self.show_page("tools")
        ).pack(
            fill="x",
            padx=18,
            pady=(0, 18)
        )

    def create_transform_page(self):
        """Membuat Halaman Transform."""

        self.pages["transform"] = ctk.CTkFrame(
            self.content_container,
            fg_color="transparent"
        )

        page = self.pages["transform"]

        page.grid_columnconfigure(0, weight=1)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(2, weight=1)

        # TRANSFORM HEADER
        ctk.CTkLabel(
            page,
            text="COLOR TRANSFORMATION",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=20,
                weight="bold"
            ),
            text_color=TEXT
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            padx=28,
            pady=(10, 4)
        )

        ctk.CTkLabel(
            page,
            text="Load an image and select a color space or RGB channel.",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=11
            ),
            text_color=TEXT_MUTED
        ).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            padx=28,
            pady=(0, 4)
        )

        # INPUT IMAGE
        original_panel = ctk.CTkFrame(
            page,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        original_panel.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=(28, 8),
            pady=(12, 12)
        )

        ctk.CTkLabel(
            original_panel,
            text="INPUT IMAGE",
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

        self.lbl_orig = ctk.CTkLabel(
            original_panel,
            text="No image loaded",
            width=400,
            height=330,
            fg_color="#050A0E",
            corner_radius=8,
            text_color=TEXT_MUTED
        )
        self.lbl_orig.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(0, 12)
        )

        image_button_frame = ctk.CTkFrame(
            original_panel,
            fg_color="transparent"
        )
        image_button_frame.pack(
            fill="x",
            padx=18,
            pady=(0, 16)
        )

        ctk.CTkButton(
            image_button_frame,
            text="OPEN IMAGE",
            height=36,
            corner_radius=7,
            fg_color="#0A2027",
            hover_color="#10333C",
            border_width=1,
            border_color=CYAN_DIM,
            text_color=CYAN_BRIGHT,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            command=self.load_image
        ).pack(
            side="left",
            fill="x",
            expand=True
        )

        ctk.CTkButton(
            image_button_frame,
            text="REMOVE",
            width=95,
            height=36,
            corner_radius=7,
            fg_color="#241015",
            hover_color="#3A1820",
            border_width=1,
            border_color=RED,
            text_color=RED,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            command=self.clear_image
        ).pack(
            side="left",
            padx=(8, 0)
        )

        # OUTPUT IMAGE
        result_panel = ctk.CTkFrame(
            page,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        result_panel.grid(
            row=2,
            column=1,
            sticky="nsew",
            padx=(8, 28),
            pady=(12, 12)
        )

        ctk.CTkLabel(
            result_panel,
            text="TRANSFORMATION OUTPUT",
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

        self.lbl_result = ctk.CTkLabel(
            result_panel,
            text="Pilih Mode Transformasi",
            width=400,
            height=330,
            fg_color="#050A0E",
            corner_radius=8,
            text_color=TEXT_MUTED
        )
        self.lbl_result.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(0, 12)
        )

        # TRANSFORMATION CONTROL
        control_frame = ctk.CTkFrame(
            result_panel,
            fg_color="transparent"
        )
        control_frame.pack(
            fill="x",
            padx=18,
            pady=(0, 16)
        )

        ctk.CTkLabel(
            control_frame,
            text="MODE",
            font=ctk.CTkFont(
                family="Consolas",
                size=9,
                weight="bold"
            ),
            text_color=TEXT_MUTED
        ).pack(
            side="left",
            padx=(0, 10)
        )

        self.mode_combo = ctk.CTkComboBox(
            control_frame,
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
            width=190,
            height=34,
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
                size=10
            ),
            command=self.convert_color
        )
        self.mode_combo.pack(
            side="left"
        )

        self.mode_combo.set("Grayscale")

        ctk.CTkButton(
            control_frame,
            text="APPLY",
            width=90,
            height=34,
            corner_radius=7,
            fg_color=CYAN_DIM,
            hover_color="#0D6E7B",
            text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            command=self.convert_color
        ).pack(
            side="right"
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

    def create_adjustments_page(self):
        """Membuat Halaman Adjustments."""

        self.pages["adjustments"] = ctk.CTkFrame(
            self.content_container,
            fg_color="transparent"
        )

        page = self.pages["adjustments"]

        ctk.CTkLabel(
            page,
            text="IMAGE ADJUSTMENTS",
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
            text="Additional image controls for brightness and contrast.",
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

        panel = ctk.CTkFrame(
            page,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        panel.pack(
            fill="x",
            padx=28,
            pady=10
        )

        ctk.CTkLabel(
            panel,
            text="BRIGHTNESS / CONTRAST",
            font=ctk.CTkFont(
                family="Consolas",
                size=11,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            anchor="w",
            padx=18,
            pady=(16, 12)
        )

        brightness_frame = ctk.CTkFrame(
            panel,
            fg_color="transparent"
        )
        brightness_frame.pack(
            fill="x",
            padx=18,
            pady=6
        )

        ctk.CTkLabel(
            brightness_frame,
            text="Brightness",
            width=100,
            anchor="w",
            text_color=TEXT_MUTED
        ).pack(side="left")

        self.brightness_value = ctk.CTkLabel(
            brightness_frame,
            text="0",
            width=45,
            text_color=TEXT
        )
        self.brightness_value.pack(side="right")

        self.brightness_slider = ctk.CTkSlider(
            brightness_frame,
            from_=-100,
            to=100,
            number_of_steps=200,
            command=lambda value: self.brightness_value.configure(
                text=str(int(value))
            )
        )
        self.brightness_slider.set(0)
        self.brightness_slider.pack(
            side="left",
            fill="x",
            expand=True,
            padx=10
        )

        contrast_frame = ctk.CTkFrame(
            panel,
            fg_color="transparent"
        )
        contrast_frame.pack(
            fill="x",
            padx=18,
            pady=6
        )

        ctk.CTkLabel(
            contrast_frame,
            text="Contrast",
            width=100,
            anchor="w",
            text_color=TEXT_MUTED
        ).pack(side="left")

        self.contrast_value = ctk.CTkLabel(
            contrast_frame,
            text="1.0",
            width=45,
            text_color=TEXT
        )
        self.contrast_value.pack(side="right")

        self.contrast_slider = ctk.CTkSlider(
            contrast_frame,
            from_=0.2,
            to=2.0,
            number_of_steps=180,
            command=lambda value: self.contrast_value.configure(
                text=f"{value:.1f}"
            )
        )
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(
            side="left",
            fill="x",
            expand=True,
            padx=10
        )

        button_frame = ctk.CTkFrame(
            panel,
            fg_color="transparent"
        )
        button_frame.pack(
            fill="x",
            padx=18,
            pady=(12, 18)
        )

        ctk.CTkButton(
            button_frame,
            text="APPLY ADJUSTMENT",
            height=34,
            corner_radius=7,
            fg_color=CYAN_DIM,
            hover_color="#0D6E7B",
            text_color=TEXT,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            command=self.apply_adjustment
        ).pack(
            side="left"
        )

        ctk.CTkButton(
            button_frame,
            text="RESET",
            width=90,
            height=34,
            corner_radius=7,
            fg_color="#211D0B",
            hover_color="#38300F",
            border_width=1,
            border_color="#66551A",
            text_color=YELLOW,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            command=self.reset_adjustment
        ).pack(
            side="right"
        )

    def create_tools_page(self):
        """Membuat Halaman Tools."""

        self.pages["tools"] = ctk.CTkFrame(
            self.content_container,
            fg_color="transparent"
        )

        page = self.pages["tools"]

        ctk.CTkLabel(
            page,
            text="TOOLS",
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
            text="Utility functions for the image processing workflow.",
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

        tools_container = ctk.CTkFrame(
            page,
            fg_color="transparent"
        )
        tools_container.pack(
            fill="x",
            padx=28,
            pady=10
        )

        save_panel = ctk.CTkFrame(
            tools_container,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        save_panel.pack(
            fill="x",
            pady=(0, 10)
        )

        ctk.CTkLabel(
            save_panel,
            text="IMAGE EXPORT",
            font=ctk.CTkFont(
                family="Consolas",
                size=11,
                weight="bold"
            ),
            text_color=CYAN
        ).pack(
            anchor="w",
            padx=18,
            pady=(16, 5)
        )

        ctk.CTkLabel(
            save_panel,
            text="Save the current transformation result as an image.",
            font=ctk.CTkFont(
                family="Segoe UI",
                size=11
            ),
            text_color=TEXT_MUTED
        ).pack(
            anchor="w",
            padx=18,
            pady=(0, 10)
        )

        ctk.CTkButton(
            save_panel,
            text="SAVE RESULT",
            height=34,
            corner_radius=7,
            fg_color="#0A2027",
            hover_color="#10333C",
            border_width=1,
            border_color=CYAN_DIM,
            text_color=CYAN_BRIGHT,
            font=ctk.CTkFont(
                family="Consolas",
                size=10,
                weight="bold"
            ),
            command=self.save_result
        ).pack(
            anchor="w",
            padx=18,
            pady=(0, 16)
        )

        info_panel = ctk.CTkFrame(
            tools_container,
            fg_color="#080D12",
            border_width=1,
            border_color=BORDER,
            corner_radius=12
        )
        info_panel.pack(
            fill="x"
        )

        ctk.CTkLabel(
            info_panel,
            text="SESSION STATUS",
            font=ctk.CTkFont(
                family="Consolas",
                size=11,
                weight="bold"
            ),
            text_color=GREEN
        ).pack(
            anchor="w",
            padx=18,
            pady=(16, 5)
        )

        self.tools_status = ctk.CTkLabel(
            info_panel,
            text="No image loaded.",
            font=ctk.CTkFont(
                family="Consolas",
                size=10
            ),
            text_color=TEXT_MUTED,
            justify="left"
        )
        self.tools_status.pack(
            anchor="w",
            padx=18,
            pady=(0, 16)
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

        adjusted = cv2.convertScaleAbs(
            self.cv_img,
            alpha=contrast,
            beta=brightness
        )

        self.result_img = adjusted

        self.render_label(
            adjusted,
            self.lbl_result
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

        if self.cv_img is not None:
            self.render_label(
                self.cv_img,
                self.lbl_result
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
    
    def show_transform(self):

        self.show_page(
            "transform"
        )
    
    def show_analysis(self):
        self.show_page(
            "analysis"
        )
    def show_adjustments(self):

        self.show_page(
            "adjustments"
        )

    def show_tools(self):

        self.show_page(
            "tools"
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

        self.render_label(
            self.cv_img,
            self.lbl_orig
        )

        self.lbl_result.configure(
            image = "",
            text = "Pilih Mode Transformasi"
        )
        
        self.lbl_result.image = None

        self.status_var.set(
            f"Citra dimuat : {file_path}"
        )

        self.update_analysis()

        if hasattr(self, "tools_status"):
            self.tools_status.configure(
                text=(
                    f"File      : {os.path.basename(file_path)}\n"
                    f"Resolution: {image.shape[1]} × {image.shape[0]} px"
                )
            )


    def clear_image(self):
        """Menghapus gambar yang sedang aktif dari workspace."""

        self.cv_img = None
        self.image_path = None
        self.result_img = None

        self.lbl_orig.configure(
            image="",
            text="No image loaded"
        )
        self.lbl_orig.image = None

        self.lbl_result.configure(
            image="",
            text="Pilih Mode Transformasi"
        )
        self.lbl_result.image = None

        if hasattr(self, "mode_combo"):
            self.mode_combo.set("Grayscale")

        if hasattr(self, "brightness_slider"):
            self.brightness_slider.set(0)

        if hasattr(self, "contrast_slider"):
            self.contrast_slider.set(1.0)

        if hasattr(self, "brightness_value"):
            self.brightness_value.configure(text="0")

        if hasattr(self, "contrast_value"):
            self.contrast_value.configure(text="1.0")

        if hasattr(self, "analysis_info"):
            self.analysis_info.configure(
                text="No image loaded."
            )

        if hasattr(self, "histogram_area"):
            for widget in self.histogram_area.winfo_children():
                widget.destroy()

        self.hist_canvas = None

        if hasattr(self, "tools_status"):
            self.tools_status.configure(
                text="No image loaded."
            )

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

        mode = self.mode_combo.get()

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

        self.result_img = result

        self.render_label(
            result,
            self.lbl_result
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