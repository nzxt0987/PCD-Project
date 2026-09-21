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
            self.content_container,
            fg_color = "transparent"
        )

        self.header_frame.pack(
            fill = "x",
            padx = 28,
            pady = (24,10),
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

        self.pages["Overview"] = ctk.CTkFrame(
            self.content_container,
            fg_color = "transparent"
        )

        page = self.pages["Overview"]

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
            command=lambda: self.show_page("Transform")
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
            command=lambda: self.show_page("Analysis")
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
            command=lambda: self.show_page("Tools")
        ).pack(
            fill="x",
            padx=18,
            pady=(0, 18)
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
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp"
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
            F"Citra dimuat : {file_path}"
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

            result = cv2.cvtColor(
                self.cv_img,
                cv2.COLOR_BGR2HSV
            )

        elif mode == "HSV Value (V)":
            hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
            value = hsv[:, :, 2]
            result = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)

        elif mode == "YCrCb":

            result = cv2.cvtColor(
                self.cv_img,
                cv2.COLOR_BGR2YCrCb
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

            result = cv2.cvtColor(
                self.cv_img,
                cv2.COLOR_BGR2LAB
            )

        else:

            result = self.cv_img.copy()

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