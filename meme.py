from tkinter import Message
from logging import root
import tkinter as tk
from tkinter import filedialog, ttk, messagebox  

import cv2
import numpy as np

from PIL import Image, ImageTk


class ColorSpaceApp:
    def __init__(self, root):

        self.root = root  

        self.root.title(
            "TUGAS NPD -  Pelengseran WOWO"
        )

        self.root.geometry(
            "1000x700"
        )

        self.cv_img = None

        self.image_path = None

        self.setup_ui()


    def setup_ui(self):

        header = tk.Frame(
            self.root
        )

        header.pack(
            fill=tk.X,
            pady=(20, 10)
        )

        title_label = tk.Label(
            header,
            text="KONVERSI FILTER GAMBAR INSTAGRAM 😹 🗿",
            font=("Times New Roman", 22, "bold")
        )

        title_label.pack()

        control_frame = tk.Frame(
            self.root
        )

        control_frame.pack(
            fill=tk.X,
            padx=20,
            pady=10
        )

        open_button = tk.Button(
            control_frame,
            text="Buka GAMBAR AJG",
            command=self.load_image,
            bg="#3498db",
            fg="white",
            font=("Arial", 12)
        )

        open_button.pack(
            side=tk.LEFT,
        )

        mode_label = tk.Label(
            control_frame,
            text="INI FILTER YA AJG : "
        )

        mode_label.pack(
            side=tk.LEFT,
            padx=(30, 10)
        )

        self.mode_var = tk.StringVar(
            value="Grayscale"
        )

        self.mode_combo = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            state="readonly",
            width=20,
            values=[
                "SADBOY",
                "HSV",
                "YCrCb",
                "FILTER SIKSA KUBUR",
                "WHANGSAF",
                "Channel Blue",
                "CIE-LAB"
            ]
        )

        self.mode_combo.pack(
            side=tk.LEFT
        )

        self.mode_combo.bind(
            "<<ComboboxSelected>>",
            self.convert_color
        )

        body = tk.Frame(
            self.root
        )

        body.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=15
        )

        left_frame = tk.LabelFrame(
            body,
            text="GAMBAR AI (ASLI INI)"
        )

        left_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(0, 10)
        )

        self.lbl_orig = tk.Label( 
            left_frame,
            text="Pilih gambar terlebih dahulu"
        )

        self.lbl_orig.pack(
            fill=tk.BOTH,
            expand=True
        )

        right_frame = tk.LabelFrame(
            body,
            text="OPTIMUM PRIDE UR UR AE AE"
        )

        right_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
            padx=(10, 0)
        )

        self.lbl_result = tk.Label(
            right_frame,
            text="Belum diproses"
        )

        self.lbl_result.pack(
            fill=tk.BOTH,
            expand=True
        )

        self.status_var = tk.StringVar(
            value="BELUM ADA GAMBAR UPLOAD DULU KOCAK"
        )

        status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w"
        )

        status_label.pack(
            fill=tk.X,
            padx=20,
            pady=(0, 10)
        )


    
    def load_image(self,):

        file_path = filedialog.askopenfilename(
            title="Pilih Citra",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp"
                )
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
                "Gambar gagal dibaca."
            )
            return

        self.cv_img = image
        self.image_path = file_path

        self.render_label(
            self.cv_img,
            self.lbl_orig
        )

        self.lbl_result.config(
            image="",
            text="Pilih Mode Transformasi."
        )

        self.lbl_result.image = None

        self.status_var.set(
            f"Citra dimuat : {file_path}"
        )



    def convert_color(self, event=None):

        if self.cv_img is None:
            messagebox.showwarning(  
                "Peringatan",
                "Silahkan Buka Citra Terlebih Dahulu."
            )
            return

        mode = self.mode_var.get()

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

        elif mode == "YCrCb":

            result = cv2.cvtColor(
                self.cv_img,
                cv2.COLOR_BGR2YCrCb
            )

        elif mode == "FILTER SIKSA KUBUR":

            result = self.cv_img.copy()

            result[:, :, 0] = 0
            result[:, :, 1] = 0  

        elif mode == "WHANGSAF":

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

        target_lbl.config(
            image=img_tk,
            text=""
        )

        target_lbl.image = img_tk 



if __name__ == "__main__":

    root = tk.Tk()

    app = ColorSpaceApp(
        root
    )

    root.mainloop()