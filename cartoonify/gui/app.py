"""
Professional Modern Tkinter GUI for Cartoonify Studio.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from cartoonify.engine import STYLES, CartoonEngine
from cartoonify.gui.theme import THEMES
from cartoonify.utils import HistoryManager, load_image, resize_keep_aspect, save_image

logger = logging.getLogger("cartoonify.gui")


class ModernCartoonifierApp:
    """Desktop GUI application with modern styling, async worker threads, and comparison views."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Cartoonify Studio Pro v2.0")
        self.root.geometry("1400x860")
        self.root.minsize(1050, 680)

        self.current_theme_name = "dark"
        self.theme = THEMES[self.current_theme_name]

        self.engine = CartoonEngine()
        self.history_manager = HistoryManager()

        # Application state
        self.input_bgr: Optional[np.ndarray] = None
        self.output_bgr: Optional[np.ndarray] = None
        self.current_style_key = "ghibli_pro"
        self.is_processing = False
        self.current_filepath: Optional[Path] = None

        self._init_ttk_styles()
        self._build_ui()
        self._apply_theme()

    def _init_ttk_styles(self):
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

    def _apply_theme(self):
        t = self.theme
        self.root.configure(bg=t["bg_dark"])

        self.style.configure(
            "TNotebook",
            background=t["bg_dark"],
            borderwidth=0,
        )
        self.style.configure(
            "TNotebook.Tab",
            background=t["bg_card"],
            foreground=t["text_primary"],
            padding=[16, 8],
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", t["primary"])],
            foreground=[("selected", "#ffffff")],
        )

        self.style.configure(
            "TProgressbar",
            troughcolor=t["slider_trough"],
            background=t["primary"],
            thickness=6,
        )

    def _build_ui(self):
        t = self.theme

        # Top Header Bar
        header = tk.Frame(self.root, bg=t["bg_panel"], height=64, padx=20, pady=10)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        brand_frame = tk.Frame(header, bg=t["bg_panel"])
        brand_frame.pack(side=tk.LEFT, fill=tk.Y)

        title_lbl = tk.Label(
            brand_frame,
            text="🎨 Cartoonify Studio Pro",
            font=("Segoe UI", 14, "bold"),
            bg=t["bg_panel"],
            fg=t["text_primary"],
        )
        title_lbl.pack(side=tk.LEFT)

        ver_lbl = tk.Label(
            brand_frame,
            text=" v2.0",
            font=("Segoe UI", 10),
            bg=t["bg_panel"],
            fg=t["text_muted"],
        )
        ver_lbl.pack(side=tk.LEFT, padx=(4, 0))

        # Right Header Actions
        hdr_actions = tk.Frame(header, bg=t["bg_panel"])
        hdr_actions.pack(side=tk.RIGHT, fill=tk.Y)

        self.theme_btn = tk.Button(
            hdr_actions,
            text="🌙 Theme",
            command=self._toggle_theme,
            font=("Segoe UI", 9),
            bg=t["bg_card"],
            fg=t["text_primary"],
            activebackground=t["bg_input"],
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
        )
        self.theme_btn.pack(side=tk.RIGHT)

        # Main Workspace (Split left controls, right canvas)
        main_workspace = tk.Frame(self.root, bg=t["bg_dark"], padx=14, pady=14)
        main_workspace.pack(fill=tk.BOTH, expand=True)

        # Left Control Panel
        left_panel = tk.Frame(main_workspace, bg=t["bg_panel"], width=420, padx=16, pady=16)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        left_panel.pack_propagate(False)

        self._build_left_panel(left_panel)

        # Right Viewport Panel
        right_panel = tk.Frame(main_workspace, bg=t["bg_panel"], padx=14, pady=14)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_right_panel(right_panel)

        # Bottom Status Bar
        status_bar = tk.Frame(self.root, bg=t["bg_input"], height=30, padx=14)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)

        self.status_lbl = tk.Label(
            status_bar,
            text="Ready • Load an image to start",
            font=("Segoe UI", 9),
            bg=t["bg_input"],
            fg=t["text_secondary"],
        )
        self.status_lbl.pack(side=tk.LEFT)

        self.prog_bar = ttk.Progressbar(status_bar, mode="indeterminate", length=140)
        self.prog_bar.pack(side=tk.RIGHT, pady=4)

    def _build_left_panel(self, parent: tk.Frame):
        t = self.theme

        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        # --- Tab 1: Single Transform ---
        single_tab = tk.Frame(notebook, bg=t["bg_panel"], pady=10)
        notebook.add(single_tab, text="Single Image")

        # Load Buttons
        btn_box = tk.Frame(single_tab, bg=t["bg_panel"])
        btn_box.pack(fill=tk.X, pady=(0, 12))

        load_btn = tk.Button(
            btn_box,
            text="📁 Open Image",
            command=self._on_open_image,
            font=("Segoe UI", 10, "bold"),
            bg=t["primary"],
            fg="#ffffff",
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
        )
        load_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        sample_btn = tk.Button(
            btn_box,
            text="✨ Sample",
            command=self._on_load_sample,
            font=("Segoe UI", 9),
            bg=t["bg_card"],
            fg=t["text_primary"],
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
        )
        sample_btn.pack(side=tk.LEFT, padx=(4, 0))

        # Styles Listbox / Radio Selection
        tk.Label(
            single_tab,
            text="CHOOSE CARTOON STYLE",
            font=("Segoe UI", 8, "bold"),
            bg=t["bg_panel"],
            fg=t["text_muted"],
        ).pack(anchor=tk.W, pady=(6, 4))

        self.style_var = tk.StringVar(value="ghibli_pro")
        styles_scroll_frame = tk.Frame(single_tab, bg=t["bg_input"], height=220)
        styles_scroll_frame.pack(fill=tk.X, pady=(0, 10))
        styles_scroll_frame.pack_propagate(False)

        style_canvas = tk.Canvas(styles_scroll_frame, bg=t["bg_input"], highlightthickness=0)
        scrollbar = tk.Scrollbar(styles_scroll_frame, orient="vertical", command=style_canvas.yview)
        scrollable_frame = tk.Frame(style_canvas, bg=t["bg_input"])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: style_canvas.configure(scrollregion=style_canvas.bbox("all")),
        )
        style_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        style_canvas.configure(yscrollcommand=scrollbar.set)

        style_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for key, s in STYLES.items():
            rb = tk.Radiobutton(
                scrollable_frame,
                text=f"{s.icon}  {s.name}",
                variable=self.style_var,
                value=key,
                command=self._on_style_changed,
                bg=t["bg_input"],
                fg=t["text_primary"],
                selectcolor=t["bg_panel"],
                activebackground=t["bg_input"],
                activeforeground=t["primary"],
                font=("Segoe UI", 9),
                anchor=tk.W,
            )
            rb.pack(fill=tk.X, padx=8, pady=3)

        # Style Description Box
        self.desc_lbl = tk.Label(
            single_tab,
            text=STYLES["ghibli_pro"].description,
            font=("Segoe UI", 8),
            bg=t["bg_card"],
            fg=t["text_secondary"],
            wraplength=360,
            justify=tk.LEFT,
            padx=10,
            pady=8,
        )
        self.desc_lbl.pack(fill=tk.X, pady=(0, 10))

        # Strength Slider
        slider_frame = tk.Frame(single_tab, bg=t["bg_panel"])
        slider_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            slider_frame,
            text="INTENSITY STRENGTH",
            font=("Segoe UI", 8, "bold"),
            bg=t["bg_panel"],
            fg=t["text_muted"],
        ).pack(side=tk.LEFT)

        self.strength_val_lbl = tk.Label(
            slider_frame,
            text="85%",
            font=("Segoe UI", 9, "bold"),
            bg=t["bg_panel"],
            fg=t["primary"],
        )
        self.strength_val_lbl.pack(side=tk.RIGHT)

        self.strength_scale = tk.Scale(
            single_tab,
            from_=10,
            to=100,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg=t["bg_panel"],
            troughcolor=t["bg_input"],
            highlightthickness=0,
            command=self._on_strength_slider,
        )
        self.strength_scale.set(85)
        self.strength_scale.pack(fill=tk.X, pady=(0, 8))

        # Face Align Checkbox
        self.face_align_var = tk.BooleanVar(value=False)
        self.face_chk = tk.Checkbutton(
            single_tab,
            text="Face-Preserved Blend (Portrait Alignment)",
            variable=self.face_align_var,
            bg=t["bg_panel"],
            fg=t["text_primary"],
            selectcolor=t["bg_input"],
            activebackground=t["bg_panel"],
            font=("Segoe UI", 9),
        )
        self.face_chk.pack(anchor=tk.W, pady=(0, 14))

        # Generate Button
        self.gen_btn = tk.Button(
            single_tab,
            text="✨ Generate Cartoon",
            command=self._on_generate,
            font=("Segoe UI", 11, "bold"),
            bg=t["success"],
            fg="#ffffff",
            activebackground=t["success_hover"],
            relief=tk.FLAT,
            pady=10,
            cursor="hand2",
        )
        self.gen_btn.pack(fill=tk.X, pady=(0, 6))

        # Save Button
        self.save_btn = tk.Button(
            single_tab,
            text="💾 Save Result",
            command=self._on_save_image,
            font=("Segoe UI", 10),
            bg=t["bg_card"],
            fg=t["text_primary"],
            relief=tk.FLAT,
            pady=8,
            state=tk.DISABLED,
            cursor="hand2",
        )
        self.save_btn.pack(fill=tk.X)

        # --- Tab 2: Batch Processing ---
        batch_tab = tk.Frame(notebook, bg=t["bg_panel"], pady=10)
        notebook.add(batch_tab, text="Batch Directory")

        tk.Label(
            batch_tab,
            text="Process multiple images in a folder:",
            font=("Segoe UI", 9),
            bg=t["bg_panel"],
            fg=t["text_secondary"],
        ).pack(anchor=tk.W, pady=(0, 8))

        batch_btn = tk.Button(
            batch_tab,
            text="📂 Select Folder & Process",
            command=self._on_batch_folder,
            font=("Segoe UI", 10, "bold"),
            bg=t["primary"],
            fg="#ffffff",
            relief=tk.FLAT,
            pady=8,
            cursor="hand2",
        )
        batch_btn.pack(fill=tk.X, pady=(0, 10))

    def _build_right_panel(self, parent: tk.Frame):
        t = self.theme

        # View Mode Tabs
        view_tabs = ttk.Notebook(parent)
        view_tabs.pack(fill=tk.BOTH, expand=True)

        # Tab: Before / After Comparison
        comp_tab = tk.Frame(view_tabs, bg=t["bg_panel"])
        view_tabs.add(comp_tab, text="🔄 Side-by-Side Comparison")

        comp_box = tk.Frame(comp_tab, bg=t["bg_panel"])
        comp_box.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Original image label
        left_img_box = tk.Frame(comp_box, bg=t["bg_input"], relief=tk.FLAT)
        left_img_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        tk.Label(
            left_img_box,
            text="ORIGINAL PHOTO",
            font=("Segoe UI", 8, "bold"),
            bg=t["bg_card"],
            fg=t["text_muted"],
            pady=4,
        ).pack(fill=tk.X)

        self.orig_lbl = tk.Label(
            left_img_box,
            text="No image loaded",
            bg=t["bg_input"],
            fg=t["text_muted"],
            font=("Segoe UI", 11),
        )
        self.orig_lbl.pack(fill=tk.BOTH, expand=True)

        # Cartoon image label
        right_img_box = tk.Frame(comp_box, bg=t["bg_input"], relief=tk.FLAT)
        right_img_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        tk.Label(
            right_img_box,
            text="CARTOON ARTWORK",
            font=("Segoe UI", 8, "bold"),
            bg=t["bg_card"],
            fg=t["text_muted"],
            pady=4,
        ).pack(fill=tk.X)

        self.cartoon_lbl = tk.Label(
            right_img_box,
            text="Click 'Generate' to create artwork",
            bg=t["bg_input"],
            fg=t["text_muted"],
            font=("Segoe UI", 11),
        )
        self.cartoon_lbl.pack(fill=tk.BOTH, expand=True)

    def _on_style_changed(self):
        key = self.style_var.get()
        self.current_style_key = key
        cfg = STYLES.get(key, STYLES["ghibli_pro"])
        self.desc_lbl.config(text=cfg.description)

    def _on_strength_slider(self, val):
        self.strength_val_lbl.config(text=f"{val}%")

    def _on_open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")],
        )
        if file_path:
            self._load_file(Path(file_path))

    def _load_file(self, path: Path):
        try:
            self.current_filepath = path
            self.input_bgr = load_image(path)
            self.output_bgr = None
            self.save_btn.config(state=tk.DISABLED)

            self._display_on_label(self.input_bgr, self.orig_lbl)
            self.cartoon_lbl.config(image="", text="Click 'Generate' to create artwork")
            self.status_lbl.config(text=f"Loaded: {path.name} ({self.input_bgr.shape[1]}x{self.input_bgr.shape[0]})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _on_load_sample(self):
        # Create a synthetic colorful test pattern
        h, w = 512, 512
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Gradient background
        for y in range(h):
            for x in range(w):
                img[y, x] = [int(255 * x / w), int(128 + 127 * np.sin(x / 30)), int(255 * y / h)]
        # Draw sample face circles
        cv2.circle(img, (256, 220), 110, (200, 230, 255), -1)
        cv2.circle(img, (216, 200), 16, (40, 40, 40), -1)
        cv2.circle(img, (296, 200), 16, (40, 40, 40), -1)
        cv2.ellipse(img, (256, 260), (30, 15), 0, 0, 180, (50, 50, 220), 6)
        self.input_bgr = img
        self.current_filepath = Path("sample_pattern.jpg")
        self._display_on_label(self.input_bgr, self.orig_lbl)
        self.status_lbl.config(text="Sample loaded.")

    def _display_on_label(self, img_bgr: np.ndarray, label: tk.Label):
        if img_bgr is None:
            return
        # Calculate fit size
        w_avail = max(200, label.winfo_width() if label.winfo_width() > 10 else 400)
        h_avail = max(200, label.winfo_height() if label.winfo_height() > 10 else 400)

        resized = resize_keep_aspect(img_bgr, max_dim=max(w_avail, h_avail))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil_img)

        label.config(image=tk_img, text="")
        label.image = tk_img  # Prevent garbage collection

    def _on_generate(self):
        if self.input_bgr is None:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
        if self.is_processing:
            return

        self.is_processing = True
        self.gen_btn.config(state=tk.DISABLED, text="⏳ Generating...")
        self.prog_bar.start(10)
        self.status_lbl.config(text="Generating artwork...")

        style = self.style_var.get()
        strength = self.strength_scale.get() / 100.0
        face_align = self.face_align_var.get()

        def worker():
            try:
                result_bgr = self.engine.process_image(
                    self.input_bgr,
                    style=style,
                    strength=strength,
                    use_face_align=face_align,
                )
                self.output_bgr = result_bgr
                self.root.after(0, self._on_generate_complete, True, None)
            except Exception as e:
                self.root.after(0, self._on_generate_complete, False, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_generate_complete(self, success: bool, err_msg: Optional[str]):
        self.is_processing = False
        self.prog_bar.stop()
        self.gen_btn.config(state=tk.NORMAL, text="✨ Generate Cartoon")

        if success and self.output_bgr is not None:
            self._display_on_label(self.output_bgr, self.cartoon_lbl)
            self.save_btn.config(state=tk.NORMAL)
            self.status_lbl.config(text="✨ Artwork generated successfully!")
        else:
            messagebox.showerror("Error", f"Generation failed: {err_msg}")
            self.status_lbl.config(text="Error generating artwork.")

    def _on_save_image(self):
        if self.output_bgr is None:
            return
        file_path = filedialog.asksaveasfilename(
            title="Save Cartoon Image",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("WebP", "*.webp")],
        )
        if file_path:
            save_image(self.output_bgr, file_path)
            messagebox.showinfo("Saved", f"Image saved successfully:\n{file_path}")

    def _on_batch_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Images")
        if not folder:
            return
        out_dir = Path(folder) / "cartoon_results"
        p = Path(folder)
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        files = [f for ext in extensions for f in p.glob(ext) if not f.name.startswith("cartoon_")]
        if not files:
            messagebox.showinfo("No Images", "No images found in the selected folder.")
            return

        style = self.style_var.get()
        strength = self.strength_scale.get() / 100.0

        def batch_worker():
            self.engine.process_batch(files, out_dir, style=style, strength=strength)
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Batch Complete", f"Processed {len(files)} images to:\n{out_dir}"
                ),
            )

        threading.Thread(target=batch_worker, daemon=True).start()
        messagebox.showinfo("Batch Started", f"Processing {len(files)} images in background...")

    def _toggle_theme(self):
        self.current_theme_name = "light" if self.current_theme_name == "dark" else "dark"
        self.theme = THEMES[self.current_theme_name]
        self._apply_theme()


def main():
    root = tk.Tk()
    ModernCartoonifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
