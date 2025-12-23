"""
Enhanced Cartoon Image Generator - Professional Pro Edition
Combines multiple cartoon/anime styles with advanced controls
Features: Style strength slider, image sizes, history, dark/light mode, batch processing
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from enhanced_processor import EnhancedProcessor
from chatgpt_processor import ChatGPTCartoonProcessor
from ghibli_transform import convert_to_ghibli_array, convert_to_ghibli_array_optimized
from sdxl_ghibli import convert_to_ghibli_sdxl_array
import webbrowser

# Simple tooltip helper for widgets
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)

    def show(self, _e=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f'+{x}+{y}')
        lbl = tk.Label(self.tip, text=self.text, bg='#333', fg='white', bd=1, relief=tk.SOLID, padx=6, pady=3, font=('Segoe UI', 8))
        lbl.pack()

    def hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

# ==================== THEME SYSTEM ====================
THEMES = {
    'dark': {
        'primary': '#2E86AB',
        'primary_light': '#3E9FD8',
        'accent': '#A23B72',
        'accent_light': '#D84A6C',
        'bg_dark': '#0F1419',
        'bg_panel': '#1A1F2E',
        'bg_lighter': '#252B3A',
        'text_primary': '#E8F1F8',
        'text_secondary': '#99A9BA',
        'success': '#06A77D',
        'error': '#D32F2F',
        'border': '#3A4452',
        'button_hover': '#2B5380',
    },
    'light': {
        'primary': '#1976D2',
        'primary_light': '#42A5F5',
        'accent': '#C2185B',
        'accent_light': '#E91E63',
        'bg_dark': '#F5F5F5',
        'bg_panel': '#FFFFFF',
        'bg_lighter': '#FAFAFA',
        'text_primary': '#212121',
        'text_secondary': '#757575',
        'success': '#4CAF50',
        'error': '#F44336',
        'border': '#E0E0E0',
        'button_hover': '#1565C0',
    }
}

@dataclass
class StyleConfig:
    name: str
    checkpoint: str = None
    use_face_align: bool = True
    face_blend_strength: float = 0.7
    description: str = ""
    icon: str = ""
    color_accent: str = ""

# Available cartoon styles
STYLES = {
    'ghibli_pro': StyleConfig(
        'Ghibli Pro',
        None,
        use_face_align=False,
        face_blend_strength=0.75,
        description='Professional Ghibli-quality hand-painted effect',
        icon='🎬',
        color_accent='#FF6B6B'
    ),
    'anime_ghibli': StyleConfig(
        'Anime Ghibli',
        'face_paint_512_v2',
        use_face_align=True,
        face_blend_strength=0.75,
        description='Neural anime with Ghibli enhancement',
        icon='✨',
        color_accent='#4ECDC4'
    ),
    'paprika': StyleConfig(
        'Paprika Vibrant',
        'paprika',
        use_face_align=True,
        face_blend_strength=0.8,
        description='Warm vibrant cartoon with restoration',
        icon='🎨',
        color_accent='#FFB84D'
    ),
    'cartoon_gan': StyleConfig(
        'Cartoon GAN',
        None,
        use_face_align=False,
        face_blend_strength=1.0,
        description='Clean cartoon with color quantization',
        icon='🌈',
        color_accent='#95E1D3'
    ),
    'watercolor': StyleConfig(
        'Watercolor Art',
        None,
        use_face_align=False,
        face_blend_strength=1.0,
        description='Soft watercolor painting effect',
        icon='🎭',
        color_accent='#C7CEEA'
    ),
    'ghibli_sdxl': StyleConfig(
        'Ghibli SDXL',
        None,
        use_face_align=False,
        face_blend_strength=1.0,
        description='High-quality diffusion Ghibli illustration (slow)',
        icon='🌟',
        color_accent='#FFD700'
    ),
}

@dataclass
class GenerationRecord:
    timestamp: str
    image_path: str
    style: str
    style_strength: float
    image_size: str
    model_used: str
    prompt: str = ""
    metadata: dict = None

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'image_path': self.image_path,
            'style': self.style,
            'style_strength': self.style_strength,
            'image_size': self.image_size,
            'model_used': self.model_used,
            'prompt': self.prompt,
            'metadata': self.metadata or {}
        }

class EnhancedCartoonifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Cartoon Image Generator Pro')
        self.root.geometry('1600x950')
        self.root.minsize(1200, 700)
        
        # App state
        self.dark_mode = True
        self.theme = THEMES['dark']
        self.input_image = None
        self.input_image_bgr = None
        self.output_image = None
        self.output_image_pil = None
        self.current_style = 'ghibli_pro'
        self.style_strength = 0.7
        self.image_size = '512x512'
        self.is_generating = False
        self.history_data = []
        self.history_file = Path('cartoon_history.json')
        # Remember last folder used when opening images so new downloads are easy to find
        self.last_open_dir = str(Path.home() / 'Downloads') if (Path.home() / 'Downloads').exists() else str(Path.home())
        
        # Load history
        self._load_history()
        
        # Initialize processors
        self.enhanced_processor = EnhancedProcessor()
        self.chatgpt_processor = ChatGPTCartoonProcessor()
        
        # Setup UI
        self._setup_ui()
        self._apply_theme()
        self.root.configure(bg=self.theme['bg_dark'])
        
    def _setup_ui(self):
        """Setup the complete UI layout"""
        
        # Main container with padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== HEADER =====
        self._create_header(main_frame)
        # Toolbar under header for quick actions
        self._create_toolbar(main_frame)
        
        # ===== CONTENT =====
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        left_panel = self._create_left_panel(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Right panel - Image display
        right_panel = self._create_right_panel(content_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Status bar at bottom
        status_frame = tk.Frame(main_frame, bg=self.theme['bg_panel'], height=28)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        self.status_label = tk.Label(status_frame, text='Ready', bg=self.theme['bg_panel'], fg=self.theme['text_secondary'], font=('Segoe UI', 9))
        self.status_label.pack(side=tk.LEFT, padx=10)
        self.progressbar = ttk.Progressbar(status_frame, mode='indeterminate', length=180)
        self.progressbar.pack(side=tk.RIGHT, padx=10)
        
    def _create_header(self, parent):
        """Create header with title and theme toggle"""
        header = tk.Frame(parent, bg=self.theme['bg_panel'], height=80)
        # pack: do not use negative padding or unsupported 'margin' param
        header.pack(fill=tk.X, padx=0, pady=(0, 10))
        header.pack_propagate(False)
        
        # Title section
        title_frame = tk.Frame(header, bg=self.theme['bg_panel'])
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        tk.Label(
            title_frame,
            text='🎨 Cartoon Image Generator Pro',
            font=('Segoe UI', 24, 'bold'),
            bg=self.theme['bg_panel'],
            fg=self.theme['primary']
        ).pack(anchor=tk.W)
        
        tk.Label(
            title_frame,
            text='Multiple styles • Style control • Image history • Advanced features',
            font=('Segoe UI', 10),
            bg=self.theme['bg_panel'],
            fg=self.theme['text_secondary']
        ).pack(anchor=tk.W)
        
        # Control buttons
        button_frame = tk.Frame(header, bg=self.theme['bg_panel'])
        button_frame.pack(side=tk.RIGHT, padx=20, pady=15)
        
        self._create_button(
            button_frame,
            '☀️ Dark' if self.dark_mode else '🌙 Light',
            self._toggle_theme,
            width=12
        ).pack(side=tk.LEFT, padx=5)
        
        self._create_button(
            button_frame,
            '📚 History',
            self._show_history_window,
            width=12
        ).pack(side=tk.LEFT, padx=5)
        # Prominent generate button in header for visibility
        self._create_button(
            button_frame,
            '✨ Generate',
            self._generate_cartoon,
            width=12,
            color=self.theme['success']
        ).pack(side=tk.LEFT, padx=5)

        self._create_button(
            button_frame,
            '⚙️ Settings',
            self._show_settings,
            width=12
        ).pack(side=tk.LEFT, padx=5)

    def _create_toolbar(self, parent):
        """Create a compact toolbar with common actions"""
        toolbar = tk.Frame(parent, bg=self.theme['bg_panel'])
        toolbar.pack(fill=tk.X, padx=10, pady=(0, 8))

        btn_open = self._create_button(toolbar, '📁 Open', self._load_image, width=10)
        btn_open.pack(side=tk.LEFT, padx=4)
        Tooltip(btn_open, 'Open an image (Ctrl+O)')

        # Main generate button (toolbar)
        btn_generate_tb = self._create_button(toolbar, '✨ Generate', self._generate_cartoon, width=12, color=self.theme['success'])
        btn_generate_tb.pack(side=tk.LEFT, padx=4)
        Tooltip(btn_generate_tb, 'Generate cartoon (Ctrl+G)')

        # Save button (toolbar)
        self.save_button = self._create_button(
            toolbar,
            '💾 Save Image',
            self._save_image,
            width=10,
            color=self.theme['primary']
        )
        self.save_button.pack(side=tk.LEFT, padx=4)
        Tooltip(self.save_button, 'Save result (Ctrl+S)')
        # disabled until result exists
        self.save_button.config(state=tk.DISABLED)

        # --- Style dropdown in toolbar so user can always see/change style ---
        style_label = tk.Label(
            toolbar,
            text='Style:',
            bg=self.theme['bg_panel'],
            fg=self.theme['text_secondary'],
            font=('Segoe UI', 9)
        )
        style_label.pack(side=tk.LEFT, padx=(20, 4))

        self.style_var = tk.StringVar(value='ghibli_pro')
        style_names = [cfg.name for cfg in STYLES.values()]
        self._style_keys_by_name = {cfg.name: key for key, cfg in STYLES.items()}
        self.style_combo = ttk.Combobox(
            toolbar,
            values=style_names,
            state='readonly',
            width=20
        )
        self.style_combo.set(STYLES['ghibli_pro'].name)
        self.style_combo.pack(side=tk.LEFT, padx=4)

        def on_style_change(event=None):
            name = self.style_combo.get()
            key = self._style_keys_by_name.get(name, 'ghibli_pro')
            self.style_var.set(key)
            # Update left-panel description if it exists
            try:
                self._on_style_selected(key)
            except Exception:
                pass

        self.style_combo.bind('<<ComboboxSelected>>', on_style_change)
        
    def _create_left_panel(self, parent):
        """Create left control panel"""
        panel = tk.Frame(parent, bg=self.theme['bg_panel'], relief=tk.RAISED, bd=1)
        panel.pack_propagate(False)
        
        # Inner frame with padding
        inner = tk.Frame(panel, bg=self.theme['bg_panel'])
        inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # ===== IMAGE INPUT =====
        tk.Label(
            inner,
            text='📷 Select Image',
            font=('Segoe UI', 12, 'bold'),
            bg=self.theme['bg_panel'],
            fg=self.theme['text_primary']
        ).pack(anchor=tk.W, pady=(0, 10))
        
        self._create_button(
            inner,
            'Browse Image',
            self._load_image,
            width=25
        ).pack(fill=tk.X, pady=(0, 15))
        
        # ===== STYLE SELECTION =====
        tk.Label(
            inner,
            text='🎭 Cartoon Style',
            font=('Segoe UI', 12, 'bold'),
            bg=self.theme['bg_panel'],
            fg=self.theme['text_primary']
        ).pack(anchor=tk.W, pady=(10, 10))
        
        style_frame = tk.Frame(inner, bg=self.theme['bg_panel'])
        style_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.style_var = tk.StringVar(value='ghibli_pro')
        for style_key, style_config in STYLES.items():
            btn = tk.Radiobutton(
                style_frame,
                text=f"{style_config.icon} {style_config.name}",
                variable=self.style_var,
                value=style_key,
                font=('Segoe UI', 10),
                bg=self.theme['bg_panel'],
                fg=self.theme['text_primary'],
                selectcolor=self.theme['primary'],
                activebackground=self.theme['bg_lighter'],
                activeforeground=self.theme['primary']
            )
            btn.pack(anchor=tk.W, pady=3)
            btn.bind('<Button-1>', lambda e, s=style_key: self._on_style_selected(s))
        
        # Style description
        self.style_desc_label = tk.Label(
            inner,
            text=STYLES['ghibli_pro'].description,
            font=('Segoe UI', 8, 'italic'),
            bg=self.theme['bg_panel'],
            fg=self.theme['text_secondary'],
            wraplength=200,
            justify=tk.LEFT
        )
        self.style_desc_label.pack(fill=tk.X, pady=(5, 15))
        
        # ===== STYLE STRENGTH =====
        tk.Label(
            inner,
            text='✨ Style Strength',
            font=('Segoe UI', 12, 'bold'),
            bg=self.theme['bg_panel'],
            fg=self.theme['text_primary']
        ).pack(anchor=tk.W, pady=(10, 5))
        
        strength_frame = tk.Frame(inner, bg=self.theme['bg_panel'])
        strength_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            strength_frame,
            text='Light',
            font=('Segoe UI', 8),
            bg=self.theme['bg_panel'],
            fg=self.theme['text_secondary']
        ).pack(side=tk.LEFT)
        
        self.strength_slider = tk.Scale(
            strength_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            bg=self.theme['bg_lighter'],
            fg=self.theme['text_primary'],
            highlightthickness=0,
            troughcolor=self.theme['border']
        )
        self.strength_slider.set(0.7)
        self.strength_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        tk.Label(
            strength_frame,
            text='Strong',
            font=('Segoe UI', 8),
            bg=self.theme['bg_panel'],
            fg=self.theme['text_secondary']
        ).pack(side=tk.LEFT)
        
        self.strength_value_label = tk.Label(
            inner,
            text='70%',
            font=('Segoe UI', 10, 'bold'),
            bg=self.theme['bg_panel'],
            fg=self.theme['success']
        )
        self.strength_value_label.pack(anchor=tk.CENTER, pady=(0, 15))
        self.strength_slider.bind('<B1-Motion>', self._update_strength_label)
        self.strength_slider.bind('<ButtonRelease-1>', self._update_strength_label)
        
        # ===== IMAGE SIZE =====
        tk.Label(
            inner,
            text='📐 Image Size',
            font=('Segoe UI', 12, 'bold'),
            bg=self.theme['bg_panel'],
            fg=self.theme['text_primary']
        ).pack(anchor=tk.W, pady=(10, 10))
        
        self.size_var = tk.StringVar(value='512x512')
        for size in ['512x512', '640x640', '768x768', '1024x1024', '512x768 (Portrait)']:
            tk.Radiobutton(
                inner,
                text=size,
                variable=self.size_var,
                value=size,
                font=('Segoe UI', 9),
                bg=self.theme['bg_panel'],
                fg=self.theme['text_primary'],
                selectcolor=self.theme['primary'],
                activebackground=self.theme['bg_lighter'],
                activeforeground=self.theme['primary']
            ).pack(anchor=tk.W, pady=2)
        
        # ===== ACTION BUTTONS =====
        tk.Label(
            inner,
            text='',
            bg=self.theme['bg_panel']
        ).pack(pady=10)
        
        button_frame = tk.Frame(inner, bg=self.theme['bg_panel'])
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.generate_button = self._create_button(
            button_frame,
            '✨ Generate',
            self._generate_cartoon,
            width=25,
            color=self.theme['success']
        )
        self.generate_button.pack(fill=tk.X, pady=(0, 5))
        
        self._create_button(
            button_frame,
            '💾 Save Image',
            self._save_image,
            width=25,
            color=self.theme['primary']
        ).pack(fill=tk.X, pady=2)
        
        return panel
    
    def _create_right_panel(self, parent):
        """Create right image display panel"""
        panel = tk.Frame(parent, bg=self.theme['bg_panel'], relief=tk.RAISED, bd=1)
        
        # Tabs for different views
        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== IMAGE TAB =====
        image_frame = tk.Frame(notebook, bg=self.theme['bg_panel'])
        notebook.add(image_frame, text='📸 Generated Image')
        
        self.image_label = tk.Label(
            image_frame,
            bg=self.theme['bg_lighter'],
            fg=self.theme['text_secondary'],
            text='No image generated yet\nClick "Generate" to start',
            font=('Segoe UI', 14)
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== COMPARISON TAB =====
        comparison_frame = tk.Frame(notebook, bg=self.theme['bg_panel'])
        notebook.add(comparison_frame, text='🔄 Before/After')
        
        comparison_inner = tk.Frame(comparison_frame, bg=self.theme['bg_panel'])
        comparison_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.before_label = tk.Label(comparison_inner, bg=self.theme['bg_lighter'], text='Original')
        self.before_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.after_label = tk.Label(comparison_inner, bg=self.theme['bg_lighter'], text='Cartoon')
        self.after_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # ===== INFO TAB =====
        info_frame = tk.Frame(notebook, bg=self.theme['bg_panel'])
        notebook.add(info_frame, text='ℹ️ Information')
        
        info_inner = tk.Frame(info_frame, bg=self.theme['bg_panel'])
        info_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.info_text = tk.Text(
            info_inner,
            bg=self.theme['bg_lighter'],
            fg=self.theme['text_primary'],
            font=('Courier New', 9),
            height=30,
            width=50,
            relief=tk.FLAT,
            bd=1
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.insert(tk.END, self._get_info_text())
        self.info_text.config(state=tk.DISABLED)
        
        return panel
    
    def _create_button(self, parent, text, command, width=15, color=None):
        """Create styled button"""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=('Segoe UI', 10, 'bold'),
            bg=color or self.theme['primary'],
            fg='white',
            activebackground=self.theme['primary_light'] if not color else color,
            activeforeground='white',
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=8,
            width=width,
            cursor='hand2'
        )
        btn.bind('<Enter>', lambda e: btn.config(bg=self.theme['button_hover']))
        btn.bind('<Leave>', lambda e: btn.config(bg=color or self.theme['primary']))
        return btn
    
    def _load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title='Select an image',
            initialdir=self.last_open_dir,
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.webp'), ('All files', '*.*')]
        )
        if file_path:
            # Remember folder for next time so newly downloaded images are visible
            try:
                self.last_open_dir = str(Path(file_path).parent)
            except Exception:
                pass
            self.input_image = Image.open(file_path)
            self.input_image_bgr = cv2.imread(file_path)
            self._display_image(self.input_image, self.before_label)
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"✅ Image loaded: {Path(file_path).name}\n\n" + self._get_info_text())
            self.info_text.config(state=tk.DISABLED)
            # Enable generate button now that an image is loaded
            try:
                self.generate_button.config(state=tk.NORMAL)
            except Exception:
                pass
            try:
                self.save_button.config(state=tk.DISABLED)
            except Exception:
                pass
            # update status
            try:
                self.status_label.config(text=f'Image loaded: {Path(file_path).name}')
            except Exception:
                pass
    
    def _on_style_selected(self, style_key):
        """Update style description when selected"""
        self.current_style = style_key
        self.style_desc_label.config(text=STYLES[style_key].description)
    
    def _update_strength_label(self, event=None):
        """Update style strength label"""
        value = self.strength_slider.get()
        self.style_strength = value
        percentage = int(value * 100)
        self.strength_value_label.config(text=f'{percentage}%')
    
    def _generate_cartoon(self):
        """Generate cartoon version of image"""
        if self.input_image_bgr is None:
            messagebox.showwarning('No Image', 'Please load an image first!')
            return
        
        if self.is_generating:
            messagebox.showinfo('Generating', 'Already processing an image...')
            return
        
        self.is_generating = True
        self.generate_button.config(state=tk.DISABLED, text='⏳ Generating...')
        self.root.update()
        # Update status and start progress
        try:
            self.status_label.config(text='Generating...')
            self.progressbar.start(10)
        except Exception:
            pass
        
        def process():
            try:
                style_key = self.style_var.get()
                style_config = STYLES[style_key]
                strength = self.style_strength
                size_str = self.size_var.get().split()[0]
                
                # Parse size
                h, w = map(int, size_str.split('x'))
                img_resized = cv2.resize(self.input_image_bgr, (w, h))
                
                # Apply style (use available processor methods and sensible fallbacks)
                if style_key == 'ghibli_pro':
                    # Ghibli-style painterly look (no neural net required)
                    result = self.chatgpt_processor.ghibli_pro(img_resized)
                elif style_key == 'anime_ghibli':
                    # Optimized AnimeGANv2 pipeline (face-aware + lighting fixes)
                    result = convert_to_ghibli_array_optimized(img_resized, weight='face_paint_512_v1')
                elif style_key == 'paprika':
                    # Optimized AnimeGANv2 pipeline with paprika weights
                    result = convert_to_ghibli_array_optimized(img_resized, weight='paprika')
                elif style_key == 'watercolor':
                    # Watercolor effect (no strength parameter in current implementation)
                    result = self.enhanced_processor.watercolor_effect(img_resized)
                elif style_key == 'cartoon_gan':
                    # Use CartoonGAN-like style for classic cartoon look
                    result = self.enhanced_processor.cartoon_gan_style(img_resized)
                elif style_key == 'ghibli_sdxl':
                    # SDXL diffusion Ghibli illustration (can be slow, especially on CPU)
                    result = convert_to_ghibli_sdxl_array(img_resized, strength=0.5)
                else:
                    # Fallback: clarity enhancement
                    result = self.enhanced_processor.enhance_clarity(img_resized, strength=strength)
                
                self.output_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                self.output_image_pil = Image.fromarray(self.output_image)
                
                # Display result
                self._display_image(self.output_image_pil, self.image_label)
                self._display_comparison()
                
                # Save to history
                self._save_to_history(style_key, strength, size_str)
                
                # Update info
                self.info_text.config(state=tk.NORMAL)
                self.info_text.delete(1.0, tk.END)
                info = f"✅ Cartoon generated!\n\n"
                info += f"Style: {style_config.name}\n"
                info += f"Strength: {int(strength * 100)}%\n"
                info += f"Size: {size_str}\n"
                info += f"Model: {style_config.checkpoint or 'OpenCV + Enhancement'}\n\n"
                info += self._get_info_text()
                self.info_text.insert(tk.END, info)
                self.info_text.config(state=tk.DISABLED)
                
            except Exception as e:
                messagebox.showerror('Error', f'Failed to generate: {str(e)}')
            finally:
                self.is_generating = False
                self.generate_button.config(state=tk.NORMAL, text='✨ Generate')
                try:
                    self.progressbar.stop()
                except Exception:
                    pass
                # enable save if output exists
                if getattr(self, 'output_image_pil', None) is not None:
                    try:
                        self.save_button.config(state=tk.NORMAL)
                    except Exception:
                        pass
                try:
                    self.status_label.config(text='Ready')
                except Exception:
                    pass
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def _display_image(self, pil_image, label):
        """Display PIL image in label"""
        # Resize to fit label
        display_size = (400, 400)
        pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_image)
        label.config(image=photo, text='')
        label.image = photo  # Keep reference
    
    def _display_comparison(self):
        """Display before/after comparison"""
        if self.input_image and self.output_image_pil:
            self._display_image(self.input_image, self.before_label)
            self._display_image(self.output_image_pil, self.after_label)
    
    def _save_image(self):
        """Save generated image"""
        if self.output_image_pil is None:
            messagebox.showwarning('No Image', 'Generate an image first!')
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg'), ('All files', '*.*')]
        )
        if file_path:
            self.output_image_pil.save(file_path)
            messagebox.showinfo('Success', f'Image saved:\n{file_path}')
    
    def _save_to_history(self, style_key, strength, size):
        """Save generation to history"""
        record = GenerationRecord(
            timestamp=datetime.now().isoformat(),
            image_path=f"cartoon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            style=style_key,
            style_strength=strength,
            image_size=size,
            model_used=STYLES[style_key].checkpoint or 'OpenCV'
        )
        self.history_data.insert(0, record)
        # Keep only last 100
        if len(self.history_data) > 100:
            self.history_data = self.history_data[:100]
        self._save_history()
    
    def _load_history(self):
        """Load history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.history_data = [GenerationRecord(**item) for item in data]
            except:
                self.history_data = []
    
    def _save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump([r.to_dict() for r in self.history_data], f, indent=2)
        except:
            pass
    
    def _show_history_window(self):
        """Show history window"""
        hist_win = tk.Toplevel(self.root)
        hist_win.title('Generation History')
        hist_win.geometry('600x500')
        hist_win.configure(bg=self.theme['bg_dark'])
        
        # Title
        tk.Label(
            hist_win,
            text='📚 Generation History',
            font=('Segoe UI', 14, 'bold'),
            bg=self.theme['bg_dark'],
            fg=self.theme['text_primary']
        ).pack(padx=10, pady=10)
        
        # List frame with scrollbar
        list_frame = tk.Frame(hist_win, bg=self.theme['bg_panel'])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            list_frame,
            bg=self.theme['bg_lighter'],
            fg=self.theme['text_primary'],
            yscrollcommand=scrollbar.set,
            font=('Segoe UI', 9)
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate history
        for record in self.history_data:
            dt = datetime.fromisoformat(record.timestamp)
            style_name = STYLES[record.style].name
            text = f"{dt.strftime('%Y-%m-%d %H:%M')} | {style_name} | {int(record.style_strength*100)}% | {record.image_size}"
            listbox.insert(tk.END, text)
        
        if not self.history_data:
            listbox.insert(tk.END, 'No history yet')
        
        # Clear button
        self._create_button(
            hist_win,
            '🗑️ Clear History',
            lambda: self._clear_history(hist_win),
            width=30
        ).pack(pady=10)
    
    def _clear_history(self, window):
        """Clear history"""
        if messagebox.askyesno('Confirm', 'Clear all history?'):
            self.history_data = []
            self._save_history()
            window.destroy()
            messagebox.showinfo('Cleared', 'History cleared!')
    
    def _show_settings(self):
        """Show settings window"""
        settings_win = tk.Toplevel(self.root)
        settings_win.title('Settings')
        settings_win.geometry('400x300')
        settings_win.configure(bg=self.theme['bg_dark'])
        
        tk.Label(
            settings_win,
            text='⚙️ Settings',
            font=('Segoe UI', 14, 'bold'),
            bg=self.theme['bg_dark'],
            fg=self.theme['text_primary']
        ).pack(padx=10, pady=10)
        
        # Settings content
        content = tk.Frame(settings_win, bg=self.theme['bg_panel'])
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        settings_text = f"""
Theme: {'Dark Mode' if self.dark_mode else 'Light Mode'}
Current Style: {STYLES[self.current_style].name}
History Items: {len(self.history_data)}
History File: {self.history_file}

Supported Styles:
{chr(10).join(f"  • {cfg.name}: {cfg.description}" for cfg in STYLES.values())}
        """
        
        tk.Label(
            content,
            text=settings_text,
            font=('Courier New', 9),
            bg=self.theme['bg_panel'],
            fg=self.theme['text_primary'],
            justify=tk.LEFT,
            wraplength=350
        ).pack(padx=10, pady=10)
    
    def _toggle_theme(self):
        """Toggle between dark and light theme"""
        self.dark_mode = not self.dark_mode
        self.theme = THEMES['dark' if self.dark_mode else 'light']
        self._apply_theme()
        messagebox.showinfo('Theme', f'Switched to {"Dark" if self.dark_mode else "Light"} Mode')
    
    def _apply_theme(self):
        """Apply theme to all widgets"""
        self.root.configure(bg=self.theme['bg_dark'])
        # Note: Full theme application would require recursive widget update
        # This is a simplified version
    
    def _get_info_text(self):
        """Get information text"""
        return """
SHORTCUTS:
• Ctrl+O: Open image
• Ctrl+S: Save image
• Ctrl+G: Generate
• Ctrl+H: Show history

TIPS:
• Start with 512x512 for quick previews
• Increase style strength for more effect
• Try different styles on the same image
• Use history to compare results

STYLES:
🎬 Ghibli Pro: Hand-painted effect
✨ Anime Ghibli: Neural anime
🎨 Paprika: Vibrant cartoon
🌈 Cartoon GAN: Clean style
🎭 Watercolor: Soft painting
        """

def main():
    root = tk.Tk()
    app = EnhancedCartoonifierApp(root)
    
    # Bind shortcuts
    root.bind('<Control-o>', lambda e: app._load_image())
    root.bind('<Control-g>', lambda e: app._generate_cartoon())
    root.bind('<Control-s>', lambda e: app._save_image())
    root.bind('<Control-h>', lambda e: app._show_history_window())
    
    root.mainloop()

if __name__ == '__main__':
    main()
