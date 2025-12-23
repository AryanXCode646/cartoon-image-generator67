"""
Interactive GUI Cartoonifier App - Professional Edition
Supports multiple cartoon styles with real-time preview and parameter tuning.
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from dataclasses import dataclass
from enhanced_processor import EnhancedProcessor
from chatgpt_processor import ChatGPTCartoonProcessor

# Color Theme (Modern Dark Professional)
THEME = {
    'primary': '#2E86AB',      # Professional blue
    'primary_light': '#3E9FD8',
    'accent': '#A23B72',       # Accent purple/magenta
    'accent_light': '#D84A6C',
    'bg_dark': '#0F1419',      # Very dark blue-grey
    'bg_panel': '#1A1F2E',     # Panel dark
    'bg_lighter': '#252B3A',   # Slightly lighter
    'text_primary': '#E8F1F8', # Light text
    'text_secondary': '#99A9BA',
    'success': '#06A77D',      # Green
    'error': '#D32F2F',        # Red
    'border': '#3A4452',
}

@dataclass
class StyleConfig:
    name: str
    checkpoint: str
    use_face_align: bool = True
    face_blend_strength: float = 0.7
    description: str = ""
    icon: str = ""

# Available cartoon styles
STYLES = {
    'ghibli_pro': StyleConfig(
        'Ghibli Pro',
        None,
        use_face_align=False,
        face_blend_strength=0.75,
        description='Professional Ghibli-quality hand-painted effect (ChatGPT level)',
        icon='🎬'
    ),
    'ghibli_enhanced': StyleConfig(
        'Anime Ghibli',
        'face_paint_512_v2',
        use_face_align=True,
        face_blend_strength=0.75,
        description='Neural anime with Ghibli enhancement + GFPGAN restoration',
        icon='✨'
    ),
    'paprika': StyleConfig(
        'Paprika Vibrant',
        'paprika',
        use_face_align=True,
        face_blend_strength=0.8,
        description='Warm vibrant cartoon with face restoration',
        icon='🎨'
    ),
    'cartoon_gan': StyleConfig(
        'Cartoon GAN',
        None,
        use_face_align=False,
        face_blend_strength=1.0,
        description='Clean cartoon with color quantization',
        icon='🌈'
    ),
    'watercolor': StyleConfig(
        'Watercolor Art',
        None,
        use_face_align=False,
        face_blend_strength=1.0,
        description='Soft watercolor painting effect',
        icon='🎭'
    ),
}

class CartoonifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Cartoon Image Generator Pro')
        self.root.geometry('1400x850')
        self.root.resizable(True, True)
        
        # Apply theme colors
        self.root.configure(bg=THEME['bg_dark'])
        
        self.input_image = None
        self.input_image_bgr = None
        self.output_image = None
        self.model = None
        self.current_style = None
        self.is_processing = False
        self.device = 'cpu'
        
        # Initialize ChatGPT processor
        self.gpt_processor = ChatGPTCartoonProcessor()
        
        self._setup_styles()
        self._setup_ui()
        
    def _setup_styles(self):
        """Configure ttk styles with theme colors"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background=THEME['bg_dark'])
        style.configure('TLabel', background=THEME['bg_dark'], foreground=THEME['text_primary'])
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground=THEME['primary_light'])
        style.configure('Heading.TLabel', font=('Segoe UI', 11, 'bold'), foreground=THEME['text_primary'])
        style.configure('Info.TLabel', font=('Segoe UI', 9), foreground=THEME['text_secondary'])
        
        style.configure('TButton', font=('Segoe UI', 10), relief=tk.FLAT, padding=8)
        style.configure('Accent.TButton', background=THEME['primary'], foreground=THEME['text_primary'])
        style.map('Accent.TButton',
            background=[('active', THEME['primary_light'])],
            foreground=[('active', THEME['text_primary'])])
        
        style.configure('TRadiobutton', background=THEME['bg_panel'], foreground=THEME['text_primary'], font=('Segoe UI', 10))
        style.map('TRadiobutton', background=[('active', THEME['bg_lighter'])])
        
        style.configure('TScale', background=THEME['bg_panel'])
        style.map('TScale', background=[('active', THEME['bg_lighter'])])
        
        style.configure('TNotebook', background=THEME['bg_dark'], borderwidth=0)
        style.configure('TNotebook.Tab', font=('Segoe UI', 10), padding=[20, 10])
        style.map('TNotebook.Tab',
            background=[('selected', THEME['bg_panel'])],
            foreground=[('selected', THEME['primary_light'])])
        
        style.configure('TSeparator', background=THEME['border'])
        
    def _setup_ui(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Header bar
        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X, padx=20, pady=(15, 10))
        ttk.Label(header, text='🎬 Cartoon Image Generator Pro', style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(header, text='Transform your photos into beautiful cartoon artwork', style='Info.TLabel').pack(anchor=tk.W)
        
        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Content frame
        content = ttk.Frame(main_frame)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        # Left panel: Controls
        left_frame = ttk.Frame(content)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))
        
        # Left panel background
        left_bg = tk.Frame(left_frame, bg=THEME['bg_panel'], highlightthickness=1, highlightbackground=THEME['border'])
        left_bg.pack(fill=tk.BOTH, padx=0)
        
        # Inner left content
        left_inner = ttk.Frame(left_bg)
        left_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # File section
        ttk.Label(left_inner, text='📂 Image Input', style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        load_btn = ttk.Button(left_inner, text='📁 Browse & Load Image', command=self._load_image)
        load_btn.pack(fill=tk.X, pady=(0, 8))
        
        self.file_label = ttk.Label(left_inner, text='No image loaded', style='Info.TLabel', wraplength=220)
        self.file_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Separator
        ttk.Separator(left_inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 15))
        
        # Style selection section
        ttk.Label(left_inner, text='🎨 Cartoon Style', style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 12))
        self.style_var = tk.StringVar(value='ghibli_pro')
        
        for key, style in STYLES.items():
            style_frame = ttk.Frame(left_inner)
            style_frame.pack(fill=tk.X, pady=4)
            
            ttk.Radiobutton(
                style_frame,
                text=f'{style.icon} {style.name}',
                variable=self.style_var,
                value=key,
                command=self._on_style_changed
            ).pack(anchor=tk.W)
            
            ttk.Label(style_frame, text=style.description, style='Info.TLabel', wraplength=200).pack(anchor=tk.W, padx=(20, 0))
        
        ttk.Separator(left_inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Parameters section
        ttk.Label(left_inner, text='⚙️ Parameters', style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        param_frame = ttk.Frame(left_inner)
        param_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(param_frame, text='Face Blend Strength', style='Info.TLabel').pack(anchor=tk.W)
        
        slider_frame = ttk.Frame(param_frame)
        slider_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.blend_slider = ttk.Scale(slider_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL)
        self.blend_slider.set(0.7)
        self.blend_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.blend_label = ttk.Label(slider_frame, text='0.70', style='Info.TLabel', width=5)
        self.blend_label.pack(side=tk.RIGHT, padx=(8, 0))
        
        self.blend_slider.configure(command=self._update_blend_label)
        
        ttk.Separator(left_inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Action buttons
        self.process_btn = ttk.Button(left_inner, text='✨ Generate Cartoon', command=self._generate)
        self.process_btn.pack(fill=tk.X, pady=(0, 8))
        
        self.progress_label = ttk.Label(left_inner, text='', style='Info.TLabel')
        self.progress_label.pack(pady=(4, 15))
        
        self.save_btn = ttk.Button(left_inner, text='💾 Save Result', command=self._save_image, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X)
        
        # Hosted inference (high-quality) button
        self.hosted_btn = ttk.Button(left_inner, text='☁️ Use Hosted High-Quality', command=self._open_hosted_dialog)
        self.hosted_btn.pack(fill=tk.X, pady=(10, 0))
        
        # Right panel: Preview
        right_frame = ttk.Frame(content)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Input preview
        input_tab = ttk.Frame(self.notebook)
        self.notebook.add(input_tab, text='📷 Input Image')
        
        input_bg = tk.Frame(input_tab, bg=THEME['bg_lighter'])
        input_bg.pack(fill=tk.BOTH, expand=True)
        
        self.input_canvas = tk.Canvas(input_bg, bg=THEME['bg_dark'], highlightthickness=0, cursor='arrow')
        self.input_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Output preview
        output_tab = ttk.Frame(self.notebook)
        self.notebook.add(output_tab, text='✨ Result Preview')
        
        output_bg = tk.Frame(output_tab, bg=THEME['bg_lighter'])
        output_bg.pack(fill=tk.BOTH, expand=True)
        
        self.output_canvas = tk.Canvas(output_bg, bg=THEME['bg_dark'], highlightthickness=0, cursor='arrow')
        self.output_canvas.pack(fill=tk.BOTH, expand=True)
        
        self._on_style_changed()
        
    def _on_style_changed(self):
        style_key = self.style_var.get()
        style = STYLES[style_key]
        self.blend_slider.set(style.face_blend_strength)

    # --- Hosted inference helpers ---
    def _open_hosted_dialog(self):
        cfg = self._load_hosted_config()
        dlg = tk.Toplevel(self.root)
        dlg.title('Hosted Model Settings')
        dlg.geometry('420x220')
        dlg.transient(self.root)

        ttk.Label(dlg, text='Provider').pack(anchor=tk.W, padx=12, pady=(12, 4))
        provider_var = tk.StringVar(value=cfg.get('provider', 'replicate'))
        ttk.Combobox(dlg, textvariable=provider_var, values=['replicate', 'huggingface']).pack(fill=tk.X, padx=12)

        ttk.Label(dlg, text='API Key').pack(anchor=tk.W, padx=12, pady=(10, 4))
        api_var = tk.StringVar(value=cfg.get('api_key', ''))
        ttk.Entry(dlg, textvariable=api_var).pack(fill=tk.X, padx=12)

        def save_and_close():
            new = {'provider': provider_var.get(), 'api_key': api_var.get().strip()}
            self._save_hosted_config(new)
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill=tk.X, pady=14, padx=12)
        ttk.Button(btn_frame, text='Save', command=save_and_close).pack(side=tk.RIGHT)

    def _hosted_config_path(self):
        return Path(__file__).parent / '.hosted_config.json'

    def _save_hosted_config(self, cfg: dict):
        try:
            with open(self._hosted_config_path(), 'w', encoding='utf-8') as f:
                json.dump(cfg, f)
        except Exception:
            pass

    def _load_hosted_config(self):
        p = self._hosted_config_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _call_hosted_model(self):
        cfg = self._load_hosted_config()
        provider = cfg.get('provider')
        api_key = cfg.get('api_key')
        if not api_key or not provider:
            messagebox.showinfo('Hosted Inference', 'Please set provider and paste your API key in Hosted Model Settings first.')
            return

        # Ensure we have an image loaded
        if self.input_image_bgr is None:
            messagebox.showwarning('No Image', 'Load an image first')
            return

        self.progress_label.config(text='⏳ Sending to hosted model...')
        self.process_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            if provider == 'replicate':
                out = self._hosted_predict_replicate(api_key)
            else:
                out = self._hosted_predict_hf(api_key)
        except Exception as e:
            messagebox.showerror('Hosted Error', f'Hosted model request failed: {e}')
            out = None

        if out is None:
            self.progress_label.config(text='✗ Hosted failed')
            self.process_btn.config(state=tk.NORMAL)
            return

        # out is bytes of image
        try:
            from io import BytesIO
            pil = Image.open(BytesIO(out)).convert('RGB')
            result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            self.output_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            self._display_output()
            self.progress_label.config(text='✓ Hosted result ready')
            self.save_btn.config(state=tk.NORMAL)
            self.notebook.select(1)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to decode hosted response: {e}')
            self.progress_label.config(text='✗ Decode error')
        finally:
            self.process_btn.config(state=tk.NORMAL)

    def _hosted_predict_replicate(self, api_token: str):
        """Send image and prompt to Replicate predictions API. Returns image bytes on success."""
        # Default model and prompt (Ghibli-style)
        model = 'stability-ai/stable-diffusion-x4-upscaler'  # fallback placeholder
        prompt = (
            "An anime portrait in the style of Studio Ghibli, detailed, soft colors, hand-painted, dramatic lighting"
        )

        # Convert current input image to bytes (png)
        import base64
        from io import BytesIO
        pil = Image.fromarray(cv2.cvtColor(self.input_image_bgr, cv2.COLOR_BGR2RGB))
        bio = BytesIO()
        pil.save(bio, format='PNG')
        bio.seek(0)
        b64 = base64.b64encode(bio.read()).decode('utf-8')

        url = 'https://api.replicate.com/v1/predictions'
        headers = {'Authorization': f'Token {api_token}', 'Content-Type': 'application/json'}
        payload = {
            'version': None,
            'input': {
                'prompt': prompt,
                'image': f'data:image/png;base64,{b64}'
            }
        }

        # Use a generic model field; keep flexible and let user change config file directly if desired
        # Note: many Replicate models expect different input names; this is a best-effort adapter.
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        if r.status_code not in (200, 201):
            raise RuntimeError(f'Replicate API error {r.status_code}: {r.text}')

        job = r.json()
        # Poll for result
        poll_url = job.get('urls', {}).get('get') or job.get('url')
        if not poll_url:
            raise RuntimeError('No job URL returned')

        # Poll until complete
        for _ in range(60):
            rr = requests.get(poll_url, headers=headers, timeout=120)
            data = rr.json()
            status = data.get('status')
            if status == 'succeeded':
                # fetch output
                out_urls = data.get('output')
                if not out_urls:
                    raise RuntimeError('No output from hosted model')
                out_url = out_urls[0]
                img_r = requests.get(out_url, timeout=120)
                return img_r.content
            elif status in ('failed', 'canceled'):
                raise RuntimeError(f'Hosted job {status}: {data}')
            else:
                import time
                time.sleep(1.5)

        raise RuntimeError('Hosted job timed out')

    def _hosted_predict_hf(self, api_key: str):
        """Placeholder for Hugging Face inference API. Returns image bytes on success."""
        # For now, delegate to user: not implemented fully.
        raise RuntimeError('Hugging Face hosted integration not implemented in this build')
        
    def _update_blend_label(self, value):
        self.blend_label.config(text=f'{float(value):.2f}')
        
    def _load_image(self):
        path = filedialog.askopenfilename(
            title='Select Image',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp'), ('All', '*.*')]
        )
        if not path:
            return
        
        self.input_image_bgr = cv2.imread(path)
        if self.input_image_bgr is None:
            messagebox.showerror('Error', 'Failed to load image')
            return
        
        self.input_image = Image.open(path)
        self.file_label.config(text=Path(path).name, foreground='black')
        self.output_image = None
        self.save_btn.config(state=tk.DISABLED)
        self._display_input()
        self.notebook.select(0)
        
    def _display_input(self):
        if self.input_image is None:
            return
        self._display_on_canvas(self.input_canvas, self.input_image)
        
    def _display_on_canvas(self, canvas, pil_img):
        # Fit image to canvas
        canvas.delete('all')
        w, h = pil_img.size
        canvas_w = canvas.winfo_width() if canvas.winfo_width() > 1 else 600
        canvas_h = canvas.winfo_height() if canvas.winfo_height() > 1 else 600
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        
        display_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.display_photo = ImageTk.PhotoImage(display_img)
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        canvas.create_image(x, y, image=self.display_photo, anchor=tk.NW)
        
    def _generate(self):
        if self.input_image_bgr is None:
            messagebox.showwarning('Warning', 'Please load an image first')
            return
        
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.progress_label.config(text='Loading model...')
        self.root.update()
        
        thread = threading.Thread(target=self._generate_in_thread)
        thread.start()
        
    def _generate_in_thread(self):
        try:
            style_key = self.style_var.get()
            style = STYLES[style_key]
            blend = float(self.blend_slider.get())
            
            self.progress_label.config(text='⏳ Initializing...')
            self.root.update_idletasks()
            
            # Load model if needed
            model = None
            if style.checkpoint is not None:
                self.progress_label.config(text='⏳ Loading AI model...')
                self.root.update_idletasks()
                model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator', pretrained=style.checkpoint)
                model.to(self.device)
                model.eval()
            
            self.progress_label.config(text='⏳ Generating cartoon...')
            self.root.update_idletasks()
            
            # Process based on style config
            if style.checkpoint and style.use_face_align:
                result = self._process_face_aligned(model, style, blend)
            else:
                result = self._process_full_image(model, style)
            
            if result is not None:
                self.output_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                self._display_output()
                self.progress_label.config(text='✓ ChatGPT-level complete!')
                self.save_btn.config(state=tk.NORMAL)
                self.notebook.select(1)
            else:
                messagebox.showerror('Error', 'Failed to process image')
                self.progress_label.config(text='✗ Error')
        except Exception as e:
            messagebox.showerror('Error', f'Processing failed: {str(e)}')
            self.progress_label.config(text='✗ Error')
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False
            self.process_btn.config(state=tk.NORMAL)
            
    def _process_full_image(self, model, style):
        """Process full image with model and post-processing"""
        # Use ChatGPT processor for best quality
        if self.style_var.get() == 'ghibli_pro':
            return self.gpt_processor.ghibli_pro(self.input_image_bgr)
        elif style.checkpoint is None:
            # Use OpenCV-based styles
            if self.style_var.get() == 'cartoon_gan':
                return EnhancedProcessor.cartoon_gan_style(self.input_image_bgr)
            elif self.style_var.get() == 'watercolor':
                return EnhancedProcessor.watercolor_effect(self.input_image_bgr)
        
        # Use neural model with ChatGPT-level post-processing
        return self.gpt_processor.cartoon_anime_gpt(self.input_image_bgr, model, device=self.device)
        
    def _process_face_aligned(self, model, style, blend_strength):
        img = self.input_image_bgr
        rect = self._detect_face(img)
        if rect is None:
            # Fallback to full-image
            return self._process_full_image(model, style)
        
        x, y, fw, fh = rect
        cx, cy = x + fw/2, y + fh/2
        new_w, new_h = fw * 1.6, fh * 1.6
        x1 = int(max(0, cx - new_w/2))
        y1 = int(max(0, cy - new_h/2))
        x2 = int(min(img.shape[1], cx + new_w/2))
        y2 = int(min(img.shape[0], cy + new_h/2))
        
        face_crop = img[y1:y2, x1:x2]
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        face_resized = face_pil.resize((512, 512), Image.BICUBIC)
        
        with torch.no_grad():
            input_t = ChatGPTCartoonProcessor.pil_to_tensor(face_resized).to(self.device)
            out = model(input_t)
            if isinstance(out, (list, tuple)):
                out = out[0]
        
        out_pil = ChatGPTCartoonProcessor.tensor_to_pil(out)
        out_face_resized = out_pil.resize((x2-x1, y2-y1), Image.BICUBIC)
        out_face_bgr = cv2.cvtColor(np.array(out_face_resized), cv2.COLOR_RGB2BGR)
        
        # ChatGPT-level enhancement
        out_face_bgr = self.gpt_processor.enhance_cartoon_quality(out_face_bgr)
        out_face_bgr = self.gpt_processor.restore_face_gfpgan(out_face_bgr)
        
        # Blend with soft mask
        h_c, w_c = out_face_bgr.shape[:2]
        mask = np.zeros((h_c, w_c), dtype=np.uint8)
        cv2.ellipse(mask, (w_c//2, h_c//2), (int(w_c*0.45), int(h_c*0.55)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask_f = (mask.astype(float)/255.0) * blend_strength
        
        result = img.copy().astype(float)
        roi = result[y1:y2, x1:x2]
        blended = (out_face_bgr.astype(float) * mask_f[:,:,None] + roi * (1-mask_f)[:,:,None]).astype(np.uint8)
        result = img.copy()
        result[y1:y2, x1:x2] = blended
        
        # Apply Ghibli enhancement if selected
        if self.style_var.get() == 'ghibli_enhanced':
            result = EnhancedProcessor.ghibli_style_enhanced(result)
        
        return result
        
    def _detect_face(self, img):
        casc_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(casc_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        areas = [w*h for (x,y,w,h) in faces]
        idx = int(np.argmax(areas))
        return faces[idx]
        
    def _pil_to_tensor(self, pil_img):
        import torchvision.transforms as T
        t = T.ToTensor()(pil_img).unsqueeze(0) * 2 - 1
        return t
        
    def _tensor_to_pil(self, tensor):
        t = tensor.squeeze(0).cpu().detach()
        t = (t + 1) / 2
        t = torch.clamp(t, 0, 1)
        arr = (t.permute(1,2,0).numpy() * 255).astype('uint8')
        return Image.fromarray(arr)
        
    def _display_output(self):
        if self.output_image is None:
            return
        self._display_on_canvas(self.output_canvas, self.output_image)
        
    def _save_image(self):
        if self.output_image is None:
            messagebox.showwarning('Warning', 'No output to save')
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension='.jpg',
            filetypes=[('JPEG', '*.jpg'), ('PNG', '*.png'), ('All', '*.*')]
        )
        if path:
            self.output_image.save(path)
            messagebox.showinfo('✓ Success', f'Image saved to {Path(path).name}')

if __name__ == '__main__':
    root = tk.Tk()
    app = CartoonifierApp(root)
    root.mainloop()
