"""
ChatGPT-Level Cartoon Generation
Uses state-of-the-art models: AnimeGAN, GFPGAN, Real-ESRGAN for professional quality
"""
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import os

# Try to import advanced models
try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except ImportError:
    HAS_GFPGAN = False

try:
    from realesrgan import RealESRGANer
    HAS_REALESRGAN = True
except ImportError:
    HAS_REALESRGAN = False


class ChatGPTCartoonProcessor:
    """Professional-grade cartoon generation matching ChatGPT quality"""
    
    def __init__(self):
        self.device = 'cpu'
        self.gfpgan_restorer = None
        self.upsampler = None
        self._init_models()
    
    def _init_models(self):
        """Initialize GFPGAN and Real-ESRGAN if available"""
        if HAS_GFPGAN:
            try:
                self.gfpgan_restorer = GFPGANer(
                    scale=2,
                    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
                    upscale_after_restore=False,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
            except Exception as e:
                print(f"GFPGAN init warning: {e}")
                self.gfpgan_restorer = None
        
        if HAS_REALESRGAN:
            try:
                self.upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model='RealESRGAN_x2plus',
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=False
                )
            except Exception as e:
                print(f"Real-ESRGAN init warning: {e}")
                self.upsampler = None
    
    @staticmethod
    def pil_to_tensor(pil_img):
        """Convert PIL to tensor"""
        t = T.ToTensor()(pil_img).unsqueeze(0) * 2 - 1
        return t
    
    @staticmethod
    def tensor_to_pil(tensor):
        """Convert tensor to PIL"""
        t = tensor.squeeze(0).cpu().detach()
        t = (t + 1) / 2
        t = torch.clamp(t, 0, 1)
        arr = (t.permute(1, 2, 0).numpy() * 255).astype('uint8')
        return Image.fromarray(arr)
    
    def enhance_cartoon_quality(self, img_bgr):
        """Apply multiple enhancement passes for ChatGPT-level quality"""
        # Step 1: Color saturation boost
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Boost saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Step 2: Adaptive histogram equalization for local contrast
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Step 3: Bilateral smoothing for painterly effect
        for _ in range(2):
            img_bgr = cv2.bilateralFilter(img_bgr, 11, 80, 80)
        
        # Step 4: Detail enhancement with unsharp mask
        gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 1.5)
        img_bgr = cv2.addWeighted(img_bgr, 1.4, gaussian, -0.4, 0)
        
        # Step 5: Sharpen edges
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.5
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        img_bgr = cv2.addWeighted(img_bgr, 0.7, sharpened, 0.3, 0)
        
        return np.clip(img_bgr, 0, 255).astype(np.uint8)
    
    def restore_face_gfpgan(self, img_bgr):
        """Apply GFPGAN for face restoration"""
        if not HAS_GFPGAN or self.gfpgan_restorer is None:
            return img_bgr
        
        try:
            # GFPGAN expects RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            _, restored_rgb, _ = self.gfpgan_restorer.enhance(
                img_rgb, 
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5
            )
            restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
            return restored_bgr
        except Exception as e:
            print(f"GFPGAN error: {e}")
            return img_bgr
    
    def upscale_realesrgan(self, img_bgr, scale=2):
        """Upscale image using Real-ESRGAN"""
        if not HAS_REALESRGAN or self.upsampler is None:
            return img_bgr
        
        try:
            output, _ = self.upsampler.enhance(img_bgr, outscale=scale)
            return output
        except Exception as e:
            print(f"Real-ESRGAN error: {e}")
            return img_bgr
    
    def cartoon_anime_gpt(self, img_bgr, model, device='cpu'):
        """Generate cartoon in ChatGPT style using AnimeGAN + enhancements"""
        # Optimize input size
        h, w = img_bgr.shape[:2]
        short = min(h, w)
        
        # If image is too large, resize smartly
        if short > 768:
            scale = 768 / short
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_bgr
            new_w, new_h = w, h
        
        # Prepare for model
        scale = 512 / min(new_h, new_w)
        final_w = max(1, int(new_w * scale))
        final_h = max(1, int(new_h * scale))
        img_model = cv2.resize(img_resized, (final_w, final_h))
        
        # Run through model
        pil_in = Image.fromarray(cv2.cvtColor(img_model, cv2.COLOR_BGR2RGB))
        input_t = self.pil_to_tensor(pil_in).to(device)
        
        with torch.no_grad():
            output = model(input_t)
            if isinstance(output, (list, tuple)):
                output = output[0]
        
        out_pil = self.tensor_to_pil(output)
        out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
        
        # Resize back to original dimensions
        output_cartoon = cv2.resize(out_bgr, (w, h))
        
        # Step 1: Enhance cartoon quality
        output_cartoon = self.enhance_cartoon_quality(output_cartoon)
        
        # Step 2: Detect and blend faces for identity preservation
        output_cartoon = self._blend_face_preserve(img_bgr, output_cartoon)
        
        # Step 3: Try face restoration if available
        output_cartoon = self.restore_face_gfpgan(output_cartoon)
        
        # Step 4: Optional upscaling for extra clarity (2x)
        if short < 720:  # Only upscale if original is not too large
            output_cartoon = self.upscale_realesrgan(output_cartoon, scale=2)
        
        return output_cartoon
    
    def _blend_face_preserve(self, original, stylized, blend_strength=0.3):
        """Blend face from original to preserve identity"""
        rect = self._detect_face_cascade(original)
        if rect is None:
            return stylized
        
        x, y, fw, fh = rect
        # Expand face region
        cx, cy = x + fw/2, y + fh/2
        new_w, new_h = fw * 1.5, fh * 1.5
        x1 = int(max(0, cx - new_w/2))
        y1 = int(max(0, cy - new_h/2))
        x2 = int(min(original.shape[1], cx + new_w/2))
        y2 = int(min(original.shape[0], cy + new_h/2))
        
        # Create soft mask
        h_region, w_region = y2 - y1, x2 - x1
        mask = np.zeros((h_region, w_region), dtype=np.uint8)
        cv2.ellipse(mask, (w_region//2, h_region//2), 
                   (int(w_region*0.4), int(h_region*0.5)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask_f = (mask.astype(float)/255.0) * blend_strength
        
        # Blend
        result = stylized.copy().astype(float)
        orig_region = original[y1:y2, x1:x2].astype(float)
        sty_region = stylized[y1:y2, x1:x2].astype(float)
        blended = sty_region * (1 - mask_f[:, :, None]) + orig_region * mask_f[:, :, None]
        result[y1:y2, x1:x2] = blended
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _detect_face_cascade(self, img_bgr):
        """Detect largest face"""
        casc_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(casc_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return None
        areas = [w*h for (x,y,w,h) in faces]
        idx = int(np.argmax(areas))
        return faces[idx]
    
    def ghibli_pro(self, img_bgr):
        """Professional Ghibli-style effect without neural net"""
        # Bilateral smoothing for painterly look
        result = img_bgr.copy()
        for _ in range(3):
            result = cv2.bilateralFilter(result, 9, 75, 75)
        
        # Enhance local contrast
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(10, 10))
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Boost colors
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.25
        hsv[:, :, 2] = hsv[:, :, 2] * 1.1
        hsv = np.clip(hsv, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Detect and emphasize edges
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        
        # Blend edge definition
        result = result.astype(float) * (1 - 0.25 * edges_3ch) + img_bgr.astype(float) * 0.25 * edges_3ch
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Final sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.2
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.addWeighted(result, 0.75, sharpened, 0.25, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
