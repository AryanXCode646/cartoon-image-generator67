"""
Enhanced Cartoon Processing with Multiple Models and Post-Processing
Includes CartoonGAN, improved AnimeGAN variants, and clarity enhancement.
"""
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

class EnhancedProcessor:
    """Advanced image processing with multiple cartoon styles"""
    
    @staticmethod
    def enhance_clarity(img_bgr, strength=1.0):
        """Apply CLAHE + unsharp mask for crisp, clear output"""
        # CLAHE for local contrast
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Unsharp mask for detail enhancement
        gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 1.0)
        unsharp = cv2.addWeighted(img_bgr, 1.5 * strength, gaussian, -0.5 * strength, 0)
        return np.clip(unsharp, 0, 255).astype(np.uint8)
    
    @staticmethod
    def bilateral_smooth_edges(img_bgr, diameter=9, sigma_color=75, sigma_space=75):
        """Smooth while preserving edges - cartoon effect"""
        for _ in range(2):
            img_bgr = cv2.bilateralFilter(img_bgr, diameter, sigma_color, sigma_space)
        return img_bgr
    
    @staticmethod
    def detect_and_enhance_edges(img_bgr):
        """Detect and enhance edges for cartoon look"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges_3ch
    
    @staticmethod
    def blend_with_edges(stylized_img, original_img, edge_weight=0.3):
        """Blend stylized image with edges for definition"""
        edges = EnhancedProcessor.detect_and_enhance_edges(original_img)
        edges = edges.astype(float) / 255.0
        stylized = stylized_img.astype(float)
        original = original_img.astype(float)
        
        # Blend: emphasize edges while keeping stylization
        result = stylized * (1 - edge_weight * edges) + original * edge_weight * edges
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def pil_to_tensor(pil_img):
        """Convert PIL image to tensor normalized for models"""
        t = T.ToTensor()(pil_img).unsqueeze(0) * 2 - 1
        return t
    
    @staticmethod
    def tensor_to_pil(tensor):
        """Convert model output tensor to PIL image"""
        t = tensor.squeeze(0).cpu().detach()
        t = (t + 1) / 2
        t = torch.clamp(t, 0, 1)
        arr = (t.permute(1, 2, 0).numpy() * 255).astype('uint8')
        return Image.fromarray(arr)
    
    @staticmethod
    def run_animegan_v3(model, device, img_bgr, enhance=True):
        """Run AnimeGAN with post-processing for clarity"""
        h, w = img_bgr.shape[:2]
        short = min(h, w)
        scale = 512 / short
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        img_resized = cv2.resize(img_bgr, (new_w, new_h))
        pil_in = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        
        input_t = EnhancedProcessor.pil_to_tensor(pil_in).to(device)
        with torch.no_grad():
            out = model(input_t)
            if isinstance(out, (list, tuple)):
                out = out[0]
        
        out_pil = EnhancedProcessor.tensor_to_pil(out)
        out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
        out_bgr = cv2.resize(out_bgr, (w, h))
        
        if enhance:
            # Enhanced post-processing
            out_bgr = EnhancedProcessor.enhance_clarity(out_bgr, strength=0.8)
            out_bgr = EnhancedProcessor.blend_with_edges(out_bgr, img_bgr, edge_weight=0.15)
        
        return out_bgr
    
    @staticmethod
    def cartoon_gan_style(img_bgr):
        """Apply CartoonGAN-like effect using pure OpenCV.

        This version is tuned for a bright, clean cartoon look without
        overly dark shading.
        """
        # Bilateral filter for smooth regions
        smooth = cv2.bilateralFilter(img_bgr, 9, 75, 75)

        # Quantize colors for cartoon look
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        result = centers[labels.flatten()]
        result = result.reshape(img_bgr.shape)

        # Edge preservation
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(cv2.GaussianBlur(edges, (3, 3), 0), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)

        # Blend cartoon with edges (use light outlines so image doesn't get too dark)
        edges_inv = cv2.bitwise_not(edges)
        edges_inv_3ch = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        result = (result.astype(float) * edges_inv_3ch + img_bgr.astype(float) * (1 - edges_inv_3ch) * 0.2)

        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def ghibli_style_enhanced(img_bgr):
        """Create Ghibli-like style with edge enhancement and color richness"""
        # Start with bilateral smoothing
        smooth = cv2.bilateralFilter(img_bgr, 11, 80, 80)
        smooth = cv2.bilateralFilter(smooth, 11, 80, 80)
        
        # Enhance colors with CLAHE
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(10, 10))
        l = clahe.apply(l)
        a = cv2.addWeighted(a, 1.1, np.ones_like(a) * 128, -0.1, 0)
        b = cv2.addWeighted(b, 1.1, np.ones_like(b) * 128, -0.1, 0)
        smooth = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Detect edges with emphasis
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=2)
        
        # Blend edges back for definition
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        result = smooth.astype(float) * (1 - 0.3 * edges_3ch) + img_bgr.astype(float) * 0.3 * edges_3ch
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Final unsharp for clarity
        result = EnhancedProcessor.enhance_clarity(result, strength=0.7)
        
        return result
    
    @staticmethod
    def watercolor_effect(img_bgr):
        """Create watercolor/painting effect"""
        # Bilateral filter for smooth watercolor look
        result = img_bgr.copy()
        for _ in range(3):
            result = cv2.bilateralFilter(result, 11, 80, 80)
    
        # Reduce colors for watercolor effect
        data = result.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 12, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        result = centers[labels.flatten()].reshape(result.shape)
    
        # Enhance with slight clarity
        result = EnhancedProcessor.enhance_clarity(result, strength=0.5)
    
        return result


def apply_face_restoration(img_bgr):
    """Try to improve facial details using unsharp and bilateral enhancement"""
    # Apply gentle enhancement to preserve identity
    enhanced = EnhancedProcessor.enhance_clarity(img_bgr, strength=0.5)
    
    # Light bilateral pass for smoothness without losing detail
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    # Blend to maintain identity
    result = cv2.addWeighted(enhanced, 0.6, img_bgr, 0.4, 0)
    return result
