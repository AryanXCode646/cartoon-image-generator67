import cv2
import numpy as np
from pathlib import Path

def find_input(p: Path):
    for name in ('user_image.jpg','test.jpg'):
        f = p / name
        if f.exists():
            return f
    # fallback: pick newest non-generated image
    candidates = [f for ext in ('*.jpg','*.jpeg','*.png','*.webp') for f in p.glob(ext)]
    candidates = [f for f in candidates if not f.name.startswith('cartoon') and not f.name.startswith('improved')]
    if candidates:
        return max(candidates, key=lambda x: x.stat().st_mtime)
    return None

def color_quantize(img, k=8):
    data = img.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    quant = centers[labels.flatten()].reshape(img.shape)
    return quant

def variant_v2(img):
    from improved_cartoon_v2 import cartoon_v2
    return cartoon_v2(img)

def variant_quant_edges(img):
    q = color_quantize(img, k=12)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 140)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations=1)
    edges_col = cv2.cvtColor(cv2.bitwise_not(edges), cv2.COLOR_GRAY2BGR)
    out = cv2.bitwise_and(q, edges_col)
    out = cv2.addWeighted(q, 0.8, out, 0.2, 0)
    return out

def variant_stylize_quant(img):
    styl = cv2.stylization(img, sigma_s=60, sigma_r=0.45)
    quant = color_quantize(styl, k=10)
    return quant

def variant_pencil_color(img):
    # use pencilSketch then blend with color quantized
    gray_sketch, color_sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    quant = color_quantize(img, k=12)
    blended = cv2.addWeighted(quant, 0.7, cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR), 0.3, 0)
    return blended

def main():
    p = Path(__file__).parent
    src = find_input(p)
    if not src:
        print('No input image found')
        return
    print('Using', src.name)
    img = cv2.imread(str(src))
    if img is None:
        print('Failed to read image')
        return

    variants = [
        ('v2', variant_v2(img)),
        ('quant_edges', variant_quant_edges(img)),
        ('stylize_quant', variant_stylize_quant(img)),
        ('pencil_color', variant_pencil_color(img)),
    ]

    out_files = []
    for name, out in variants:
        out_path = p / f'variant_{name}.jpg'
        cv2.imwrite(str(out_path), out)
        out_files.append(out_path)
        print('Wrote', out_path.name)

    # open the stylized quant (likely closest) by default
    try:
        import subprocess, os
        subprocess.Popen(['start', str(out_files[2])], shell=True)
    except Exception:
        pass

if __name__ == '__main__':
    main()
