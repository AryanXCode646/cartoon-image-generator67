import cv2
from pathlib import Path
import shutil

def find_input(p: Path):
    user_src = p / 'user_image.jpg'
    fallback = p / 'test.jpg'
    if user_src.exists():
        return user_src
    if fallback.exists():
        return fallback
    # pick newest non-generated image
    candidates = []
    for ext in ('*.jpg','*.jpeg','*.png','*.webp'):
        for f in p.glob(ext):
            name = f.name.lower()
            if name.startswith('cartoon_') or name.startswith('cartoon_user_'):
                continue
            if f.name in ('test.jpg','user_image.jpg'):
                continue
            candidates.append(f)
    if candidates:
        return max(candidates, key=lambda x: x.stat().st_mtime)
    return None

def cartoonify(img):
    # img: BGR numpy array
    # reduce size for speed (while keeping aspect)
    h, w = img.shape[:2]
    max_dim = 1000
    if max(h,w) > max_dim:
        scale = max_dim / max(h,w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # Color smoothing: repeated bilateral filter
    color = img.copy()
    for i in range(3):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    # Optional detail enhancement
    try:
        color = cv2.detailEnhance(color, sigma_s=10, sigma_r=0.15)
    except Exception:
        pass

    # Edge mask: grayscale -> median -> adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)

    # Convert edges to color and combine
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # reduce edge thickness by morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Blend color with edges: use bitwise and so edges remain black
    cartoon = cv2.bitwise_and(color, edges_color)

    # If the bitwise made the image too dark, combine more softly
    cartoon = cv2.addWeighted(color, 0.8, cartoon, 0.2, 0)

    return cartoon

def main():
    p = Path(__file__).parent
    src = find_input(p)
    if not src:
        print('No input image found to process.')
        return
    print(f'Using input: {src.name}')
    img = cv2.imread(str(src))
    if img is None:
        print('Failed to read input image')
        return

    out = cartoonify(img)
    out_path = p / 'improved_cartoon_user.jpg'
    cv2.imwrite(str(out_path), out)
    print(f'Wrote {out_path.name}')

if __name__ == '__main__':
    main()
