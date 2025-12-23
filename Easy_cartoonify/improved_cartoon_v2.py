import cv2
from pathlib import Path

def find_input(p: Path):
    # reuse same discovery as previous script
    user_src = p / 'user_image.jpg'
    fallback = p / 'test.jpg'
    if user_src.exists():
        return user_src
    if fallback.exists():
        return fallback
    candidates = []
    for ext in ('*.jpg','*.jpeg','*.png','*.webp'):
        for f in p.glob(ext):
            name = f.name.lower()
            if name.startswith('cartoon_') or name.startswith('cartoon_user_'):
                continue
            candidates.append(f)
    if candidates:
        return max(candidates, key=lambda x: x.stat().st_mtime)
    return None

def unsharp_mask(img, radius=1.0, amount=1.0):
    # amount: strength, radius: gaussian blur sigma
    blurred = cv2.GaussianBlur(img, (0,0), radius)
    sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharp

def cartoon_v2(img):
    # keep original size
    img_color = img.copy()

    # gentle bilateral filter to smooth colors but keep detail
    img_color = cv2.bilateralFilter(img_color, d=7, sigmaColor=50, sigmaSpace=50)

    # create edge map using Canny on luminance after small blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)

    # dilate edges slightly to make them more visible
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # invert edges: white background, black edges -> create mask
    edges_inv = cv2.bitwise_not(edges)
    edges_inv_color = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

    # combine color and edges: preserve color but darken where edges are
    # create a mask of edge pixels
    edge_mask = edges > 0
    result = img_color.copy()
    # darken edges on result
    result[edge_mask] = (result[edge_mask] * 0.3).astype(result.dtype)

    # slightly enhance color contrast using CLAHE on L channel
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # sharpen a bit with unsharp mask
    result = unsharp_mask(result, radius=1.0, amount=0.8)

    return result

def main():
    p = Path(__file__).parent
    src = find_input(p)
    if not src:
        print('No input image found')
        return
    print(f'Using input: {src.name}')
    img = cv2.imread(str(src))
    if img is None:
        print('Failed to read image')
        return

    out = cartoon_v2(img)
    out_path = p / 'improved_cartoon_user_v2.jpg'
    cv2.imwrite(str(out_path), out)
    print(f'Wrote {out_path.name}')

if __name__ == '__main__':
    main()
