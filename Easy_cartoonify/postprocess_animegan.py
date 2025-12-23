import cv2
from pathlib import Path

def unsharp_mask(img, kernel_size=(0,0), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = cv2.absdiff(img, blurred) < threshold
        np_img = img.copy()
        np_img[low_contrast_mask] = img[low_contrast_mask]
        return np_img
    return sharpened

def enhance(path_in, path_out):
    img = cv2.imread(str(path_in))
    if img is None:
        print('Input not found:', path_in)
        return False

    # apply detail enhancement to bring out features
    try:
        detail = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    except Exception:
        detail = img.copy()

    # convert to LAB and apply CLAHE on L channel
    lab = cv2.cvtColor(detail, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    img_clahe = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # unsharp mask
    sharp = unsharp_mask(img_clahe, kernel_size=(0,0), sigma=1.0, amount=1.0, threshold=0)

    # optionally do a slight bilateral filter and blend to reduce artifacts
    bf = cv2.bilateralFilter(sharp, d=5, sigmaColor=75, sigmaSpace=75)
    out = cv2.addWeighted(sharp, 0.85, bf, 0.15, 0)

    cv2.imwrite(str(path_out), out)
    return True

if __name__ == '__main__':
    p = Path(__file__).parent
    src = p / 'animegan_result.jpg'
    out = p / 'animegan_result_sharp.jpg'
    ok = enhance(src, out)
    if ok:
        print('Wrote', out.name)
    else:
        print('Failed to postprocess')
