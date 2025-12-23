import cv2
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np

MODEL_NAMES = [
    'face_paint_512_v2',
    'paprika',
    'hayao',
    'shinkai'
]

def detect_face_rect(img):
    casc_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(casc_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    areas = [w*h for (x,y,w,h) in faces]
    idx = int(np.argmax(areas))
    return faces[idx]

def expand_rect(rect, img_shape, scale=1.6):
    x,y,w,h = rect
    cx = x + w/2
    cy = y + h/2
    new_w = w * scale
    new_h = h * scale
    x1 = int(max(0, cx - new_w/2))
    y1 = int(max(0, cy - new_h/2))
    x2 = int(min(img_shape[1], cx + new_w/2))
    y2 = int(min(img_shape[0], cy + new_h/2))
    return x1,y1,x2,y2

def pil_to_tensor(img):
    t = T.ToTensor()(img).unsqueeze(0) * 2 - 1
    return t

def tensor_to_pil(tensor):
    t = tensor.squeeze(0).cpu().detach()
    t = (t + 1) / 2
    t = torch.clamp(t, 0, 1)
    arr = (t.permute(1,2,0).numpy() * 255).astype('uint8')
    return Image.fromarray(arr)

def run_model_on_pil(model, device, pil_img):
    inp = pil_to_tensor(pil_img).to(device)
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, (list, tuple)):
            out = out[0]
    return tensor_to_pil(out)

def run_variant_on_image(model_name, img_bgr, out_dir, device='cpu'):
    try:
        print('Loading', model_name)
        model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator', pretrained=model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print('Skipping', model_name, '(', e, ')')
        return

    # Full-image run (resize to 512 preserving aspect by resize short side)
    h,w = img_bgr.shape[:2]
    short = min(h,w)
    scale = 512/short
    new_w = max(1, int(w*scale))
    new_h = max(1, int(h*scale))
    pil_in = Image.fromarray(cv2.cvtColor(cv2.resize(img_bgr, (new_w,new_h)), cv2.COLOR_BGR2RGB))
    out_pil = run_model_on_pil(model, device, pil_in)
    out_full = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
    out_full_resized = cv2.resize(out_full, (w,h))
    cv2.imwrite(str(out_dir / f'animegan_variant_full_{model_name}.jpg'), out_full_resized)
    print('Wrote full variant', model_name)

    # Face-aligned variant, if face detected
    rect = detect_face_rect(img_bgr)
    if rect is None:
        print('No face detected for face-aligned variant', model_name)
        return
    x1,y1,x2,y2 = expand_rect(rect, img_bgr.shape, scale=1.6)
    face_crop = img_bgr[y1:y2, x1:x2]
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    face_resized = face_pil.resize((512,512), Image.BICUBIC)
    out_face_pil = run_model_on_pil(model, device, face_resized)
    out_face_resized = out_face_pil.resize((x2-x1, y2-y1), Image.BICUBIC)
    out_face_bgr = cv2.cvtColor(np.array(out_face_resized), cv2.COLOR_RGB2BGR)

    # full-replace composite (stronger) and soft blend composite
    res_replace = img_bgr.copy()
    res_replace[y1:y2, x1:x2] = out_face_bgr
    cv2.imwrite(str(out_dir / f'animegan_variant_face_replace_{model_name}.jpg'), res_replace)

    # soft blend
    h_c, w_c = out_face_bgr.shape[:2]
    mask = np.zeros((h_c, w_c), dtype=np.uint8)
    cv2.ellipse(mask, (w_c//2, h_c//2), (int(w_c*0.45), int(h_c*0.55)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31,31), 0)
    mask_f = mask.astype(float)/255.0
    res_blend = img_bgr.copy().astype(float)
    roi = res_blend[y1:y2, x1:x2]
    blended = (out_face_bgr.astype(float) * mask_f[:,:,None] + roi * (1-mask_f)[:,:,None]).astype(np.uint8)
    res_blend = img_bgr.copy()
    res_blend[y1:y2, x1:x2] = blended
    cv2.imwrite(str(out_dir / f'animegan_variant_face_blend_{model_name}.jpg'), res_blend)
    print('Wrote face variants for', model_name)

def main():
    p = Path(__file__).parent
    img_candidates = [p / 'user_image.jpg', p / 'test.jpg']
    src = None
    for c in img_candidates:
        if c.exists():
            src = c
            break
    if src is None:
        imgs = [f for ext in ('*.jpg','*.jpeg','*.png','*.webp') for f in p.glob(ext)]
        imgs = [f for f in imgs if not f.name.startswith('animegan_variant')]
        if not imgs:
            print('No input image found')
            return
        src = max(imgs, key=lambda x: x.stat().st_mtime)

    img = cv2.imread(str(src))
    if img is None:
        print('Failed to read', src)
        return

    out_dir = p
    device = 'cpu'
    for name in MODEL_NAMES:
        run_variant_on_image(name, img, out_dir, device=device)

if __name__ == '__main__':
    main()
