import cv2
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np

def detect_face_rect(img):
    # use OpenCV Haar cascade
    casc_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(casc_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # pick largest
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

def run_on_face(model, device, face_img_pil):
    input_t = pil_to_tensor(face_img_pil).to(device)
    with torch.no_grad():
        out = model(input_t)
        if isinstance(out, (list, tuple)):
            out = out[0]
    out_img = tensor_to_pil(out)
    return out_img

def main():
    p = Path(__file__).parent
    # original image
    candidates = [f for f in (p / 'user_image.jpg', p / 'test.jpg') if f.exists()]
    if candidates:
        src = candidates[0]
    else:
        # fallback to newest non-generated
        imgs = [f for ext in ('*.jpg','*.jpeg','*.png','*.webp') for f in p.glob(ext)]
        imgs = [f for f in imgs if not f.name.startswith('animegan')]
        if not imgs:
            print('No input image')
            return
        src = max(imgs, key=lambda x: x.stat().st_mtime)

    img_bgr = cv2.imread(str(src))
    if img_bgr is None:
        print('Failed to read', src)
        return

    rect = detect_face_rect(img_bgr)
    if rect is None:
        print('No face detected — running full-image AnimeGAN instead')
        # fallback to full run
        from run_animegan import main as run_full
        run_full()
        return

    x1,y1,x2,y2 = expand_rect(rect, img_bgr.shape, scale=1.6)
    face_crop = img_bgr[y1:y2, x1:x2]
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    # resize to 512
    face_pil_resized = face_pil.resize((512,512), Image.BICUBIC)

    device = 'cpu'
    print('Loading AnimeGANv2 model...')
    model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator', pretrained='face_paint_512_v2')
    model.to(device)
    model.eval()

    print('Running model on face crop...')
    out_pil = run_on_face(model, device, face_pil_resized)
    # resize back to crop size
    out_resized = out_pil.resize((x2-x1, y2-y1), Image.BICUBIC)
    out_bgr = cv2.cvtColor(np.array(out_resized), cv2.COLOR_RGB2BGR)

    # blend with original using an elliptical mask
    h_c, w_c = out_bgr.shape[:2]
    mask = np.zeros((h_c, w_c), dtype=np.uint8)
    cv2.ellipse(mask, (w_c//2, h_c//2), (int(w_c*0.45), int(h_c*0.55)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31,31), 0)
    mask_f = mask.astype(float)/255.0

    res = img_bgr.copy()
    roi = res[y1:y2, x1:x2].astype(float)
    blended = (out_bgr.astype(float) * mask_f[:,:,None] + roi * (1-mask_f)[:,:,None]).astype(np.uint8)
    res[y1:y2, x1:x2] = blended

    out_path = p / 'animegan_face_aligned.jpg'
    cv2.imwrite(str(out_path), res)
    print('Wrote', out_path.name)
    try:
        import subprocess
        subprocess.Popen(['start', str(out_path)], shell=True)
    except Exception:
        pass

if __name__ == '__main__':
    main()
