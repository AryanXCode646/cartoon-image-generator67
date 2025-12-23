import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np

def load_image(path, size=None):
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    return img

def to_tensor(img):
    t = T.ToTensor()(img).unsqueeze(0) * 2 - 1  # [-1,1]
    return t

def to_image(tensor):
    t = tensor.squeeze(0).cpu().detach()
    t = (t + 1) / 2
    t = torch.clamp(t, 0, 1)
    arr = (t.permute(1,2,0).numpy() * 255).astype('uint8')
    return Image.fromarray(arr)

def main():
    p = Path(__file__).parent
    # choose input image
    src = p / 'user_image.jpg'
    if not src.exists():
        # fallback to newest image
        candidates = [f for ext in ('*.jpg','*.jpeg','*.png','*.webp') for f in p.glob(ext)]
        candidates = [f for f in candidates if not f.name.startswith('variant') and not f.name.startswith('cartoon') and not f.name.startswith('improved')]
        if not candidates:
            print('No input image found for AnimeGAN')
            return
        src = max(candidates, key=lambda x: x.stat().st_mtime)

    print('Using input:', src.name)

    device = 'cpu'

    print('Loading AnimeGANv2 model from torch.hub (this will download weights)...')
    try:
        model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator', pretrained='face_paint_512_v2')
        model.to(device)
    except Exception as e:
        print('Failed to load model via torch.hub:', e)
        return

    model.eval()

    img = load_image(src, size=512)
    input_t = to_tensor(img).to(device)

    with torch.no_grad():
        try:
            out = model(input_t)
        except Exception as e:
            # some hub models return tuple
            try:
                out = model(input_t)[0]
            except Exception:
                print('Model inference failed:', e)
                return

    out_img = to_image(out)
    out_path = p / 'animegan_result.jpg'
    out_img.save(out_path)
    print('Wrote', out_path.name)

    # open result on Windows
    try:
        import subprocess
        subprocess.Popen(['start', str(out_path)], shell=True)
    except Exception:
        pass

if __name__ == '__main__':
    main()
