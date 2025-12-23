import cv2
from pathlib import Path
import shutil

def main():
    p = Path(__file__).parent
    user_src = p / 'user_image.jpg'
    fallback = p / 'test.jpg'

    # If user_image.jpg is not provided, fall back to test.jpg
    if not user_src.exists():
        if fallback.exists():
            shutil.copy2(fallback, user_src)
            print(f"No `user_image.jpg` found — copied fallback `test.jpg` to `user_image.jpg`")
        else:
            # try any common image file in the directory
            # pick the most recently modified image file excluding generated cartoons
            candidates = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                for f in p.glob(ext):
                    name = f.name.lower()
                    if name.startswith('cartoon_') or name.startswith('cartoon_user_'):
                        continue
                    if f.name in ('test.jpg', 'user_image.jpg'):
                        continue
                    candidates.append(f)
            if candidates:
                # pick newest
                user_src = max(candidates, key=lambda x: x.stat().st_mtime)
                print(f"No `user_image.jpg` or `test.jpg`; using newest image: {user_src.name}")
            else:
                print('No input image found (user_image.jpg, test.jpg, or any other image).')
                return
            

    img = cv2.imread(str(user_src))
    if img is None:
        print(f"Failed to read image: {user_src}")
        return

    out1 = cv2.stylization(img, sigma_s=150, sigma_r=0.25)
    out2 = cv2.stylization(img, sigma_s=60, sigma_r=0.5)

    out1_path = p / 'cartoon_user_1.jpg'
    out2_path = p / 'cartoon_user_2.jpg'

    cv2.imwrite(str(out1_path), out1)
    cv2.imwrite(str(out2_path), out2)

    print(f"Wrote: {out1_path.name}")
    print(f"Wrote: {out2_path.name}")

if __name__ == '__main__':
    main()
