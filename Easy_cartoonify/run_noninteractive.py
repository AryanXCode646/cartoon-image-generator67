import cv2
from pathlib import Path

# Non-interactive runner for Easy_cartoonify project.
# Reads `test.jpg` in the project folder and writes two outputs:
# `cartoon_1.jpg` and `cartoon_2.jpg`.

def main():
    p = Path(__file__).parent
    src = p / 'test.jpg'
    if not src.exists():
        print(f"Source image not found: {src}")
        return

    img = cv2.imread(str(src))
    if img is None:
        print(f"Failed to read image: {src}")
        return

    print(f"Processing image: {src.name}")

    out1 = cv2.stylization(img, sigma_s=150, sigma_r=0.25)
    out2 = cv2.stylization(img, sigma_s=60, sigma_r=0.5)

    out1_path = p / 'cartoon_1.jpg'
    out2_path = p / 'cartoon_2.jpg'

    cv2.imwrite(str(out1_path), out1)
    cv2.imwrite(str(out2_path), out2)

    print(f"Wrote: {out1_path.name}")
    print(f"Wrote: {out2_path.name}")


if __name__ == '__main__':
    main()
