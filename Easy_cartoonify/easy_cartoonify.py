"""
Easy Cartoonify - Interactive CLI Runner.
Refactored to be fast, resilient, and error-free without disk freezing.
"""

import sys
from pathlib import Path
import cv2

# Add parent directory to path to enable cartoonify engine import
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cartoonify.engine import CartoonEngine, STYLES


def find_image(file_name: str, directory_name: str) -> Path:
    """Find image in the specified directory or project folder without full-disk freezing."""
    dir_path = Path(directory_name.strip() if directory_name.strip() else ".")
    if not dir_path.exists():
        dir_path = Path(__file__).parent

    target = dir_path / file_name.strip()
    if target.exists() and target.is_file():
        return target

    # Local search up to 2 levels deep
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        for f in dir_path.glob(f"**/{file_name}"):
            if f.is_file():
                return f

    # Fallback to any image in current folder
    candidates = [f for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp") for f in Path(__file__).parent.glob(ext)]
    if candidates:
        print(f"Notice: '{file_name}' not found. Defaulting to local image: {candidates[0].name}")
        return candidates[0]

    raise FileNotFoundError(f"Could not locate image '{file_name}' in {dir_path}")


def main():
    print("=" * 60)
    print("   🎨 Easy Cartoonify Interactive Runner")
    print("=" * 60)

    image_name = input("Please enter the name or path of the image file (or press Enter for default): ").strip()
    if not image_name:
        image_name = "bollywood-actress-kiara-advani-presents-a-creation-by-the-designer-duo-falguni-shane-peacock.webp"

    image_directory = input("Please enter the directory containing the image (press Enter for current folder): ").strip()
    if not image_directory:
        image_directory = str(Path(__file__).parent)

    try:
        image_path = find_image(image_name, image_directory)
        print(f"✅ Loading image from: {image_path}")
    except Exception as e:
        print(f"❌ {e}")
        return

    color_image = cv2.imread(str(image_path))
    if color_image is None:
        print(f"❌ Failed to decode image at {image_path}")
        return

    print("\nSelect a Cartoon Style:")
    print("  1) Classic Smooth (OpenCV Bilateral)")
    print("  2) Classic Sharp (OpenCV Detail)")
    print("  3) Studio Ghibli Pro (Lush Hand-Painted)")
    print("  4) Comic Book Pop Art")
    print("  5) Watercolor Dream")

    style_choice = input("Enter choice (1-5, default: 3): ").strip()
    style_map = {
        "1": "classic_v1",
        "2": "classic_v2",
        "3": "ghibli_pro",
        "4": "comic_pop",
        "5": "watercolor",
    }
    selected_style = style_map.get(style_choice, "ghibli_pro")

    engine = CartoonEngine()
    print(f"\nProcessing image with style '{selected_style}'...")
    cartoon_result = engine.process_image(color_image, style=selected_style, strength=0.85)

    out_path = Path(__file__).parent / f"cartoon_output_{selected_style}.jpg"
    cv2.imwrite(str(out_path), cartoon_result)
    print(f"✨ Artwork saved to: {out_path.name}")

    try:
        cv2.imshow(f"Cartoonify - {selected_style}", cartoon_result)
        print("Press any key in the image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
