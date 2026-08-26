<div align="center">

# 🎨 Cartoonify Studio Pro
### Professional AI & Computer Vision Image Cartoonization Suite

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  Transform ordinary photos, portraits, and landscapes into stunning anime, Studio Ghibli, pop-art comic, and watercolor artwork. Features an ultra-modern <strong>Web Studio</strong> with an interactive Before/After split slider, a sleek <strong>Desktop GUI</strong>, and a powerful <strong>Command-Line Interface (CLI)</strong>.
</p>

</div>

---

## 🌟 Visual Style Showcase

| Style | Category | Description | Accent |
| :--- | :---: | :--- | :---: |
| **🎬 Studio Ghibli Pro** | *Artistic CV* | Lush hand-painted animation aesthetic with warm color harmonies and soft atmospheric lighting. | `#FF6B6B` |
| **✨ Anime Portrait Soft** | *Neural AI* | State-of-the-art neural anime stylization with expressive eyes and crisp face preservation. | `#4ECDC4` |
| **🎨 Paprika Vibrant** | *Neural AI* | Warm saturated cinematic cartoon look inspired by Satoshi Kon feature animation. | `#FFB84D` |
| **🍃 Hayao Anime** | *Neural AI* | Miyazaki-inspired vivid green fields, brilliant blue skies, and cinematic lighting. | `#2ECC71` |
| **🌌 Shinkai Luminous** | *Neural AI* | Makoto Shinkai high-contrast radiant lighting and glowing atmospheric accents. | `#9B59B6` |
| **💥 Comic Pop Art** | *Artistic CV* | Bold graphic novel inked contours with vibrant cell-shaded primary colors. | `#E74C3C` |
| **🎭 Watercolor Dream** | *Artistic CV* | Soft pigment diffusion, gentle paper texture wash, and fluid color bleeding. | `#A8D8EA` |
| **✏️ Pencil & Charcoal** | *Artistic CV* | Detailed graphite crosshatch line work with smooth continuous shading. | `#95A5A6` |
| **🖍️ Colored Pencil** | *Artistic CV* | Hand-drawn color sketch with textured paper grain and vibrant strokes. | `#F39C12` |
| **⚡ Neon Cyberpunk** | *Artistic CV* | Darkened cinematic scene with electric cyan and magenta glowing edges. | `#00F0FF` |
| **📺 Retro 90s TV Toon** | *Artistic CV* | Nostalgic Saturday morning cartoon aesthetic with thick black ink borders. | `#FD79A8` |
| **🎛️ Custom Pro Shader** | *Parametric* | Real-time tunable parameters: line thickness, opacity, color count, saturation, and contrast. | `#00CEC9` |

---

## 🚀 Key Features

- **🌐 Modern Web Studio**: Glassmorphism UI with interactive drag-and-drop, real-time Before/After comparison slider, live camera snapshot, and responsive dark/light themes.
- **✨ Showcase Landing Page Website (`docs/`)**: Zero-dependency static showcase website ready for GitHub Pages with interactive client-side Canvas simulator.
- **🖥️ Desktop GUI**: Cross-platform responsive desktop app with side-by-side comparison, non-blocking asynchronous processing threads, and history gallery.
- **⚡ Neural & CV Engine**: Combines deep learning (PyTorch Hub AnimeGANv2) and high-performance OpenCV algorithms for instant, offline-capable rendering.
- **👤 Face-Preserved Alignment**: Smart face detection with elliptical feathered alpha masking to preserve identity on portrait photos.
- **📦 Batch Processing**: Convert entire image directories in seconds and export directly as a ZIP archive.
- **🧪 100% Test Coverage**: Complete Pytest test suite and GitHub Actions automated CI workflows.

---

## 🛠️ Quick Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AryanXCode646/cartoon-image-generator67.git
cd cartoon-image-generator67
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# Or install in editable developer mode:
pip install -e .
```

---

## 🎯 How to Run

### 🌐 1. Launch the Web Studio (Recommended)
Run the 1-click script or CLI command:
```bash
# Windows 1-click launcher:
run_web.bat

# Or via CLI:
python -m cartoonify web
```
Then open your browser at **`http://127.0.0.1:8000`**.

---

### 🖥️ 2. Launch the Desktop GUI
```bash
# Windows 1-click launcher:
run_gui.bat

# Or via CLI:
python -m cartoonify gui
```

---

### 💻 3. Command-Line Interface (CLI)

#### Convert a Single Image:
```bash
python -m cartoonify process input.jpg -o output.jpg --style ghibli_pro --strength 0.85
```

#### Face-Preserved Portrait Mode:
```bash
python -m cartoonify process portrait.jpg -o cartoon_portrait.jpg --style anime_soft --face-align
```

#### Batch Process an Entire Directory:
```bash
python -m cartoonify batch ./photos -o ./cartoons --style comic_pop
```

#### List All Available Styles:
```bash
python -m cartoonify list-styles
```

---

## 🐍 Python SDK / API Usage

You can easily integrate Cartoonify into your own Python applications:

```python
from cartoonify import CartoonEngine

# Initialize the engine (automatically detects CUDA GPU or CPU)
engine = CartoonEngine()

# 1. Transform an image file directly
output_path = engine.process_file(
    "portrait.jpg",
    "cartoon_output.jpg",
    style="ghibli_pro",
    strength=0.85,
    use_face_align=True
)
print(f"Saved artwork to: {output_path}")

# 2. In-Memory OpenCV Numpy Array Processing
import cv2

image_bgr = cv2.imread("landscape.jpg")
cartoon_bgr = engine.process_image(
    image_bgr,
    style="watercolor",
    strength=0.8
)
cv2.imwrite("watercolor_landscape.jpg", cartoon_bgr)

# 3. Custom Parametric Shader
custom_artwork = engine.process_image(
    image_bgr,
    style="custom",
    custom_params={
        "line_thickness": 2,
        "line_opacity": 0.8,
        "num_colors": 12,
        "saturation": 1.4,
        "contrast": 1.2,
        "sharpness": 0.9,
    }
)
```

---

## 🧪 Testing & Verification

Run the comprehensive unit test suite:

```bash
# Windows 1-click test runner:
run_tests.bat

# Or via Pytest:
python -m pytest tests/ -v
```

---

## 📁 Repository Structure

```
cartoon-image-generator67/
├── .github/workflows/ci.yml       # Automated GitHub Actions CI pipeline
├── cartoonify/                    # Core Python package
│   ├── __init__.py               # Package exports & version
│   ├── engine.py                 # Central CartoonEngine & Style Registry
│   ├── cli.py                    # Unified CLI commands
│   ├── utils.py                  # Image I/O, resizing & history manager
│   ├── filters/                  # Modular transformation filters
│   │   ├── artistic.py           # Ghibli Pro, Watercolor, Comic, Neon, Pencil
│   │   ├── neural.py             # PyTorch AnimeGANv2 model wrapper & caching
│   │   ├── classic.py            # OpenCV bilateral & edge filters
│   │   ├── custom.py             # Real-time parametric shader
│   │   └── face.py               # Face detection & feathered blending
│   ├── api/                      # FastAPI REST API backend
│   │   └── app.py                # Endpoints for processing, batch, & health
│   ├── web/                      # Ultra-Modern Web Studio
│   │   ├── index.html            # Glassmorphic single-page app
│   │   ├── style.css             # Theme system & responsive layout
│   │   └── app.js                # Split slider, webcam, & batch controller
│   └── gui/                      # Desktop GUI
│       ├── app.py                # Tkinter desktop application
│       └── theme.py              # Dark & light theme tokens
├── Easy_cartoonify/              # Backward compatibility layer
├── tests/                        # Comprehensive test suite
│   ├── test_engine.py            # Tests all 12 styles
│   ├── test_filters.py           # Tests individual filter algorithms
│   ├── test_api.py               # Tests REST API routes
│   └── test_cli.py               # Tests CLI command parsing
├── pyproject.toml                # Modern Python packaging configuration
├── setup.py                      # Setuptools configuration
├── requirements.txt              # Pinned core dependencies
├── run_web.bat                   # 1-click Web Studio launcher
├── run_gui.bat                   # 1-click Desktop GUI launcher
├── run_cli.bat                   # 1-click CLI helper
├── run_tests.bat                 # 1-click Pytest runner
└── README.md                     # Documentation & showcase
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
