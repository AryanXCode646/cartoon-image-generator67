# Cartoon Image Generator - GUI App

An interactive desktop application to transform photos into cartoon/anime artwork with real-time preview and customizable styles.

## Features

✨ **Multiple Cartoon Styles**
- **Anime (Soft)** - Soft anime style with face preservation
- **Paprika** - Warm, vibrant cartoon style
- **Full Stylize** - Full-image anime transformation

🎨 **Intelligent Processing**
- Face detection and alignment for identity preservation
- Adjustable face blend strength (0.0 - 1.0)
- Real-time preview while processing
- Full-image fallback if no face detected

💾 **User-Friendly Interface**
- Drag-and-drop image loading
- Side-by-side input/output preview
- Save results to JPEG or PNG
- Progress indicators

## Installation & Setup

### 1. Install Dependencies

```bash
cd "C:\Users\aryan\Desktop\cartoon image generater\Easy_cartoonify"
.\.venv\Scripts\pip install opencv-python torch pillow torchvision
```

### 2. Launch the App

```bash
.\.venv\Scripts\python.exe launch_app.py
```

Or directly:

```bash
.\.venv\Scripts\python.exe app_gui.py
```

## Usage

1. **Load Image**: Click "📁 Load Image" button to select a photo
2. **Choose Style**: Select one of the cartoon styles (Anime Soft, Paprika, or Full Stylize)
3. **Adjust Blend**: Use the slider to control how strongly the face is stylized (higher = stronger effect)
4. **Generate**: Click "✨ Generate Cartoon" to process the image
5. **Save**: Once complete, click "💾 Save Result" to export the cartoon image

## How It Works

### Face-Aligned Processing (Default)
- Detects the largest face using OpenCV Haar cascade
- Crops face region with ~1.6x expansion
- Resizes to 512×512 and runs AnimeGAN model
- Composites back using soft elliptical mask for natural blending

### Full-Image Processing
- Resizes image preserving aspect ratio to 512px short side
- Runs AnimeGAN model on entire image
- Returns full-image cartoon result

## Supported Models

- **face_paint_512_v2** - Soft anime style trained on painted faces
- **paprika** - Warm vibrant cartoon style

## Performance

- First run downloads ~100MB model weights (cached afterward)
- Processing time: 10-30 seconds depending on image size
- CPU-based inference (optimized for CPU)

## Output Examples

Input: Portrait photo  
Output: Cartoon portrait with anime aesthetic

See `animegan_variant_*.jpg` files in the project folder for example outputs.

## Technical Details

- **Framework**: Tkinter (cross-platform GUI)
- **ML Model**: AnimeGANv2 via PyTorch
- **Image Processing**: OpenCV
- **Threading**: Asynchronous processing to keep UI responsive

## Troubleshooting

**"Model not found" error**
- First run requires internet to download model weights (~100MB)
- Weights are cached for subsequent runs

**"No face detected"**
- The app falls back to full-image stylization
- Ensure face is clearly visible and not obscured

**App is slow/unresponsive**
- Processing happens in a background thread
- Check "Generate..." status indicator
- Processing time depends on image size

## Tips for Best Results

1. Use clear, well-lit portrait photos
2. Face should occupy ~30-50% of image
3. Try different blend strengths for different effects:
   - 0.3-0.5: Subtle cartoon effect
   - 0.6-0.8: Balanced cartoon style (recommended)
   - 0.9-1.0: Strong full cartoon transformation
4. For non-portrait images, select "Full Stylize" style

## Future Enhancements

- Batch processing
- Additional style checkpoints
- GFPGAN face restoration option
- Real-time parameter preview
- Model switching without restart
