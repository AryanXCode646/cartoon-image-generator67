"""
FastAPI Backend for Cartoonify Web Studio with Enterprise Security Hardening.
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cartoonify.engine import STYLES, CartoonEngine
from cartoonify.utils import (
    MAX_FILE_BYTES,
    HistoryManager,
    base64_to_image,
    image_to_base64,
    image_to_bytes,
    load_image,
    sanitize_filename,
)

logger = logging.getLogger("cartoonify.api")

app = FastAPI(
    title="Cartoonify Studio API",
    description="Enterprise-Grade AI & Computer Vision Cartoonization Backend",
    version="2.0.0",
)

# Security Middleware: CORS and Security Headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


engine = CartoonEngine()
history_manager = HistoryManager()
WEB_DIR = Path(__file__).parent.parent / "web"


class ProcessRequest(BaseModel):
    image: str = Field(..., description="Base64 data URL of the image")
    style: str = Field("ghibli_pro", max_length=50)
    strength: float = Field(0.8, ge=0.0, le=1.0)
    use_face_align: bool = Field(False)
    custom_params: Optional[Dict[str, Any]] = None
    max_dimension: Optional[int] = Field(1600, ge=32, le=8192)


@app.get("/api/styles")
async def get_styles():
    """Retrieve catalog of all cartoon styles."""
    return [style.to_dict() for style in engine.list_styles()]


@app.post("/api/process")
async def process_image_json(req: ProcessRequest):
    """Process an image passed as a Base64 data URL string with security validation."""
    try:
        t0 = time.time()
        img_bgr = base64_to_image(req.image)
        orig_h, orig_w = img_bgr.shape[:2]

        result_bgr = engine.process_image(
            img_bgr,
            style=req.style,
            strength=req.strength,
            use_face_align=req.use_face_align,
            custom_params=req.custom_params,
            max_dimension=req.max_dimension,
        )
        elapsed = round(time.time() - t0, 3)

        out_b64 = image_to_base64(result_bgr, format_ext=".jpg", quality=95)
        thumb_b64 = image_to_base64(
            cv2.resize(result_bgr, (128, 128), interpolation=cv2.INTER_AREA),
            format_ext=".jpg",
            quality=80,
        )

        style_cfg = engine.get_style(req.style)
        history_manager.add_record(
            style=req.style,
            style_name=style_cfg.name,
            input_info=f"{orig_w}x{orig_h}",
            thumbnail_b64=thumb_b64,
            parameters={
                "strength": req.strength,
                "face_align": req.use_face_align,
                "elapsed_seconds": elapsed,
            },
        )

        return {
            "success": True,
            "image": out_b64,
            "style": req.style,
            "style_name": style_cfg.name,
            "elapsed_seconds": elapsed,
            "width": result_bgr.shape[1],
            "height": result_bgr.shape[0],
        }
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal image processing error.")


@app.post("/api/process-upload")
async def process_image_upload(
    file: UploadFile = File(...),
    style: str = Form("ghibli_pro"),
    strength: float = Form(0.8),
    use_face_align: bool = Form(False),
    custom_params_json: Optional[str] = Form(None),
):
    """Process an uploaded multipart image file safely and return raw JPEG binary."""
    try:
        content = await file.read()
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds maximum allowed size (50MB).")

        img_bgr = load_image(content)

        custom_params = None
        if custom_params_json:
            try:
                custom_params = json.loads(custom_params_json)
            except Exception:
                custom_params = None

        result_bgr = engine.process_image(
            img_bgr,
            style=style,
            strength=strength,
            use_face_align=use_face_align,
            custom_params=custom_params,
        )

        jpeg_bytes = image_to_bytes(result_bgr, format_ext=".jpg", quality=95)
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Image upload processing failed: {e}")


@app.post("/api/batch-zip")
async def process_batch_zip(
    files: List[UploadFile] = File(...),
    style: str = Form("ghibli_pro"),
    strength: float = Form(0.8),
    use_face_align: bool = Form(False),
):
    """Process multiple uploaded images and return a secure ZIP archive (DoS / Zip Slip protected)."""
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files per batch request.")

    try:
        zip_buffer = io.BytesIO()
        total_uncompressed = 0
        MAX_BATCH_UNCOMPRESSED = 200 * 1024 * 1024  # 200MB limit

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in files:
                content = await file.read()
                if len(content) > MAX_FILE_BYTES:
                    continue

                try:
                    img_bgr = load_image(content)
                    result_bgr = engine.process_image(
                        img_bgr,
                        style=style,
                        strength=strength,
                        use_face_align=use_face_align,
                    )
                    out_bytes = image_to_bytes(result_bgr, format_ext=".jpg", quality=95)

                    total_uncompressed += len(out_bytes)
                    if total_uncompressed > MAX_BATCH_UNCOMPRESSED:
                        raise HTTPException(status_code=413, detail="Batch output size exceeded limit.")

                    safe_stem = sanitize_filename(Path(file.filename or "image").stem)
                    zf.writestr(f"cartoon_{style}_{safe_stem}.jpg", out_bytes)
                except Exception as file_err:
                    logger.warning(f"Skipping {file.filename}: {file_err}")

        zip_buffer.seek(0)
        safe_style_name = sanitize_filename(style)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=cartoonify_batch_{safe_style_name}.zip"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history():
    """Retrieve recent generation records."""
    return history_manager.load()


@app.delete("/api/history")
async def clear_history():
    """Clear generation history."""
    history_manager.clear()
    return {"success": True, "message": "History cleared"}


@app.get("/api/health")
async def health_check():
    """API health and diagnostic status."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "device": engine.neural_model.device,
        "torch_available": engine.neural_model.is_torch_available(),
    }


# Serve Web UI files
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index_path = WEB_DIR / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return "<h1>Cartoonify Studio</h1><p>Web UI assets loading...</p>"
