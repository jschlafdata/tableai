from __future__ import annotations

import json
from typing import List, Dict, Optional, Union, Any, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from pydantic import BaseModel
import hashlib

# Assuming your refactored PDFModel is available
from tableai.pdf.pdf_model import PDFModel

# Local inference - will be updated to return a Pydantic model
from inference import DetectronTableDetector, DetectronDetectionResult, PageDimensions

import PIL.Image
# Check if the old constants are missing (i.e., we are on Pillow >= 10.0.0)
if not hasattr(PIL.Image, 'LINEAR'):
    print("-> Monkey-patching PIL.Image for Pillow v10.0.0+ compatibility with older libraries.")
    # Add the old constants back, pointing to the new Resampling enum
    PIL.Image.LINEAR = PIL.Image.Resampling.BILINEAR # LINEAR is an alias for BILINEAR
    PIL.Image.CUBIC = PIL.Image.Resampling.BICUBIC
    PIL.Image.NEAREST = PIL.Image.Resampling.NEAREST

# ────────────────────────────────────────────────────────────────
# FastAPI setup
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TableAI Detectron2 Detection Service",
    description="Detectron2-based table (and region) detection service",
    version="0.2.0" # Version bump for refactor
)

# A simple cache for Detectron models, keyed by dataset name
_detector_cache: dict[str, DetectronTableDetector] = {}

# ────────────────────────────────────────────────────────────────
# Pydantic Models for API Layer
# ────────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    model_used: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    cached_models: List[str]
    supported_datasets: List[str]

def _get_detector(dataset: str, config_opts: Optional[Dict[str, Any]] = None) -> DetectronTableDetector:
    """
    Returns a cached DetectronTableDetector instance. A new instance is created
    if no detector with the exact same dataset and config options exists.
    """
    # Create a unique signature for this configuration
    config_str = json.dumps(config_opts, sort_keys=True) if config_opts else "{}"
    cache_key = f"{dataset}_{hashlib.sha256(config_str.encode()).hexdigest()[:8]}"

    if cache_key not in _detector_cache:
        print(f"-> No cached model found for key '{cache_key}'. Creating new instance.")
        _detector_cache[cache_key] = DetectronTableDetector(
            dataset=dataset,
            extra_config_opts=config_opts
        )
    return _detector_cache[cache_key]

# ────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        service="tableai-detectron-service",
        version="0.2.0",
        cached_models=list(_detector_cache.keys()),
        supported_datasets=list(DetectronTableDetector.list_available_datasets())
    )


# ─────────────────── CLEANED UP ENDPOINT ───────────────────────
@app.post(
    "/detect/png",
    response_model=Union[DetectronDetectionResult, ErrorResponse], # Assuming ErrorResponse is your error model
    tags=["Detection"]
)
async def detect_png(
    image_file: UploadFile = File(..., description="PNG image of a single document page."),
    pdf_file: UploadFile = File(..., description="The original PDF file for coordinate mapping."),
    page_num: int = Form(..., description="The 0-indexed page number the image corresponds to."),
    dataset: str = Form("PubLayNet", description="Which Detectron2 model to use."),
    zoom: float = Form(2.0, description="The zoom factor used to generate the image."),
    selector: Optional[str] = Form(None, description="The block type to treat as the primary detection target."),
    model_overrides: Optional[str] = Form(None, description="JSON string of Detectron2 config options (e.g., '{\"MODEL.ROI_HEADS.SCORE_THRESH_TEST\": 0.7}')."),
):
    """
    Detects layout regions in a single page image using Detectron2,
    dynamically configured via `model_overrides`.
    """
    config_dict = None
    try:
        # Validate JSON early
        if model_overrides:
            config_dict = json.loads(model_overrides)

        if pdf_file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Uploaded 'pdf_file' must be a PDF.")

        pdf_bytes = await pdf_file.read()
        pdf_model = PDFModel.from_bytes(pdf_bytes, name=pdf_file.filename)

        if not (0 <= page_num < pdf_model.doc.page_count):
            raise HTTPException(status_code=400, detail=f"Invalid page_num {page_num}. PDF has {pdf_model.doc.page_count} pages.")

        image_bytes = await image_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if np_img is None:
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image file.")

        # Get a detector instance for this specific configuration
        detector = _get_detector(dataset=dataset, config_opts=config_dict)

        result = detector.run_on_image(
            np_img=np_img,
            pdf_model=pdf_model,
            page_num=page_num,
            zoom=zoom,
            selector=selector
        )
        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in model_overrides.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        return ErrorResponse(
            message=f"An unexpected error occurred: {str(exc)}",
            model_used=dataset
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)