from __future__ import annotations

import json
from typing import List, Dict, Optional, Union, Any, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from pydantic import BaseModel

# Assuming your refactored PDFModel is available
from tableai.pdf.pdf_model import PDFModel

# Local inference - will be updated to return a Pydantic model
from inference import DetectronTableDetector, DetectronDetectionResult, PageDimensions

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

# ────────────────────────────────────────────────────────────────
# Utility Helpers
# ────────────────────────────────────────────────────────────────
def _get_detector(dataset: str, confidence: Optional[float] = None) -> DetectronTableDetector:
    """Returns a cached DetectronTableDetector instance."""
    # Create a unique key for the cache based on dataset and confidence
    cache_key = f"{dataset}_{confidence or 0.5}"
    if cache_key not in _detector_cache:
        _detector_cache[cache_key] = DetectronTableDetector(dataset=dataset, confidence_threshold=confidence)
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


# ─────────────────── NEW PNG ENDPOINT ───────────────────────
@app.post(
    "/detect/png",
    response_model=Union[DetectronDetectionResult, ErrorResponse],
    tags=["Detection"]
)
async def detect_png(
    image_file: UploadFile = File(..., description="PNG image of a single document page."),
    pdf_file: UploadFile = File(..., description="The original PDF file for coordinate mapping."),
    page_num: int = Form(..., description="The 0-indexed page number the image corresponds to."),
    zoom: float = Form(2.0, description="The zoom factor used to generate the image."),
    dataset: str = Form("PubLayNet", description="Which Detectron2 model to use."),
    selector: Optional[str] = Form(None, description="The block type to treat as the primary detection target (e.g., 'Table')."),
    confidence_threshold: Optional[float] = Form(None, description="Override the model's default confidence threshold."),
):
    """
    Detects layout regions in a single page image (PNG) using Detectron2.
    """
    try:
        if pdf_file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Uploaded 'pdf_file' must be a PDF.")

        pdf_bytes = await pdf_file.read()
        # Use the from_bytes factory method for flexibility
        pdf_model = PDFModel.from_bytes(pdf_bytes, name=pdf_file.filename)

        if not (0 <= page_num < pdf_model.source_doc.page_count):
            raise HTTPException(status_code=400, detail=f"Invalid page_num {page_num}. PDF has {pdf_model.source_doc.page_count} pages.")

        image_bytes = await image_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if np_img is None:
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image file.")

        detector = _get_detector(dataset=dataset, confidence=confidence_threshold)

        # Use the new run_on_image method
        result = detector.run_on_image(
            np_img=np_img,
            pdf_model=pdf_model,
            page_num=page_num,
            zoom=zoom,
            selector=selector
        )
        return result

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions to let FastAPI handle them
        raise http_exc
    except Exception as exc:
        return ErrorResponse(
            message=f"An unexpected error occurred: {str(exc)}",
            model_used=dataset
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)