from __future__ import annotations

import json
from typing import List, Dict, Optional, Union, Any

from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from pydantic import BaseModel
from tableai.readers.files import FileReader
from tableai.pdf.models import PDFModel

# Local inference
from .inference import YOLOTableDetector

# ────────────────────────────────────────────────────────────────
# FastAPI setup
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TableAI YOLO Detection Service",
    description="Standalone YOLO-based table detection service (param-compatible with Detectron endpoint)",
    version="0.2.0"
)

# Single detector cache (keeps weights on disk / in memory)
_detector_cache: dict[str, YOLOTableDetector] = {}   # keyed by model_type


# ────────────────────────────────────────────────────────────────
# Pydantic response models
# ────────────────────────────────────────────────────────────────
class DetectionResponse(BaseModel):
    success: bool
    detections: Any = []
    page_dimensions: Optional[Dict[int, Dict[str, int]]] = None
    page_count: Optional[int] = None
    coordinate_system: str = "pdf_points"
    model_used: str = ""
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    cached_models: List[str]


# ────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────
def _get_detector(model_type: str) -> YOLOTableDetector:
    """
    Return a (cached) YOLOTableDetector for the requested `model_type`.
    """
    if model_type not in YOLOTableDetector.AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid model_type '{model_type}'. "
                f"Valid options: {list(YOLOTableDetector.AVAILABLE_MODELS.keys())}"
            )
        )

    if model_type not in _detector_cache:
        _detector_cache[model_type] = YOLOTableDetector()
    return _detector_cache[model_type]


# ────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="tableai-yolo-service",
        version="0.2.0",
        cached_models=list(_detector_cache.keys()),
    )


@app.post("/detect/pdf", response_model=DetectionResponse)
async def detect_pdf(
    # ── parameters copied 1-for-1 from Detectron endpoint ─────────
    pdf_file: Optional[UploadFile] = File(None),
    pdf_path: Optional[str] = Form(None),
    zoom: Optional[str] = Form(None),
    coordinate_system: Optional[str] = Form(None),
    model_overrides: Optional[str] = Form(None),
    selector: Optional[str] = Form(None),
    dataset: Optional[str] = Form(None),          # keep same name but maps to YOLO model_type
):
    """
    Detect tables in a PDF. *Interface identical to Detectron* so front-end /
    client code can call either service without changes.

    • `dataset`   → which YOLO model to load (keremberke, foduucom, doclaynet).
    • `selector`  is accepted for parity but **ignored** by YOLO (kept for future use).
    """

    debug_info = {}  # will be appended to `message` for easier troubleshooting
    try:
        zoom_val = float(zoom) if zoom else 2.0
        coord_system = coordinate_system or "pdf"
        model_type = dataset or "keremberke"       # default YOLO model
        debug_info.update(
            zoom_raw=zoom, zoom_parsed=zoom_val,
            coordinate_system_raw=coordinate_system, coordinate_system_parsed=coord_system,
            dataset_raw=dataset, model_type=model_type,
        )
        if pdf_file is not None:
            if pdf_file.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
            pdf_bytes = await pdf_file.read()
            pdf_source = pdf_bytes
        elif pdf_path:
            pdf_source = pdf_path.strip()
        else:
            raise HTTPException(status_code=400, detail="Provide either 'pdf_file' or 'pdf_path'")

        overrides = None
        if model_overrides:
            try:
                overrides = json.loads(model_overrides)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in model_overrides")

        pdf_model = PDFModel(path=pdf_source)
        detector = _get_detector(model_type)
        result = detector.run(
            pdf_model,
            zoom=zoom_val,
            model_overrides=overrides,
            model_name=model_type
        )

        detections: List[Dict[str, Union[int, float, tuple]]] = []
        for page_num, bbox_list in result["tbl_coordinates"].items():
            for bbox in bbox_list:
                detections.append(
                    {"page": page_num, "bbox": bbox, "confidence": 1.0}
                )

        return DetectionResponse(
            success=True,
            detections=result,
            page_dimensions=result.get("page_dimensions"),
            page_count=pdf_model.pages,
            coordinate_system=coord_system,
            model_used=model_type,
            message=f"Detected {len(detections)} tables across {pdf_model.pages} pages. "
                    f"Debug: {debug_info}"
        )

    except Exception as exc:
        return DetectionResponse(
            success=False,
            detections=[],
            page_dimensions=None,
            page_count=None,
            coordinate_system=coordinate_system or "pdf",
            model_used=dataset or "keremberke",
            message=f"Error: {str(exc)}. Debug: {debug_info}"
        )


@app.get("/models/info")
def models_info():
    """Mirror Detectron `/models/info` shape."""
    return {
        "available_models": list(YOLOTableDetector.AVAILABLE_MODELS.keys()),
        "cached_models": list(_detector_cache.keys()),
        "coordinate_systems": ["image_pixels", "pdf_points"],
        "endpoints": {
            "/detect/pdf": "PDF detection (Detectron-compatible interface)",
            "/health": "Health check",
            "/models/info": "Model information",
        },
    }


# ────────────────────────────────────────────────────────────────
# Local dev entry point
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)