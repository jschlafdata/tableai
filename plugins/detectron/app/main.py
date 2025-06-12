from __future__ import annotations

import io
from typing import List, Dict, Optional, Union

from fastapi import FastAPI, UploadFile, HTTPException, Depends, Form, File
from pydantic import BaseModel, Field
from PIL import Image

# Shared utilities
from tableai.readers.files import FileReader
from tableai.pdf.models import PDFModel

# Local inference
from inference import DetectronTableDetector
import json 

app = FastAPI(
    title="TableAI Detectron2 Detection Service",
    description="Detectron2-based table (and region) detection via PDFModel",
    version="0.1.0"
)


class DetectionResponse(BaseModel):
    success: bool
    # one entry per detected table/region (flattened)
    detections: List[Dict[str, Union[int, float, tuple]]] = []
    # per-page other labels (metadata) in combined-PDF coords
    metadata: Optional[Dict[int, Dict[str, List[tuple]]]] = None
    page_dimensions: Optional[Dict[int, Dict[str, int]]]   = None
    page_count: Optional[int]     = None
    coordinate_system: str = "pdf_points"
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    supported_datasets: List[str]


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="tableai-detectron-service",
        version="0.0.1",
        model_loaded=False,
        supported_datasets=list(DetectronTableDetector._DATASET_CONFIGS.keys())
    )


@app.post("/detect/pdf", response_model=DetectionResponse)
async def detect_pdf(
    pdf_file: Optional[UploadFile] = File(None),
    pdf_path: Optional[str] = Form(None),
    zoom: Optional[str] = Form(None),
    coordinate_system: Optional[str] = Form(None),
    model_overrides: Optional[str] = Form(None),
    selector: Optional[str] = Form(None),
    dataset: Optional[str] = Form(None)
):
    """
    Detect tables (or specified regions) in a PDF. Supports:
      • Multipart upload: pdf_file (must be application/pdf)
      • S3: pdf_path="s3://bucket/key.pdf"
      • HTTP: pdf_path="https://example.com/file.pdf"
      • Local: pdf_path="/path/to/file.pdf"
    """
    try:
        # Handle defaults and type conversion explicitly
        zoom_val = float(zoom) if zoom else 2.0
        coord_system = coordinate_system if coordinate_system else "pdf"
        dataset_val = dataset if dataset else "PubLayNet"  # Set explicit default
        
        # Debug: Add received parameters to response for debugging
        debug_info = {
            "pdf_file_received": pdf_file is not None,
            "pdf_path": pdf_path,
            "zoom_raw": zoom,
            "zoom_parsed": zoom_val,
            "coordinate_system_raw": coordinate_system,
            "coordinate_system_parsed": coord_system,
            "model_overrides": model_overrides,
            "selector": selector,
            "dataset_raw": dataset,
            "dataset_parsed": dataset_val
        }
        
        # 1) Determine PDF source (bytes vs path)
        if pdf_file is not None:
            if pdf_file.content_type != 'application/pdf':
                raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
            pdf_bytes = await pdf_file.read()
            pdf_source = pdf_bytes
        elif pdf_path:
            pdf_source = pdf_path.strip()
        else:
            raise HTTPException(status_code=400, detail="Provide either 'pdf_file' or 'pdf_path'")

        # 3) Load PDFModel (handles S3, HTTP, local, bytes)
        pdf_model = PDFModel(path=pdf_source)

        # Parse model overrides
        model_overrides_dict = None
        if model_overrides:
            try:
                model_overrides_dict = json.loads(model_overrides)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in model_overrides")
        
        confidence_threshold = model_overrides_dict.get("confidence_threshold") if model_overrides_dict else None
        
        # Global detector instance (default: PubLayNet, threshold 0.5)
        detector = DetectronTableDetector(dataset=dataset_val, confidence_threshold=confidence_threshold)
        
        result = detector.run(
            pdf_model,
            zoom=zoom_val,
            confidence_threshold=confidence_threshold,
            selector=selector
        )

        # 5) Build flat list of table detections
        detections: List[Dict[str, Union[int, float, tuple]]] = []
        for page_num, bbox_list in result["tbl_coordinates"].items():
            for bbox in bbox_list:
                detections.append({
                    "page": page_num,
                    "bbox": bbox,
                    "confidence": 1.0
                })

        return DetectionResponse(
            success=True,
            detections=detections,
            metadata=result.get("metadata"),
            page_dimensions=result.get("page_dimensions"),
            page_count=pdf_model.pages,
            coordinate_system=coord_system,
            message=f"Detected {len(detections)} regions across {pdf_model.pages} pages. Debug: {debug_info}"
        )

    except Exception as e:
        return DetectionResponse(
            success=False,
            detections=[],
            metadata=None,
            page_dimensions=None,
            page_count=None,
            coordinate_system=coord_system,
            message=f"Error: {str(e)}. Debug: {debug_info if 'debug_info' in locals() else 'Debug info not available'}"
        )


@app.get("/models/info")
async def get_model_info():
    """
    Get info about the currently loaded Detectron2 model.
    """
    # Note: You'll need to handle this differently since detector is now local to the endpoint
    return {
        "supported_datasets": list(DetectronTableDetector._DATASET_CONFIGS.keys()),
        "coordinate_systems": ["image_pixels", "pdf_points"],
        "endpoints": {
            "/detect/pdf": "PDF detection (all sources via FileReader)",
            "/health": "Health check",
            "/models/info": "Model information"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)