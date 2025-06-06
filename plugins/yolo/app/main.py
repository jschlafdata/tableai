from __future__ import annotations

import io
from typing import List, Dict, Optional, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import json
# Import from tableai.tools (shared utilities)
from tableai_tools.readers.files import FileReader
from tableai_tools.pdf.models import PDFModel
from typing import Optional, Dict, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

# Import local inference
from .inference import YOLOTableDetector

app = FastAPI(
    title="TableAI YOLO Detection Service",
    description="Standalone YOLO-based table detection service",
    version="0.1.0"
)

# Global detector instance
detector = YOLOTableDetector()


class DetectionRequest(BaseModel):
    pdf_file: Optional[UploadFile] = None
    pdf_path: Optional[str] = None
    page_number: Optional[int] = None
    zoom: float = Field(default=2.0, gt=0)
    coordinate_system: str = "pdf"
    model_overrides: Optional[Dict[str, Union[float, int, bool]]] = None

class DetectionResponse(BaseModel):
    success: bool
    detections: List[Dict[str, Union[int, float, tuple]]] = []
    page_number: Optional[int] = None
    coordinate_system: str = "pdf_points"
    message: str = ""

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="tableai-yolo-service", 
        version="0.1.0",
        model_loaded=detector.model is not None
    )


@app.post("/detect/pdf", response_model=DetectionResponse)
async def detect_pdf(
    request: DetectionRequest = Depends()
):
    """
    Detect tables in a PDF - handles ALL sources via PDFModel/FileReader
    
    Supports:
    - Upload: pdf_file (multipart upload)
    - S3: pdf_path="s3://bucket/key.pdf" 
    - HTTP: pdf_path="https://example.com/file.pdf"
    - Local: pdf_path="/path/to/file.pdf"
    """
    try:
        # Determine PDF source - prioritize file upload, then path
        if request.pdf_file is not None:
            # Handle uploaded file
            if not request.pdf_file.content_type == 'application/pdf':
                raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
            pdf_bytes = await request.pdf_file.read()
            pdf_source = pdf_bytes
        elif request.pdf_path:
            # Handle path (S3, HTTP, local) - FileReader determines source automatically
            pdf_source = request.pdf_path.strip()
        else:
            raise HTTPException(status_code=400, detail="Provide either 'pdf_file' or 'pdf_path'")
        
        # PDFModel + FileReader handles ALL sources transparently
        pdf_model = PDFModel(path=pdf_source)
        
        # Run detection
        detection_result = detector.run(
            pdf_model,
            zoom=request.zoom, 
            model_overrides=request.model_overrides or {}
        )
        
        # Transform detections to list format
        detections = []
        for page, page_detections in detection_result['tbl_coordinates'].items():
            for bbox in page_detections:
                detections.append({
                    'page': page,
                    'bbox': bbox,
                    'confidence': 1.0  # Add a default confidence if not available
                })
        
        return DetectionResponse(
            success=True,
            detections=detections,
            page_number=request.page_number,
            coordinate_system=request.coordinate_system,
            message=f"Detected {len(detections)} tables on page {request.page_number}"
        )
    
    except Exception as e:
        return DetectionResponse(
            success=False,
            detections=[],
            page_number=request.page_number,
            message=str(e)
        )


@app.get("/models/info")
async def get_model_info():
    """Get information about the loaded YOLO model"""
    return {
        "model_type": "YOLOv8",
        "repository": detector.model_type,
        "task": "table_detection",
        "confidence_threshold": detector.confidence_threshold,
        "iou_threshold": detector.iou_threshold,
        "max_detections": detector.max_detections,
        "model_loaded": detector.model is not None,
        "tableai_tools_integration": True,
        "supported_pdf_sources": [
            "multipart upload (pdf_file)",
            "S3 paths (pdf_path='s3://bucket/key')",
            "HTTP URLs (pdf_path='https://...')",
            "local paths (pdf_path='/path/to/file.pdf')",
            "bytes/streams (automatic detection)"
        ],
        "coordinate_systems": ["image_pixels", "pdf_points"],
        "endpoints": {
            "/detect/image": "Single image detection",
            "/detect/pdf": "PDF detection (all sources via FileReader)",
            "/detect/batch": "Batch image processing",
            "/health": "Health check",
            "/models/info": "Model information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)