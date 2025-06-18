from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, Form, File, Depends
from pydantic import BaseModel, Field
from tableai.pdf.pdf_model import PDFModel
# Local inference - models have been moved inside this module
from .inference import YOLOTableDetector, YOLODetectionResult, PageDimensions
import json
# ────────────────────────────────────────────────────────────────
# FastAPI setup
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TableAI YOLO Detection Service",
    description="Standalone YOLO-based table detection service",
    version="0.3.2"  # Version bump for the fix
)

_detector_cache: dict[str, YOLOTableDetector] = {}

# ────────────────────────────────────────────────────────────────
# Pydantic Models for API Layer
# ────────────────────────────────────────────────────────────────

#
# >>> PngDetectionParams class is REMOVED. It was the source of the problem. <<<
#

class DetectionResponse(BaseModel):
    # This model is now for error responses
    success: bool
    detections: List = []
    page_dimensions: Optional[Dict[int, PageDimensions]] = None
    page_count: Optional[int] = None
    coordinate_system: str = "pdf_points"
    model_used: str = ""
    message: str = ""

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    cached_models: List[str]

def _get_detector(model_type: str) -> YOLOTableDetector:
    available_models = YOLOTableDetector.list_available_models()
    if model_type not in available_models:
        raise HTTPException(status_code=400, detail=f"Invalid model_type '{model_type}'. Valid options: {list(available_models.keys())}")
    if model_type not in _detector_cache:
        _detector_cache[model_type] = YOLOTableDetector()
    return _detector_cache[model_type]

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", service="tableai-yolo-service", version="0.3.2", cached_models=list(_detector_cache.keys()))

@app.post(
    "/detect/png",
    response_model=Union[YOLODetectionResult, DetectionResponse], # Assuming DetectionResponse is your error model
    tags=["Detection"]
)
async def detect_png(
    image_file: UploadFile = File(..., description="PNG image of a single document page."),
    pdf_file: UploadFile = File(..., description="The original PDF file for coordinate mapping."),
    page_num: int = Form(..., description="The 0-indexed page number the image corresponds to."),
    dataset: str = Form("keremberke", description="Which YOLO model to load."),
    zoom: float = Form(2.0, description="The zoom factor used to generate the image."),
    model_overrides: Optional[str] = Form(None, description="JSON string of Ultralytics args (e.g., '{\"conf\": 0.35, \"iou\": 0.5}')."),
):
    """
    Detect tables in a single page image (PNG). Dynamically configures the
    YOLO model based on the `model_overrides` parameter.
    """
    overrides_dict = None
    try:
        # Validate JSON early
        if model_overrides:
            overrides_dict = json.loads(model_overrides)

        if pdf_file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Uploaded 'pdf_file' must be a PDF.")

        pdf_bytes = await pdf_file.read()
        # Use the from_bytes factory method for the PDFModel
        pdf_model = PDFModel.from_bytes(pdf_bytes, name=pdf_file.filename)
        image_bytes = await image_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if np_img is None:
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image file.")

        # Get a general detector instance. The specific model+params will be handled inside.
        detector = _get_detector(dataset) # _get_detector now just ensures the class instance exists

        result = detector.run_on_image(
            np_img=np_img,
            pdf_model=pdf_model,
            page_num=page_num,
            zoom=zoom,
            model_name=dataset,
            model_overrides=overrides_dict # Pass the validated dictionary
        )
        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in model_overrides.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        # Ensure DetectionResponse is your defined error model
        return DetectionResponse(
            success=False,
            message=f"An unexpected error occurred: {str(exc)}",
            model_used=dataset
        )

# Simple _get_detector, as heavy lifting is now inside the class
def _get_detector(model_type: str) -> YOLOTableDetector:
    # This function now just needs to manage one instance of the *class*, not per-param models.
    if "yolo_detector" not in _detector_cache:
        _detector_cache["yolo_detector"] = YOLOTableDetector()
    return _detector_cache["yolo_detector"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import List, Dict, Optional, Union, Any

# import cv2  # <-- ADD
# import numpy as np # <-- ADD

# from fastapi import FastAPI, UploadFile, HTTPException, Form, File
# from pydantic import BaseModel
# # from tableai.readers.files import FileReader # No longer used
# from tableai.pdf.pdf_model import PDFModel

# # Local inference
# from .inference import YOLOTableDetector

# # ────────────────────────────────────────────────────────────────
# # FastAPI setup
# # ────────────────────────────────────────────────────────────────
# app = FastAPI(
#     title="TableAI YOLO Detection Service",
#     description="Standalone YOLO-based table detection service (param-compatible with Detectron endpoint)",
#     version="0.2.1" #<-- Version bump
# )

# # Single detector cache (keeps weights on disk / in memory)
# _detector_cache: dict[str, YOLOTableDetector] = {}   # keyed by model_type

# # ────────────────────────────────────────────────────────────────
# # Pydantic response models (UNCHANGED)
# # ────────────────────────────────────────────────────────────────
# class DetectionResponse(BaseModel):
#     success: bool
#     detections: Any = []
#     page_dimensions: Optional[Dict[int, Dict[str, int]]] = None
#     page_count: Optional[int] = None
#     coordinate_system: str = "pdf_points"
#     model_used: str = ""
#     message: str = ""


# class HealthResponse(BaseModel):
#     status: str
#     service: str
#     version: str
#     cached_models: List[str]


# # ────────────────────────────────────────────────────────────────
# # Utility helpers (UNCHANGED)
# # ────────────────────────────────────────────────────────────────
# def _get_detector(model_type: str) -> YOLOTableDetector:
#     """
#     Return a (cached) YOLOTableDetector for the requested `model_type`.
#     """
#     if model_type not in YOLOTableDetector.AVAILABLE_MODELS:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 f"Invalid model_type '{model_type}'. "
#                 f"Valid options: {list(YOLOTableDetector.AVAILABLE_MODELS.keys())}"
#             )
#         )
#     if model_type not in _detector_cache:
#         _detector_cache[model_type] = YOLOTableDetector()
#     return _detector_cache[model_type]


# # ────────────────────────────────────────────────────────────────
# # Routes
# # ────────────────────────────────────────────────────────────────
# @app.get("/health", response_model=HealthResponse)
# def health() -> HealthResponse:
#     return HealthResponse(
#         status="healthy",
#         service="tableai-yolo-service",
#         version="0.2.1", # <-- Version bump
#         cached_models=list(_detector_cache.keys()),
#     )

# # ────────────────── NEW ENDPOINT FOR PNG ─────────────────────────
# @app.post("/detect/png", response_model=DetectionResponse)
# async def detect_png(
#     image_file: UploadFile = File(..., description="PNG image of a single document page."),
#     pdf_file: Optional[UploadFile] = File(None),
#     page_num: int = Form(..., description="The 0-indexed page number the image corresponds to."),
#     zoom: str = Form("2.0", description="The zoom factor used to generate the image."),
#     coordinate_system: str = Form("pdf_points"),
#     model_overrides: Optional[str] = Form(None),
#     selector: Optional[str] = Form(None),
#     dataset: str = Form("keremberke", description="Which YOLO model to load."),
# ):
#     """
#     Detect tables in a single page image (PNG). Requires the original PDF path
#     and page number for accurate coordinate translation.
#     """
#     debug_info = {}
#     try:
#         zoom_val = float(zoom)
#         model_type = dataset
#         debug_info.update(
#             zoom_raw=zoom, zoom_parsed=zoom_val,
#             coordinate_system_raw=coordinate_system, coordinate_system_parsed=coordinate_system,
#             dataset_raw=dataset, model_type=model_type,
#             pdf_file=pdf_file, page_num=page_num
#         )

#         if pdf_file is not None:
#             if pdf_file.content_type != "application/pdf":
#                 raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
#             pdf_bytes = await pdf_file.read()
#             pdf_source = pdf_bytes
#         else:
#             raise HTTPException(status_code=400, detail="Provide either 'pdf_file' or 'pdf_path'")

#         overrides = json.loads(model_overrides) if model_overrides else None
#         pdf_model = PDFModel(path=pdf_source)
#         # 2. Load image into numpy array
#         image_bytes = await image_file.read()
#         np_arr = np.frombuffer(image_bytes, np.uint8)
#         np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         if not (0 <= page_num < pdf_model.pages):
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid page_num {page_num}. PDF has {pdf_model.pages} pages (0 to {pdf_model.pages - 1})."
#             )

#         # 4. Get detector and overrides
#         detector = _get_detector(model_type)
#         overrides = json.loads(model_overrides) if model_overrides else None

#         # 5. Run detection on the single image
#         result = detector.run_on_image(
#             np_img=np_img,
#             pdf_model=pdf_model,
#             page_num=page_num,
#             zoom=zoom_val,
#             model_overrides=overrides,
#             model_name=model_type
#         )
        
#         # 6. Format response (same as before)
#         detections: List[Dict[str, Union[int, float, tuple]]] = []
#         for p_num, bbox_list in result["tbl_coordinates"].items():
#             for bbox in bbox_list:
#                 detections.append({"page": p_num, "bbox": bbox, "confidence": 1.0})
        
#         return DetectionResponse(
#             success=True,
#             detections=result,
#             page_dimensions=result.get("page_dimensions"),
#             page_count=pdf_model.pages,
#             coordinate_system=coordinate_system,
#             model_used=model_type,
#             message=f"Detected {len(detections)} tables on page {page_num}. "
#                     f"Debug: {debug_info}"
#         )

#     except Exception as exc:
#         return DetectionResponse(
#             success=False, message=f"Error: {str(exc)}. Debug: {debug_info}"
#         )


# # ────────────────── ORIGINAL PDF ENDPOINT (UNCHANGED) ────────────
# @app.post("/detect/pdf", response_model=DetectionResponse)
# async def detect_pdf(
#     pdf_file: Optional[UploadFile] = File(None),
#     pdf_path: Optional[str] = Form(None),
#     zoom: Optional[str] = Form(None),
#     coordinate_system: Optional[str] = Form(None),
#     model_overrides: Optional[str] = Form(None),
#     selector: Optional[str] = Form(None),
#     dataset: Optional[str] = Form(None),
# ):
#     """
#     Detect tables in a PDF. *Interface identical to Detectron* so front-end /
#     client code can call either service without changes.
#     """

#     debug_info = {}
#     try:
#         zoom_val = float(zoom) if zoom else 2.0
#         coord_system = coordinate_system or "pdf"
#         model_type = dataset or "keremberke"
#         debug_info.update(
#             zoom_raw=zoom, zoom_parsed=zoom_val,
#             coordinate_system_raw=coordinate_system, coordinate_system_parsed=coord_system,
#             dataset_raw=dataset, model_type=model_type,
#         )
#         if pdf_file is not None:
#             if pdf_file.content_type != "application/pdf":
#                 raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
#             pdf_bytes = await pdf_file.read()
#             pdf_source = pdf_bytes
#         elif pdf_path:
#             pdf_source = pdf_path.strip()
#         else:
#             raise HTTPException(status_code=400, detail="Provide either 'pdf_file' or 'pdf_path'")

#         overrides = json.loads(model_overrides) if model_overrides else None

#         pdf_model = PDFModel(path=pdf_source)
#         detector = _get_detector(model_type)
#         # This now calls the original .run() method
#         result = detector.run(
#             pdf_model,
#             zoom=zoom_val,
#             model_overrides=overrides,
#             model_name=model_type
#         )

#         detections: List[Dict[str, Union[int, float, tuple]]] = []
#         for page_num, bbox_list in result["tbl_coordinates"].items():
#             for bbox in bbox_list:
#                 detections.append({"page": page_num, "bbox": bbox, "confidence": 1.0})

#         return DetectionResponse(
#             success=True,
#             detections=result,
#             page_dimensions=result.get("page_dimensions"),
#             page_count=pdf_model.pages,
#             coordinate_system=coord_system,
#             model_used=model_type,
#             message=f"Detected {len(detections)} tables across {pdf_model.pages} pages. "
#                     f"Debug: {debug_info}"
#         )

#     except Exception as exc:
#         return DetectionResponse(
#             success=False, message=f"Error: {str(exc)}. Debug: {debug_info}"
#         )


# @app.get("/models/info")
# def models_info():
#     """Mirror Detectron `/models/info` shape."""
#     return {
#         "available_models": list(YOLOTableDetector.AVAILABLE_MODELS.keys()),
#         "cached_models": list(_detector_cache.keys()),
#         "coordinate_systems": ["image_pixels", "pdf_points"],
#         "endpoints": {
#             "/detect/pdf": "PDF detection (processes full PDF).",
#             "/detect/png": "Single page PNG detection (requires original PDF path).", # <-- ADDED
#             "/health": "Health check",
#             "/models/info": "Model information",
#         },
#     }


# ────────────────────────────────────────────────────────────────
# Local dev entry point (UNCHANGED)
# ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# from __future__ import annotations

# import json
# from typing import List, Dict, Optional, Union, Any

# from fastapi import FastAPI, UploadFile, HTTPException, Form, File
# from pydantic import BaseModel
# from tableai.readers.files import FileReader
# from tableai.pdf.models import PDFModel

# # Local inference
# from .inference import YOLOTableDetector

# # ────────────────────────────────────────────────────────────────
# # FastAPI setup
# # ────────────────────────────────────────────────────────────────
# app = FastAPI(
#     title="TableAI YOLO Detection Service",
#     description="Standalone YOLO-based table detection service (param-compatible with Detectron endpoint)",
#     version="0.2.0"
# )

# # Single detector cache (keeps weights on disk / in memory)
# _detector_cache: dict[str, YOLOTableDetector] = {}   # keyed by model_type


# # ────────────────────────────────────────────────────────────────
# # Pydantic response models
# # ────────────────────────────────────────────────────────────────
# class DetectionResponse(BaseModel):
#     success: bool
#     detections: Any = []
#     page_dimensions: Optional[Dict[int, Dict[str, int]]] = None
#     page_count: Optional[int] = None
#     coordinate_system: str = "pdf_points"
#     model_used: str = ""
#     message: str = ""


# class HealthResponse(BaseModel):
#     status: str
#     service: str
#     version: str
#     cached_models: List[str]


# # ────────────────────────────────────────────────────────────────
# # Utility helpers
# # ────────────────────────────────────────────────────────────────
# def _get_detector(model_type: str) -> YOLOTableDetector:
#     """
#     Return a (cached) YOLOTableDetector for the requested `model_type`.
#     """
#     if model_type not in YOLOTableDetector.AVAILABLE_MODELS:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 f"Invalid model_type '{model_type}'. "
#                 f"Valid options: {list(YOLOTableDetector.AVAILABLE_MODELS.keys())}"
#             )
#         )

#     if model_type not in _detector_cache:
#         _detector_cache[model_type] = YOLOTableDetector()
#     return _detector_cache[model_type]


# # ────────────────────────────────────────────────────────────────
# # Routes
# # ────────────────────────────────────────────────────────────────
# @app.get("/health", response_model=HealthResponse)
# def health() -> HealthResponse:
#     return HealthResponse(
#         status="healthy",
#         service="tableai-yolo-service",
#         version="0.2.0",
#         cached_models=list(_detector_cache.keys()),
#     )


# @app.post("/detect/pdf", response_model=DetectionResponse)
# async def detect_pdf(
#     # ── parameters copied 1-for-1 from Detectron endpoint ─────────
#     pdf_file: Optional[UploadFile] = File(None),
#     pdf_path: Optional[str] = Form(None),
#     zoom: Optional[str] = Form(None),
#     coordinate_system: Optional[str] = Form(None),
#     model_overrides: Optional[str] = Form(None),
#     selector: Optional[str] = Form(None),
#     dataset: Optional[str] = Form(None),          # keep same name but maps to YOLO model_type
# ):
#     """
#     Detect tables in a PDF. *Interface identical to Detectron* so front-end /
#     client code can call either service without changes.

#     • `dataset`   → which YOLO model to load (keremberke, foduucom, doclaynet).
#     • `selector`  is accepted for parity but **ignored** by YOLO (kept for future use).
#     """

#     debug_info = {}  # will be appended to `message` for easier troubleshooting
#     try:
#         zoom_val = float(zoom) if zoom else 2.0
#         coord_system = coordinate_system or "pdf"
#         model_type = dataset or "keremberke"       # default YOLO model
#         debug_info.update(
#             zoom_raw=zoom, zoom_parsed=zoom_val,
#             coordinate_system_raw=coordinate_system, coordinate_system_parsed=coord_system,
#             dataset_raw=dataset, model_type=model_type,
#         )
#         if pdf_file is not None:
#             if pdf_file.content_type != "application/pdf":
#                 raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
#             pdf_bytes = await pdf_file.read()
#             pdf_source = pdf_bytes
#         elif pdf_path:
#             pdf_source = pdf_path.strip()
#         else:
#             raise HTTPException(status_code=400, detail="Provide either 'pdf_file' or 'pdf_path'")

#         overrides = None
#         if model_overrides:
#             try:
#                 overrides = json.loads(model_overrides)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid JSON in model_overrides")

#         pdf_model = PDFModel(path=pdf_source)
#         detector = _get_detector(model_type)
#         result = detector.run(
#             pdf_model,
#             zoom=zoom_val,
#             model_overrides=overrides,
#             model_name=model_type
#         )

#         detections: List[Dict[str, Union[int, float, tuple]]] = []
#         for page_num, bbox_list in result["tbl_coordinates"].items():
#             for bbox in bbox_list:
#                 detections.append(
#                     {"page": page_num, "bbox": bbox, "confidence": 1.0}
#                 )

#         return DetectionResponse(
#             success=True,
#             detections=result,
#             page_dimensions=result.get("page_dimensions"),
#             page_count=pdf_model.pages,
#             coordinate_system=coord_system,
#             model_used=model_type,
#             message=f"Detected {len(detections)} tables across {pdf_model.pages} pages. "
#                     f"Debug: {debug_info}"
#         )

#     except Exception as exc:
#         return DetectionResponse(
#             success=False,
#             detections=[],
#             page_dimensions=None,
#             page_count=None,
#             coordinate_system=coordinate_system or "pdf",
#             model_used=dataset or "keremberke",
#             message=f"Error: {str(exc)}. Debug: {debug_info}"
#         )


# @app.get("/models/info")
# def models_info():
#     """Mirror Detectron `/models/info` shape."""
#     return {
#         "available_models": list(YOLOTableDetector.AVAILABLE_MODELS.keys()),
#         "cached_models": list(_detector_cache.keys()),
#         "coordinate_systems": ["image_pixels", "pdf_points"],
#         "endpoints": {
#             "/detect/pdf": "PDF detection (Detectron-compatible interface)",
#             "/health": "Health check",
#             "/models/info": "Model information",
#         },
#     }


# # ────────────────────────────────────────────────────────────────
# # Local dev entry point
# # ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)