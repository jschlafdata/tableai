from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from ultralytics import YOLO, YOLOv10
from tableai.pdf.pdf_model import PDFModel
# Added for type hinting PDFModel without circular import
# if "PDFModel" not in locals():
    # from typing import TYPE_CHECKING
    # if TYPE_CHECKING:
        # from tableai.pdf.pdf_model import PDFModel

os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "False"

# ────────────────────────────────────────────────────────────────
# 1. Pydantic Output Models for Detection Results
# ────────────────────────────────────────────────────────────────
BoundingBox = Tuple[float, float, float, float]
DetectionMetadata = Dict[str, List[BoundingBox]]

class PageDimensions(BaseModel):
    image_width: int
    image_height: int

class YOLODetectionResult(BaseModel):
    """Standardized output schema for YOLO detection results."""
    coordinates_by_page: Dict[int, List[BoundingBox]]
    table_bboxes: List[BoundingBox]
    all_model_bounds: Dict[int, Optional[DetectionMetadata]]
    page_dimensions: Dict[int, PageDimensions]

# ────────────────────────────────────────────────────────────────
# 2. Module-level Model Configuration
# ────────────────────────────────────────────────────────────────
_AVAILABLE_MODELS = {
    "keremberke": {
        "type": "huggingface", "repo_id": "keremberke/yolov8m-table-extraction",
        "filename": "best.pt", "description": "YOLOv8‑m: table only",
    },
    "foduucom": {
        "type": "huggingface", "repo_id": "foduucom/table-detection-and-extraction",
        "filename": "best.pt", "description": "YOLOv8: table only",
    },
    "doclaynet": {
        "type": "github_release", "url": "https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt",
        "filename": "yolov10x_best.pt", "description": "YOLOv10‑x on DocLayNet (tables + 10 other classes)",
        "expected_sha256": None,
    },
}

_DOC_LAYNET_LABEL_MAP = {
    0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
    4: "Page-footer", 5: "Page-header", 6: "Picture", 7: "Section-header",
    8: "Table", 9: "Text", 10: "Title",
}

# ────────────────────────────────────────────────────────────────
# 3. YOLOTableDetector Class
# ────────────────────────────────────────────────────────────────
class YOLOTableDetector:
    """YOLO‐based detector with dynamic model reloading for parameter overrides."""

    def __init__(self, model_data_dir: str = "/data"):
        # Default thresholds are removed from here. They are now managed by the model's defaults.
        self.model_data_dir = Path(model_data_dir)
        self.model_data_dir.mkdir(parents=True, exist_ok=True)
        # The cache now stores models keyed by a unique signature of their parameters.
        self._model_cache: Dict[str, Union[YOLO, YOLOv10]] = {}

    # ... (_download_github_asset and _sha256 methods are unchanged) ...
    def _download_github_asset(self, cfg: Dict) -> str:
        # (Implementation is identical)
        url, filename = cfg["url"], cfg["filename"]
        path = self.model_data_dir / filename
        if path.exists(): return str(path)
        print(f"Downloading {filename} …")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192): f.write(chunk)
        print(f"✓ downloaded → {path}")
        if cfg.get("expected_sha256") and self._sha256(path) != cfg["expected_sha256"]:
            path.unlink(missing_ok=True)
            raise RuntimeError("SHA‑256 mismatch!")
        return str(path)
        
    @staticmethod
    def _sha256(p: Path) -> str:
        # (Implementation is identical)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
        return h.hexdigest()

    # --- THIS IS THE KEY CHANGE ---
    def _load_model(self, name: str, overrides: Optional[Dict] = None) -> Union[YOLO, YOLOv10]:
        """
        Loads a YOLO model, using a cache keyed by both model name and its override parameters.
        If a model with the same parameters is already loaded, it's returned.
        If parameters differ, a new model instance is created and cached.
        """
        # Create a unique signature for this model configuration
        overrides_str = json.dumps(overrides, sort_keys=True) if overrides else "{}"
        model_signature = f"{name}_{hashlib.sha256(overrides_str.encode()).hexdigest()[:8]}"

        if model_signature in self._model_cache:
            # print(f"✓ Returning cached model '{model_signature}'")
            return self._model_cache[model_signature]

        print(f"-> Loading new model instance: '{model_signature}' (name: {name}, overrides: {overrides_str})")

        # 1. Load the base model from file
        if name not in _AVAILABLE_MODELS:
            raise ValueError(f"Unknown model '{name}'")
        cfg = _AVAILABLE_MODELS[name]
        pt_path = (
            hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"], cache_dir=str(self.model_data_dir))
            if cfg["type"] == "huggingface"
            else self._download_github_asset(cfg)
        )
        
        # Patch torch.load if necessary (good practice to keep)
        orig_load = torch.load
        def _patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)
        torch.load = _patched
        try:
            model = YOLOv10(pt_path) if name == "doclaynet" else YOLO(pt_path)
        finally:
            torch.load = orig_load

        # 2. Apply the overrides
        if overrides:
            model.overrides.update(overrides)
            print(f"✓ Applied overrides to '{model_signature}': {model.overrides}")
        
        self._model_cache[model_signature] = model
        print(f"✓ Caching new model '{model_signature}'")
        return model

    # ... (_label_for_class and _detect_on_page methods are unchanged) ...
    @classmethod
    def _label_for_class(cls, class_id: int, model_name: str) -> str:
        if model_name == "doclaynet":
            return _DOC_LAYNET_LABEL_MAP.get(class_id, f"class_{class_id}")
        return "Table"

    def _detect_on_page(
        self, np_img: np.ndarray, pdf_model: PDFModel, page_num: int, zoom: float,
        model: Union[YOLO, YOLOv10], model_name: str,
    ) -> Tuple[List[BoundingBox], DetectionMetadata]:
        img_h, img_w, _ = np_img.shape
        dets = model.predict(np_img, verbose=False)[0].boxes.cpu().data.numpy()
        tables, others = [], {}
        for x1, y1, x2, y2, conf, cls_id in dets:
            label = self._label_for_class(int(cls_id), model_name)
            # Use page-relative coordinates for simplicity in a single-page service
            page_bbox = pdf_model.img_bbox_to_page_bbox((x1, y1, x2, y2), (img_w, img_h), page_num, zoom=zoom)
            if label == "Table":
                tables.append(page_bbox)
            else:
                others.setdefault(label, []).append(page_bbox)
        return tables, others

    # --- `run_on_image` is now simplified ---
    def run_on_image(
        self,
        np_img: np.ndarray,
        pdf_model: PDFModel,
        page_num: int,
        zoom: float = 2.0,
        model_name: str = "keremberke",
        model_overrides: Optional[Dict[str, Any]] = None,
    ) -> YOLODetectionResult:
        """Processes a single image, dynamically loading the model with specified overrides."""
        # Load the model with the specific overrides for this request
        model = self._load_model(model_name, overrides=model_overrides)

        img_h, img_w, _ = np_img.shape
        page_dims = {page_num: PageDimensions(image_width=img_w, image_height=img_h)}

        tables, others = self._detect_on_page(
            np_img, pdf_model, page_num, zoom, model, model_name
        )

        return YOLODetectionResult(
            # Using more generic naming for consistency
            coordinates_by_page={page_num: tables},
            table_bboxes=tables, # For this single page, these are the same
            all_model_bounds={page_num: others if others else None},
            page_dimensions=page_dims,
        )

    # The 'run' method for full PDFs would now also use this new _load_model logic
    def run(self, *args, **kwargs):
        # ... (implementation would be similar, calling _load_model with overrides) ...
        pass

    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        return {n: cfg["description"] for n, cfg in _AVAILABLE_MODELS.items()}