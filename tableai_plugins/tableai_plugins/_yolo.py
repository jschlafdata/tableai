from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from ultralytics import YOLO, YOLOv10
import os, tempfile

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
    coordinates_by_page: Dict[int, List[BoundingBox]]
    table_bboxes: List[DetectionMetadata]
    all_model_bounds: Dict[int, Optional[DetectionMetadata]]
    page_dimensions: Dict[int, PageDimensions]

    @property
    def data(self) -> Dict[str, Any]:
        return self.model_dump()


def _resolve_cache_dir(passed: Optional[Union[str, Path]] = None) -> Path:
    """
    Choose a writable cache directory in this order:
      1) explicit arg
      2) env: TABLEAI_CACHE_DIR, MODEL_CACHE_DIR, HF_HOME, HF_HUB_CACHE, TORCH_HOME, XDG_CACHE_HOME
      3) ~/.cache/tableai   (cross-platform safe enough)
      4) a unique temp dir
    """
    def _ok(p: Path) -> bool:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    candidates = []
    if passed:
        candidates.append(Path(passed))

    envs = (
        "TABLEAI_CACHE_DIR",
        "MODEL_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
    )
    for e in envs:
        v = os.getenv(e)
        if not v:
            continue
        p = Path(v)
        # nest under a subfolder when it’s a general cache root
        if e in {"HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"}:
            p = p / "tableai"
        candidates.append(p)

    # user cache
    candidates.append(Path.home() / ".cache" / "tableai")

    for c in candidates:
        if _ok(c):
            return c

    # final fallback: unique temp
    return Path(tempfile.mkdtemp(prefix="tableai_cache_"))

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

    def __init__(self, model_data_dir: Optional[Union[str, Path]] = None):
        # Default thresholds are removed from here. They are now managed by the model's defaults.
        self.model_data_dir = _resolve_cache_dir(model_data_dir)
        self._model_cache: Dict[str, Union[YOLO, YOLOv10]] = {}

    def _download_github_asset(self, cfg: Dict) -> str:
        url, filename = cfg["url"], cfg["filename"]
        path = self.model_data_dir / filename
        if path.exists():
            return str(path)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        if cfg.get("expected_sha256") and self._sha256(path) != cfg["expected_sha256"]:
            path.unlink(missing_ok=True)
            raise RuntimeError("SHA-256 mismatch!")
        return str(path)

    @staticmethod
    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_model(self, name: str, overrides: Optional[Dict] = None) -> Union[YOLO, YOLOv10]:
        overrides_str = json.dumps(overrides, sort_keys=True) if overrides else "{}"
        model_signature = f"{name}_{hashlib.sha256(overrides_str.encode()).hexdigest()[:8]}"
        if model_signature in self._model_cache:
            return self._model_cache[model_signature]

        if name not in _AVAILABLE_MODELS:
            raise ValueError(f"Unknown model '{name}'")
        cfg = _AVAILABLE_MODELS[name]
        pt_path = (
            hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"], cache_dir=str(self.model_data_dir))
            if cfg["type"] == "huggingface"
            else self._download_github_asset(cfg)
        )

        orig_load = torch.load
        def _patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)
        torch.load = _patched
        try:
            model = YOLOv10(pt_path) if name == "doclaynet" else YOLO(pt_path)
        finally:
            torch.load = orig_load

        if overrides:
            model.overrides.update(overrides)
        self._model_cache[model_signature] = model
        return model

    @classmethod
    def _label_for_class(cls, class_id: int, model_name: str) -> str:
        return _DOC_LAYNET_LABEL_MAP.get(class_id, f"class_{class_id}") if model_name == "doclaynet" else "Table"

    def _detect_on_page(
        self,
        np_img: np.ndarray,
        page_num: int,
        zoom: float,
        model: Union[YOLO, YOLOv10],
        model_name: str,
    ) -> Tuple[List[BoundingBox], DetectionMetadata]:
        """Return (tables, others) as tuples, matching the Pydantic schema."""
        dets = model.predict(np_img, verbose=False)[0].boxes.cpu().data.numpy()
        tables: List[BoundingBox] = []
        others: DetectionMetadata = {}
        for x1, y1, x2, y2, conf, cls_id in dets:
            bbox: BoundingBox = (float(x1), float(y1), float(x2), float(y2))
            label = self._label_for_class(int(cls_id), model_name)
            if label == "Table":
                tables.append(bbox)
            else:
                others.setdefault(label, []).append(bbox)
        return tables, others

    def run_on_image(
        self,
        np_img: np.ndarray,
        page_num: int,
        zoom: float = 2.0,
        model_name: str = "keremberke",
        model_overrides: Optional[Dict[str, Any]] = None,
    ) -> YOLODetectionResult:
        model = self._load_model(model_name, overrides=model_overrides)
        img_h, img_w, _ = np_img.shape
        tables, others = self._detect_on_page(np_img, page_num, zoom, model, model_name)
        return YOLODetectionResult(
            coordinates_by_page={page_num: tables},
            table_bboxes=[{"Table": tables}],  # list-per-page; adjust if you batch pages
            all_model_bounds={page_num: others or None},
            page_dimensions={page_num: PageDimensions(image_width=img_w, image_height=img_h)},
        )

    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        return {n: cfg["description"] for n, cfg in _AVAILABLE_MODELS.items()}