# tableai_plugins/yolo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np
from pydantic import BaseModel

from .loader import MissingOptionalDependency

BoundingBox = Tuple[float, float, float, float]

class PageDimensions(BaseModel):
    image_width: int
    image_height: int

class YOLODetectionResult(BaseModel):
    coordinates_by_page: Dict[int, List[BoundingBox]]
    table_bboxes: List[Dict[str, List[BoundingBox]]]
    all_model_bounds: Dict[int, Optional[Dict[str, List[BoundingBox]]]]
    page_dimensions: Dict[int, PageDimensions]

    @property
    def data(self) -> Dict[str, Any]:
        return self.model_dump()

# Available models and sources
_AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "keremberke": {
        "type": "huggingface",
        "repo_id": "keremberke/yolov8m-table-extraction",
        "filename": "best.pt",
        "description": "YOLOv8‑m: table only",
    },
    "foduucom": {
        "type": "huggingface",
        "repo_id": "foduucom/table-detection-and-extraction",
        "filename": "best.pt",
        "description": "YOLOv8: table only",
    },
    "doclaynet": {
        "type": "github_release",
        "url": "https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt",
        "filename": "yolov10x_best.pt",
        "description": "YOLOv10‑x on DocLayNet (tables + 10 other classes)",
        "expected_sha256": None,  # set to a hash string if you want strict verification
    },
}

_DOCLAYNET_LABEL_MAP = {
    0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
    4: "Page-footer", 5: "Page-header", 6: "Picture", 7: "Section-header",
    8: "Table", 9: "Text", 10: "Title",
}

def _resolve_cache_dir(passed: Optional[Union[str, Path]] = None) -> Path:
    """
    Choose a writable cache directory in priority order.
    """
    def _ok(p: Path) -> bool:
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".write_test"
            t.write_text("ok", encoding="utf-8"); t.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    candidates: List[Path] = []
    if passed:
        candidates.append(Path(passed))

    for e in ("TABLEAI_CACHE_DIR", "MODEL_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        v = os.getenv(e)
        if not v:
            continue
        p = Path(v)
        if e in {"HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"}:
            p = p / "tableai"
        candidates.append(p)

    candidates.append(Path.home() / ".cache" / "tableai")

    for c in candidates:
        if _ok(c):
            return c

    return Path(tempfile.mkdtemp(prefix="tableai_cache_"))

@dataclass
class YOLOPlugin:
    """
    Dynamic YOLO plugin.
    Usage:
        from tableai_plugins import get_plugin
        YOLO = get_plugin("yolo")
        det = YOLO(model_name="keremberke")
        result = det.detect(np_image, page_num=0)
    """
    name: str = "yolo"
    model_name: str = "keremberke"
    model_overrides: Optional[Dict[str, Any]] = None
    model_data_dir: Optional[Union[str, Path]] = None

    # Internal (initialized in __post_init__)
    _hf_hub_download: Any = None
    _torch: Any = None
    _YOLO: Any = None
    _YOLOv10: Any = None
    _model_cache: Dict[str, Any] = None

    def __post_init__(self):
        # Heavy deps are imported lazily here
        try:
            from ultralytics import YOLO as _YOLO  # type: ignore
            try:
                from ultralytics import YOLOv10 as _YOLOv10  # type: ignore
            except Exception:
                _YOLOv10 = None
        except Exception as e:
            raise MissingOptionalDependency(
                "Ultralytics is required for the YOLO plugin.\n"
                "Install with: pip install 'tableai-plugins[yolo]'"
            ) from e
        try:
            import torch as _torch  # type: ignore
        except Exception as e:
            raise MissingOptionalDependency(
                "PyTorch is required for the YOLO plugin.\n"
                "Install a compatible torch build, or include it in your extras."
            ) from e
        try:
            from huggingface_hub import hf_hub_download as _hf_hub_download  # type: ignore
        except Exception as e:
            raise MissingOptionalDependency(
                "huggingface-hub is required to fetch certain weights."
            ) from e

        self._YOLO = _YOLO
        self._YOLOv10 = _YOLOv10
        self._torch = _torch
        self._hf_hub_download = _hf_hub_download

        self.model_overrides = self.model_overrides or {}
        self.model_data_dir = _resolve_cache_dir(self.model_data_dir)
        self._model_cache = {}

    # ----------------------------- Public API -----------------------------

    def detect(
        self,
        image: Union[np.ndarray, "Image.Image"],
        page_num: int = 0,
        model_name: Optional[str] = None,
        model_overrides: Optional[Dict[str, Any]] = None,
    ) -> YOLODetectionResult:
        """
        Run detection on a single RGB/BGR numpy array or PIL Image.
        Returns a pydantic YOLODetectionResult.
        """
        np_img = self._to_numpy(image)
        name = model_name or self.model_name
        overrides = {**self.model_overrides, **(model_overrides or {})}
        model = self._load_model(name, overrides)

        # Ultralytics inference
        results = model.predict(np_img, verbose=False)
        res0 = results[0]
        # Best-effort: .boxes.data is the stable API
        dets = res0.boxes.cpu().data.numpy()  # [N, 6] -> x1,y1,x2,y2,conf,cls

        tables: List[BoundingBox] = []
        others: Dict[str, List[BoundingBox]] = {}

        for x1, y1, x2, y2, conf, cls_id in dets:
            bbox: BoundingBox = (float(x1), float(y1), float(x2), float(y2))
            label = self._class_label(int(cls_id), name)
            if label == "Table":
                tables.append(bbox)
            else:
                others.setdefault(label, []).append(bbox)

        h, w = np_img.shape[:2]
        return YOLODetectionResult(
            coordinates_by_page={page_num: tables},
            table_bboxes=[{"Table": tables}],
            all_model_bounds={page_num: (others or None)},
            page_dimensions={page_num: PageDimensions(image_width=w, image_height=h)},
        )

    @staticmethod
    def list_available_models() -> Dict[str, str]:
        return {n: cfg["description"] for n, cfg in _AVAILABLE_MODELS.items()}

    # ---------------------------- Internals -------------------------------

    @staticmethod
    def _to_numpy(x: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise MissingOptionalDependency(
                "Pillow is required when passing PIL Images."
            ) from e
        if isinstance(x, Image.Image):
            return np.asarray(x)
        raise TypeError(f"Unsupported image type: {type(x)!r}")

    @staticmethod
    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def _download_github_asset(self, cfg: Dict[str, Any]) -> str:
        url, filename = cfg["url"], cfg["filename"]
        path = self.model_data_dir / filename
        if path.exists():
            return str(path)
        import requests  # defer import
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        if cfg.get("expected_sha256"):
            if self._sha256(path) != cfg["expected_sha256"]:
                path.unlink(missing_ok=True)
                raise RuntimeError("SHA-256 mismatch for downloaded weight file.")
        return str(path)

    def _load_model(self, name: str, overrides: Optional[Dict[str, Any]] = None):
        if name not in _AVAILABLE_MODELS:
            raise ValueError(f"Unknown YOLO model '{name}'. "
                             f"Valid: {list(_AVAILABLE_MODELS)}")
        overrides_str = json.dumps(overrides or {}, sort_keys=True)
        sig = f"{name}_{hashlib.sha256(overrides_str.encode()).hexdigest()[:8]}"
        if sig in self._model_cache:
            return self._model_cache[sig]

        cfg = _AVAILABLE_MODELS[name]
        if cfg["type"] == "huggingface":
            pt_path = self._hf_hub_download(
                repo_id=cfg["repo_id"],
                filename=cfg["filename"],
                cache_dir=str(self.model_data_dir),
            )
        else:
            pt_path = self._download_github_asset(cfg)

        # Some ultralytics variants read via torch.load; make sure weights_only=False
        orig_load = self._torch.load
        def _patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)
        self._torch.load = _patched
        try:
            if name == "doclaynet" and self._YOLOv10 is not None:
                model = self._YOLOv10(pt_path)
            else:
                model = self._YOLO(pt_path)
        finally:
            self._torch.load = orig_load

        if overrides:
            model.overrides.update(overrides)

        self._model_cache[sig] = model
        return model

    @staticmethod
    def _class_label(cls_id: int, model_name: str) -> str:
        if model_name == "doclaynet":
            return _DOCLAYNET_LABEL_MAP.get(cls_id, f"class_{cls_id}")
        return "Table"
