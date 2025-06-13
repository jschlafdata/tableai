from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import os, hashlib, requests, torch, numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO, YOLOv10

os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "False"


class YOLOTableDetector:
    """
    YOLO‐based table (and document‑layout) detector that returns the *same*
    schema as DetectronTableDetector:

        {
          "tbl_coordinates": {page: [(x1,y1,x2,y2), …]},
          "table_bboxes":    [(x1,y1,x2,y2), …],          # flattened
          "metadata":        {page: {label: [(x1,y1,x2,y2), …]} | None},
          "page_dimensions": {page: {"image_width": w, "image_height": h}}
        }
    """

    # ──────────────────────────────────────────────────────────────
    # 1.  MODEL CATALOGUE
    # ──────────────────────────────────────────────────────────────
    AVAILABLE_MODELS = {
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
            "expected_sha256": None,  # fill in if you have it
        },
    }

    # DocLayNet class‑id → label
    DOC_LAYNET_LABEL_MAP = {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "List-item",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",           # <‑‑ we treat this as “table”
        9: "Text",
        10: "Title",
    }

    # ──────────────────────────────────────────────────────────────
    # 2.  INITIALISER
    # ──────────────────────────────────────────────────────────────
    def __init__(
        self,
        model_data_dir: str = "/data",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 1_000,
    ):
        self.model_data_dir = Path(model_data_dir)
        self.model_data_dir.mkdir(parents=True, exist_ok=True)

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        self._model_cache: Dict[str, Union[YOLO, YOLOv10]] = {}

    # ──────────────────────────────────────────────────────────────
    # 3.  INTERNAL HELPERS (download / load)
    # ──────────────────────────────────────────────────────────────
    def _download_github_asset(self, cfg: Dict) -> str:
        url, filename = cfg["url"], cfg["filename"]
        path = self.model_data_dir / filename
        if path.exists():
            return str(path)

        print(f"Downloading {filename} …")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        print(f"✓ downloaded → {path}")

        if cfg.get("expected_sha256"):
            if self._sha256(path) != cfg["expected_sha256"]:
                path.unlink(missing_ok=True)
                raise RuntimeError("SHA‑256 mismatch!")
        return str(path)

    @staticmethod
    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_model(self, name: str) -> Union[YOLO, YOLOv10]:
        if name in self._model_cache:
            return self._model_cache[name]

        if name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model '{name}'")

        cfg = self.AVAILABLE_MODELS[name]

        # Resolve .pt path
        if cfg["type"] == "huggingface":
            pt_path = hf_hub_download(
                repo_id=cfg["repo_id"],
                filename=cfg["filename"],
                cache_dir=str(self.model_data_dir),
            )
        elif cfg["type"] == "github_release":
            pt_path = self._download_github_asset(cfg)
        else:
            raise ValueError(f"Unsupported model type '{cfg['type']}'")

        # Patch out Ultralytics’ strict load‑weights behaviour
        orig_load = torch.load

        def _patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)

        torch.load = _patched
        try:
            model = YOLOv10(pt_path) if name == "doclaynet" else YOLO(pt_path)
        finally:
            torch.load = orig_load

        # Runtime overrides
        model.overrides.update(
            dict(
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                agnostic_nms=False,
            )
        )

        self._model_cache[name] = model
        print(f"✓ loaded {name} model")
        return model

    # ──────────────────────────────────────────────────────────────
    # 4.  CLASS‑ID → LABEL (DocLayNet only)
    # ──────────────────────────────────────────────────────────────
    @classmethod
    def _label_for_class(cls, class_id: int, model_name: str) -> str:
        if model_name == "doclaynet":
            return cls.DOC_LAYNET_LABEL_MAP.get(class_id, f"class_{class_id}")
        # keremberke / foduucom – single “Table” class
        return "Table"

    # ──────────────────────────────────────────────────────────────
    # 5.  INFERENCE
    # ──────────────────────────────────────────────────────────────
    def get_yolo_coords(
        self,
        pdf_model,
        zoom: float = 2.0,
        model_name: str = "keremberke",
        model_overrides: Optional[Dict[str, Union[int, float, bool]]] = None,
    ) -> None:

        model = self._load_model(model_name)

        # apply per‑call overrides (conf, iou, etc.)
        runtime_overrides = dict(
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            agnostic_nms=False,
        )
        if model_overrides:
            runtime_overrides.update(model_overrides)
        model.overrides.update(runtime_overrides)

        self.tbl_coordinates, self.table_bboxes = {}, []
        self.metadata, self.page_dimensions = {}, {}

        for page_num, _ in enumerate(pdf_model.doc):
            _, np_img, img_w, img_h = pdf_model.page_to_numpy(page_num, zoom=zoom)
            self.page_dimensions[page_num] = {"image_width": img_w, "image_height": img_h}

            dets = model.predict(np_img, verbose=False)[0].boxes.cpu().data.numpy()

            tables, others = [], {}
            for x1, y1, x2, y2, conf, cls_id in dets:
                label = self._label_for_class(int(cls_id), model_name)
                page_bbox = pdf_model.img_bbox_to_page_bbox((x1, y1, x2, y2),
                                                            (img_w, img_h),
                                                            page_num, zoom=zoom)
                combined = tuple(map(float,
                                     pdf_model.page_bbox_to_combined(page_num, page_bbox)))
                if label == "Table":
                    tables.append(combined); self.table_bboxes.append(combined)
                else:
                    others.setdefault(label, []).append(combined)

            self.tbl_coordinates[page_num] = tables
            self.metadata[page_num] = others if model_name == "doclaynet" else None

    # 6.  PUBLIC ORCHESTRATOR
    # ──────────────────────────────────────────────────────────────
    def run(
        self,
        pdf_model,
        zoom: float = 2.0,
        model_name: str = "keremberke",
        model_overrides: Optional[Dict[str, Union[int, float, bool]]] = None,
    ) -> Dict[str, object]:

        self.get_yolo_coords(
            pdf_model,
            zoom=zoom,
            model_name=model_name,
            model_overrides=model_overrides,
        )

        return {
            "tbl_coordinates": self.tbl_coordinates,
            "table_bboxes": self.table_bboxes,
            "metadata": self.metadata,
            "page_dimensions": self.page_dimensions,
        }
    # ──────────────────────────────────────────────────────────────
    # 7.  UTIL
    # ──────────────────────────────────────────────────────────────
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        return {n: cfg["description"] for n, cfg in cls.AVAILABLE_MODELS.items()}


# from huggingface_hub import hf_hub_download
# import torch
# import ultralytics
# import torch.nn.modules.container
# import ultralytics.nn.tasks
# import ultralytics.nn.modules
# from ultralytics import YOLO, YOLOv10
# import numpy as np
# from PIL import Image
# from typing import List, Dict, Union
# import os
# import requests
# import hashlib
# from pathlib import Path

# os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'

# class YOLOTableDetector:
#     """Standalone YOLO table detector service with multiple model support"""
    
#     # Available model configurations
#     AVAILABLE_MODELS = {
#         "keremberke": {
#             "type": "huggingface",
#             "repo_id": "keremberke/yolov8m-table-extraction",
#             "filename": "best.pt",
#             "description": "YOLOv8 medium model for table extraction"
#         },
#         "foduucom": {
#             "type": "huggingface", 
#             "repo_id": "foduucom/table-detection-and-extraction",
#             "filename": "best.pt",
#             "description": "Alternative YOLOv8 model for table detection"
#         },
#         "doclaynet": {
#             "type": "github_release",
#             "url": "https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt",
#             "filename": "yolov10x_best.pt",
#             "description": "YOLOv10 model for document layout analysis including tables",
#             "expected_sha256": None
#         }
#     }
    
#     def __init__(
#         self,
#         model_data_dir: str = "/data",
#         confidence_threshold: float = 0.25,
#         iou_threshold: float = 0.45,
#         max_detections: int = 1000
#     ):
#         self.model_data_dir = Path(model_data_dir)
#         self.model_data_dir.mkdir(parents=True, exist_ok=True)
#         self.confidence_threshold = confidence_threshold
#         self.iou_threshold = iou_threshold
#         self.max_detections = max_detections
        
#         # Cache for loaded models to avoid reloading
#         self.model_cache = {}
    
#     def _download_github_model(self, model_config: dict) -> str:
#         """Download model from GitHub releases"""
#         url = model_config["url"]
#         filename = model_config["filename"]
#         local_path = self.model_data_dir / filename
        
#         # Check if file already exists
#         if local_path.exists():
#             print(f"Model {filename} already exists at {local_path}")
#             return str(local_path)
        
#         print(f"Downloading {filename} from {url}...")
        
#         try:
#             response = requests.get(url, stream=True)
#             response.raise_for_status()
            
#             # Download with progress
#             total_size = int(response.headers.get('content-length', 0))
#             downloaded_size = 0
            
#             with open(local_path, 'wb') as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     if chunk:
#                         f.write(chunk)
#                         downloaded_size += len(chunk)
#                         if total_size > 0:
#                             progress = (downloaded_size / total_size) * 100
#                             print(f"\rDownload progress: {progress:.1f}%", end='')
            
#             print(f"\nDownload completed: {local_path}")
            
#             # Verify file size
#             if total_size > 0 and local_path.stat().st_size != total_size:
#                 raise Exception(f"Downloaded file size mismatch. Expected: {total_size}, Got: {local_path.stat().st_size}")
            
#             # Optional: Verify SHA256 if provided
#             if model_config.get("expected_sha256"):
#                 file_hash = self._calculate_sha256(local_path)
#                 if file_hash != model_config["expected_sha256"]:
#                     raise Exception(f"SHA256 verification failed. Expected: {model_config['expected_sha256']}, Got: {file_hash}")
#                 print("SHA256 verification passed")
            
#             return str(local_path)
            
#         except Exception as e:
#             # Clean up partial download
#             if local_path.exists():
#                 local_path.unlink()
#             raise Exception(f"Failed to download model from {url}: {str(e)}")
    
#     def _calculate_sha256(self, file_path: Path) -> str:
#         """Calculate SHA256 hash of a file"""
#         sha256_hash = hashlib.sha256()
#         with open(file_path, "rb") as f:
#             for chunk in iter(lambda: f.read(4096), b""):
#                 sha256_hash.update(chunk)
#         return sha256_hash.hexdigest()
        
#     def _safe_tensor_load_model(self, model_type: str = None) -> Union[YOLO, YOLOv10]:
#         """Load YOLO or YOLOv10 model based on type"""
#         if model_type is None:
#             model_type = self.model_type

#         model_config = self.AVAILABLE_MODELS[model_type]

#         print(model_config)
#         # Get model path
#         if model_config["type"] == "huggingface":
#             local_pt = hf_hub_download(
#                 repo_id=model_config["repo_id"],
#                 filename=model_config["filename"], 
#                 cache_dir=str(self.model_data_dir)
#             )
#         elif model_config["type"] == "github_release":
#             local_pt = self._download_github_model(model_config)
#         else:
#             raise ValueError(f"Unsupported model type: {model_config['type']}")

#         # Patch torch.load to avoid "weights_only" errors in Ultralytics internals
#         original_torch_load = torch.load

#         def patched_torch_load(*args, **kwargs):
#             kwargs.setdefault('weights_only', False)
#             return original_torch_load(*args, **kwargs)

#         torch.load = patched_torch_load

#         try:
#             if model_type == "doclaynet":
#                 model = YOLOv10(local_pt)
#             else:
#                 model = YOLO(local_pt)
#             print(f"Successfully loaded {model_type} model from {local_pt}")
#             return model
#         finally:
#             torch.load = original_torch_load
    
#     def _load_model(self, model_type: str) -> Union[YOLO,YOLOv10]:
#         """Load and cache YOLO model"""
#         # Return cached model if available
#         if model_type in self.model_cache:
#             print(f"Using cached {model_type} model")
#             return self.model_cache[model_type]
        
#         # Validate model type
#         if model_type not in self.AVAILABLE_MODELS:
#             raise ValueError(f"Model type '{model_type}' not supported. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
#         print(f"Loading model: {model_type}")
#         model = self._safe_tensor_load_model(model_type)
        
#         # Set model parameters
#         model.overrides['conf'] = self.confidence_threshold
#         model.overrides['iou'] = self.iou_threshold
#         model.overrides['agnostic_nms'] = False
#         model.overrides['max_det'] = self.max_detections
        
#         # Cache the model
#         self.model_cache[model_type] = model
#         return model

#     def get_yolo_coords(self, pdf_model, zoom: float = 2.0, model_overrides: dict = None, model_type: str = "keremberke") -> None:
#         """
#         Detect table coordinates across PDF pages using YOLO model.
        
#         Args:
#             pdf_model: PDFModel instance to process
#             zoom: Zoom factor for rendering pages
#             model_overrides: Optional dictionary to override default model parameters
#             model_type: Model type to use for this detection
        
#         Populates:
#         - self.tbl_coordinates: Dictionary of page-wise table coordinates
#         - self.page_dimensions: Dictionary of page image dimensions
#         - self.table_bboxes: Flattened list of all table bounding boxes
#         """
#         # Load the requested model
#         model = self._load_model(model_type)
        
#         # Default model parameters
#         default_overrides = {
#             'conf': 0.25,    # NMS confidence threshold
#             'iou': 0.45,     # NMS IoU threshold
#             'agnostic_nms': False,  # NMS class-agnostic
#             'max_det': 1000  # maximum number of detections per image
#         }
        
#         # Apply any custom model overrides
#         if model_overrides:
#             default_overrides.update(model_overrides)
        
#         # Set model parameters
#         for key, value in default_overrides.items():
#             model.overrides[key] = value
        
#         self.tbl_coordinates: dict[int, list[tuple[float, float, float, float]]] = {}
#         self.page_dimensions: dict[int, dict[str, int]] = {}
#         self.table_bboxes: list[tuple[float, float, float, float]] = []
        
#         for page_number, page in enumerate(pdf_model.doc):
#             # 1. render page as numpy
#             _, np_img, img_w, img_h = pdf_model.page_to_numpy(
#                 page_number, zoom=zoom
#             )
#             self.page_dimensions[page_number] = dict(image_width=img_w, image_height=img_h)
            
#             # 2. YOLO inference
#             yolo_boxes = model.predict(np_img)[0].boxes.cpu().data.numpy()
#             print(yolo_boxes)
#             combined_boxes = []
            
#             for x1, y1, x2, y2, *_ in yolo_boxes:
#                 # 2.a px → PDF-page pts
#                 page_bbox = pdf_model.img_bbox_to_page_bbox(
#                     (x1, y1, x2, y2), (img_w, img_h), page_number, zoom=zoom
#                 )
                
#                 # 2.b PDF-page pts → combined-PDF pts
#                 combined_bbox = pdf_model.page_bbox_to_combined(page_number, page_bbox)
#                 combined_boxes.append(tuple(map(float, combined_bbox)))
#                 self.table_bboxes.append(tuple(map(float, combined_bbox)))
            
#             self.tbl_coordinates[page_number] = combined_boxes

#     def run(self, pdf_model, zoom: float = 2.0, model_overrides: dict = None, model_type: str = "keremberke") -> dict[int, list[tuple[float, float, float, float]]]:
#         """
#         Run table detection and return coordinates.
        
#         Args:
#             pdf_model: PDFModel instance to process
#             zoom: Zoom factor for rendering pages
#             model_overrides: Optional dictionary to override default model parameters
#             model_type: Model type to use for this detection
        
#         Returns:
#         - Dictionary containing table coordinates by page and flattened table bboxes
#         """
#         self.get_yolo_coords(pdf_model, zoom=zoom, model_overrides=model_overrides, model_type=model_type)
#         return {
#             'tbl_coordinates': self.tbl_coordinates, 
#             'table_bboxes': self.table_bboxes,
#             'model_used': model_type
#         }
    
#     @classmethod
#     def list_available_models(cls) -> dict:
#         """List all available models with their descriptions"""
#         return {name: config["description"] for name, config in cls.AVAILABLE_MODELS.items()}