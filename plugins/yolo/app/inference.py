from huggingface_hub import hf_hub_download
import torch
import ultralytics
import torch.nn.modules.container
import ultralytics.nn.tasks
import ultralytics.nn.modules
from ultralytics import YOLO
import numpy as np
from PIL import Image
from typing import List, Dict, Union
import os
import requests
import hashlib
from pathlib import Path

os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'

class YOLOTableDetector:
    """Standalone YOLO table detector service with multiple model support"""
    
    # Available model configurations
    AVAILABLE_MODELS = {
        "keremberke": {
            "type": "huggingface",
            "repo_id": "keremberke/yolov8m-table-extraction",
            "filename": "best.pt",
            "description": "YOLOv8 medium model for table extraction"
        },
        "foduucom": {
            "type": "huggingface", 
            "repo_id": "foduucom/table-detection-and-extraction",
            "filename": "best.pt",
            "description": "Alternative YOLOv8 model for table detection"
        },
        "doclaynet": {
            "type": "github_release",
            "url": "https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt",
            "filename": "yolov10x_best.pt",
            "description": "YOLOv10 model for document layout analysis including tables",
            "expected_sha256": None  # Add if you have the hash
        }
    }
    
    def __init__(
        self,
        model_data_dir: str = "/data",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 1000
    ):
        self.model_data_dir = Path(model_data_dir)
        self.model_data_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        
        # Cache for loaded models to avoid reloading
        self.model_cache = {}
    
    def _download_github_model(self, model_config: dict) -> str:
        """Download model from GitHub releases"""
        url = model_config["url"]
        filename = model_config["filename"]
        local_path = self.model_data_dir / filename
        
        # Check if file already exists
        if local_path.exists():
            print(f"Model {filename} already exists at {local_path}")
            return str(local_path)
        
        print(f"Downloading {filename} from {url}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end='')
            
            print(f"\nDownload completed: {local_path}")
            
            # Verify file size
            if total_size > 0 and local_path.stat().st_size != total_size:
                raise Exception(f"Downloaded file size mismatch. Expected: {total_size}, Got: {local_path.stat().st_size}")
            
            # Optional: Verify SHA256 if provided
            if model_config.get("expected_sha256"):
                file_hash = self._calculate_sha256(local_path)
                if file_hash != model_config["expected_sha256"]:
                    raise Exception(f"SHA256 verification failed. Expected: {model_config['expected_sha256']}, Got: {file_hash}")
                print("SHA256 verification passed")
            
            return str(local_path)
            
        except Exception as e:
            # Clean up partial download
            if local_path.exists():
                local_path.unlink()
            raise Exception(f"Failed to download model from {url}: {str(e)}")
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
        
    def _safe_tensor_load_model(self, model_type: str = None) -> YOLO:
        """Load YOLO model with safe tensor loading"""
        if model_type is None:
            model_type = self.model_type
            
        model_config = self.AVAILABLE_MODELS[model_type]
        
        # Download/get model path based on type
        if model_config["type"] == "huggingface":
            local_pt = hf_hub_download(
                repo_id=model_config["repo_id"],
                filename=model_config["filename"], 
                cache_dir=str(self.model_data_dir)
            )
        elif model_config["type"] == "github_release":
            local_pt = self._download_github_model(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        # Monkey patch torch.load to use weights_only=False for ultralytics
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # Temporarily replace torch.load
        torch.load = patched_torch_load
        
        try:
            model = YOLO(local_pt)
            print(f"Successfully loaded {model_type} model from {local_pt}")
            return model
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
        
    def _load_model(self, model_type: str) -> YOLO:
        """Load and cache YOLO model"""
        # Return cached model if available
        if model_type in self.model_cache:
            print(f"Using cached {model_type} model")
            return self.model_cache[model_type]
        
        # Validate model type
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        print(f"Loading model: {model_type}")
        model = self._safe_tensor_load_model(model_type)
        
        # Set model parameters
        model.overrides['conf'] = self.confidence_threshold
        model.overrides['iou'] = self.iou_threshold
        model.overrides['agnostic_nms'] = False
        model.overrides['max_det'] = self.max_detections
        
        # Cache the model
        self.model_cache[model_type] = model
        return model

    def get_yolo_coords(self, pdf_model, zoom: float = 2.0, model_overrides: dict = None, model_type: str = "keremberke") -> None:
        """
        Detect table coordinates across PDF pages using YOLO model.
        
        Args:
            pdf_model: PDFModel instance to process
            zoom: Zoom factor for rendering pages
            model_overrides: Optional dictionary to override default model parameters
            model_type: Model type to use for this detection
        
        Populates:
        - self.tbl_coordinates: Dictionary of page-wise table coordinates
        - self.page_dimensions: Dictionary of page image dimensions
        - self.table_bboxes: Flattened list of all table bounding boxes
        """
        # Load the requested model
        model = self._load_model(model_type)
        
        # Default model parameters
        default_overrides = {
            'conf': 0.25,    # NMS confidence threshold
            'iou': 0.45,     # NMS IoU threshold
            'agnostic_nms': False,  # NMS class-agnostic
            'max_det': 1000  # maximum number of detections per image
        }
        
        # Apply any custom model overrides
        if model_overrides:
            default_overrides.update(model_overrides)
        
        # Set model parameters
        for key, value in default_overrides.items():
            model.overrides[key] = value
        
        self.tbl_coordinates: dict[int, list[tuple[float, float, float, float]]] = {}
        self.page_dimensions: dict[int, dict[str, int]] = {}
        self.table_bboxes: list[tuple[float, float, float, float]] = []
        
        for page_number, page in enumerate(pdf_model.doc):
            # 1. render page as numpy
            _, np_img, img_w, img_h = pdf_model.page_to_numpy(
                page_number, zoom=zoom
            )
            self.page_dimensions[page_number] = dict(image_width=img_w, image_height=img_h)
            
            # 2. YOLO inference
            yolo_boxes = model.predict(np_img)[0].boxes.cpu().data.numpy()
            combined_boxes = []
            
            for x1, y1, x2, y2, *_ in yolo_boxes:
                # 2.a px → PDF-page pts
                page_bbox = pdf_model.img_bbox_to_page_bbox(
                    (x1, y1, x2, y2), (img_w, img_h), page_number, zoom=zoom
                )
                
                # 2.b PDF-page pts → combined-PDF pts
                combined_bbox = pdf_model.page_bbox_to_combined(page_number, page_bbox)
                combined_boxes.append(tuple(map(float, combined_bbox)))
                self.table_bboxes.append(tuple(map(float, combined_bbox)))
            
            self.tbl_coordinates[page_number] = combined_boxes

    def run(self, pdf_model, zoom: float = 2.0, model_overrides: dict = None, model_type: str = "keremberke") -> dict[int, list[tuple[float, float, float, float]]]:
        """
        Run table detection and return coordinates.
        
        Args:
            pdf_model: PDFModel instance to process
            zoom: Zoom factor for rendering pages
            model_overrides: Optional dictionary to override default model parameters
            model_type: Model type to use for this detection
        
        Returns:
        - Dictionary containing table coordinates by page and flattened table bboxes
        """
        self.get_yolo_coords(pdf_model, zoom=zoom, model_overrides=model_overrides, model_type=model_type)
        return {
            'tbl_coordinates': self.tbl_coordinates, 
            'table_bboxes': self.table_bboxes,
            'model_used': model_type
        }
    
    @classmethod
    def list_available_models(cls) -> dict:
        """List all available models with their descriptions"""
        return {name: config["description"] for name, config in cls.AVAILABLE_MODELS.items()}