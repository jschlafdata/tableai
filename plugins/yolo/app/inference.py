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

os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'

class YOLOTableDetector:
    """Standalone YOLO table detector service"""
    
    def __init__(
        self,
        model_type: str = "keremberke",
        model_data_dir: str = "/data",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 1000
    ):
        self.model_type = (
            "keremberke/yolov8m-table-extraction"
            if model_type == "keremberke"
            else "foduucom/table-detection-and-extraction"
        )
        self.model_data_dir = model_data_dir
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.model = None
        
        # Initialize model on startup
        self._load_model()
        
    def _safe_tensor_load_model(self):
        """Load YOLO model with safe tensor loading"""
        self.local_pt = hf_hub_download(
            repo_id=self.model_type,
            filename="best.pt", 
            cache_dir=self.model_data_dir
        )
        
        # Monkey patch torch.load to use weights_only=False for ultralytics
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # Temporarily replace torch.load
        torch.load = patched_torch_load
        
        try:
            model = YOLO(self.local_pt)
            return model
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
        
    def _load_model(self):
        """Initialize the YOLO model"""
        if self.model is None:
            self.model = self._safe_tensor_load_model()
            
            # Set model parameters
            self.model.overrides['conf'] = self.confidence_threshold
            self.model.overrides['iou'] = self.iou_threshold
            self.model.overrides['agnostic_nms'] = False
            self.model.overrides['max_det'] = self.max_detections

    def get_yolo_coords(self, pdf_model, zoom: float = 2.0, model_overrides: dict = None) -> None:
        """
        Detect table coordinates across PDF pages using YOLO model.
        
        Args:
            pdf_model: PDFModel instance to process
            zoom: Zoom factor for rendering pages
            model_overrides: Optional dictionary to override default model parameters
        
        Populates:
        - self.tbl_coordinates: Dictionary of page-wise table coordinates
        - self.page_dimensions: Dictionary of page image dimensions
        - self.table_bboxes: Flattened list of all table bounding boxes
        """
        model = self._safe_tensor_load_model()
        
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

    def run(self, pdf_model, zoom: float = 2.0, model_overrides: dict = None) -> dict[int, list[tuple[float, float, float, float]]]:
        """
        Run table detection and return coordinates.
        
        Args:
            pdf_model: PDFModel instance to process
            zoom: Zoom factor for rendering pages
            model_overrides: Optional dictionary to override default model parameters
        
        Returns:
        - Dictionary containing table coordinates by page and flattened table bboxes
        """
        self.get_yolo_coords(pdf_model, zoom=zoom, model_overrides=model_overrides)
        return {
            'tbl_coordinates': self.tbl_coordinates, 
            'table_bboxes': self.table_bboxes
        }