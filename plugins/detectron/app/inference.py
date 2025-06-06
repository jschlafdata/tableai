import os
import yaml
from typing import List, Dict, Union, Optional
import layoutparser as lp
import numpy as np
from collections import defaultdict

# Import your existing classes
from tableai_tools.pdf.models import PDFModel

class ModelConfigLoader:
    """Load model configurations from YAML files"""
    
    def __init__(self, base_dir='/data/models'):
        self.base_dir = base_dir
        self.label_map, self.config_maps = self._load_maps()

    def _load_maps(self):
        yaml_file_path = os.path.join(self.base_dir, 'LabelMaps/label_map.yaml')
        conf_file_path = os.path.join(self.base_dir, 'LabelMaps/config_map.yaml')
        
        with open(yaml_file_path, 'r') as file:
            label_map = yaml.safe_load(file)
        
        with open(conf_file_path, 'r') as file:
            config_maps = yaml.safe_load(file)
        
        return label_map, config_maps

    def get_model_confs(self, model):
        model_conf_maps = self.config_maps[model]
        dataset = model_conf_maps['dataset']
        model_name = model_conf_maps['model']
        label_mappings = self.label_map[dataset]
        model_file_path = os.path.join(self.base_dir, dataset, f"models/{model_name}.pth")
        config_file_path = os.path.join(self.base_dir, dataset, f"configs/{model_name}.yml")
        
        return config_file_path, model_file_path, label_mappings

    def get_mods(self, model, threshold=0.8, nms_threshold=0.3):
        """
        Configure and return a Detectron2LayoutModel with specified thresholds.

        Parameters:
        - model: The model configuration to use.
        - threshold (float): Minimum confidence score for detections to be considered. 
        - nms_threshold (float): Non-maximum suppression threshold for removing overlapping bounding boxes.

        Returns:
        - model: Configured Detectron2LayoutModel instance.
        """
        config_file_path, model_file_path, label_mappings = self.get_model_confs(model)

        model = lp.Detectron2LayoutModel(
            config_path=config_file_path,
            model_path=model_file_path,
            label_map=label_mappings,
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST", threshold
                # "MODEL.ROI_HEADS.NMS_THRESH_TEST", nms_threshold
            ]
        )
        return model


class DetectronTableDetector:
    """Standalone Detectron2 table detector service following YOLO framework"""
    
    def __init__(
        self,
        model_type: str = "PubLayNet",
        model_data_dir: str = "/data/models",
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.3
    ):
        self.model_type = model_type
        self.model_data_dir = model_data_dir
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = None
        self.model_loader = None
        
        # Initialize model on startup
        self._load_model()
        
    def _load_model(self):
        """Initialize the Detectron2 model"""
        if self.model is None:
            self.model_loader = ModelConfigLoader(base_dir=self.model_data_dir)
            
            # Set model source based on type
            if self.model_type == 'PubLayNet':
                model_source = 'PubLayNet/mask_rcnn_R_50_FPN_3x'
            elif self.model_type == 'TableBank':
                model_source = 'TableBank/faster_rcnn_R_50_FPN_3x'
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.model = self.model_loader.get_mods(
                model_source, 
                threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold
            )

    def detected_table_rects(self, model_results, selector='Table'):
        """Extract table rectangles from model results"""
        # Filter results for tables only
        table_blocks = model_results.filter_by(selector, include_parent=False)
        
        # Extract coordinates
        coordinates = []
        for block in table_blocks:
            # Get bounding box coordinates
            x1, y1, x2, y2 = block.coordinates
            coordinates.append((x1, y1, x2, y2))
        
        return coordinates

    def get_detectron_coords(self, pdf_model: PDFModel, zoom: float = 2.0, model_overrides: dict = None) -> None:
        """
        Detect table coordinates across PDF pages using Detectron2 model.
        
        Args:
            pdf_model: PDFModel instance to process
            zoom: Zoom factor for rendering pages
            model_overrides: Optional dictionary to override default model parameters
        
        Populates:
        - self.model_results: Dictionary of page-wise model detection results
        - self.tbl_coordinates: Dictionary of page-wise table coordinates
        - self.page_dimensions: Dictionary of page image dimensions
        - self.table_bboxes: Flattened list of all table bounding boxes
        """
        # Apply model overrides if provided
        if model_overrides:
            # You might want to reinitialize the model with new parameters
            if 'confidence_threshold' in model_overrides:
                self.confidence_threshold = model_overrides['confidence_threshold']
            if 'nms_threshold' in model_overrides:
                self.nms_threshold = model_overrides['nms_threshold']
            # Reload model with new parameters
            self.model = None
            self._load_model()
        
        self.model_results: dict[int, object] = {}
        self.tbl_coordinates: dict[int, list[tuple[float, float, float, float]]] = {}
        self.page_dimensions: dict[int, dict[str, int]] = {}
        self.table_bboxes: list[tuple[float, float, float, float]] = []
        
        for page_number, page in enumerate(pdf_model.doc):
            # 1. Render page as numpy array
            _, np_img, img_w, img_h = pdf_model.page_to_numpy(
                page_number, zoom=zoom
            )
            self.page_dimensions[page_number] = dict(image_width=img_w, image_height=img_h)
            
            # 2. Detectron2 inference
            model_results = self.model.detect(np_img)
            self.model_results[page_number] = model_results
            
            # 3. Extract table coordinates
            tbl_coordinates_px = self.detected_table_rects(model_results, selector='Table')
            combined_boxes = []
            
            for x1, y1, x2, y2 in tbl_coordinates_px:
                # 3.a px → PDF-page pts
                page_bbox = pdf_model.img_bbox_to_page_bbox(
                    (x1, y1, x2, y2), (img_w, img_h), page_number, zoom=zoom
                )
                
                # 3.b PDF-page pts → combined-PDF pts
                combined_bbox = pdf_model.page_bbox_to_combined(page_number, page_bbox)
                combined_boxes.append(tuple(map(float, combined_bbox)))
                self.table_bboxes.append(tuple(map(float, combined_bbox)))
            
            self.tbl_coordinates[page_number] = combined_boxes

    def process_detectron_coords(self, pdf_model: PDFModel, margins: dict = None):
        """
        Process detected coordinates with additional logic.
        This is where you'd add your Coordinates, TotalsMagic, etc. logic
        """
        if margins is None:
            margins = {}
            
        table_coordinates = defaultdict(lambda: {'tables': {}, 'totals': {}})

        for page_number, page in enumerate(pdf_model.doc):
            # Your existing coordinate processing logic would go here
            # For now, just return the basic table coordinates
            model_results = self.model_results[page_number]
            tbl_coordinates = self.tbl_coordinates[page_number]
            image_width = self.page_dimensions[page_number]['image_width']
            image_height = self.page_dimensions[page_number]['image_height']

            for index, tbl_bbox in enumerate(tbl_coordinates):
                # Convert to the format your downstream code expects
                table_coordinates[page_number]['tables'][index] = {
                    'rect': tbl_bbox,
                    'confidence': 1.0,  # You might want to extract actual confidence from model_results
                    'model_results': model_results  # Store full results if needed
                }

        return dict(table_coordinates)

    def run(self, pdf_model: PDFModel, zoom: float = 2.0, model_overrides: dict = None) -> dict:
        """
        Run table detection and return coordinates.
        
        Args:
            pdf_model: PDFModel instance to process
            zoom: Zoom factor for rendering pages
            model_overrides: Optional dictionary to override default model parameters
        
        Returns:
        - Dictionary containing table coordinates by page and additional metadata
        """
        self.get_detectron_coords(pdf_model, zoom=zoom, model_overrides=model_overrides)
        processed_coords = self.process_detectron_coords(pdf_model)
        
        return {
            'tbl_coordinates': self.tbl_coordinates,
            'table_bboxes': self.table_bboxes,
            'processed_coordinates': processed_coords,
            'model_results': self.model_results,
            'page_dimensions': self.page_dimensions
        }