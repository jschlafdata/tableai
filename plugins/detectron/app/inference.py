from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import layoutparser as lp
import numpy as np
from pydantic import BaseModel
import json

# Assuming your refactored PDFModel is available
from tableai.pdf.pdf_model import PDFModel

# ──────────────────────────────────────────────────────────────────────────
# 1. Pydantic Output Models
# ──────────────────────────────────────────────────────────────────────────
BoundingBox = Tuple[float, float, float, float]
DetectionMetadata = Dict[str, List[BoundingBox]]

class PageDimensions(BaseModel):
    image_width: int
    image_height: int

class DetectronDetectionResult(BaseModel):
    """Standardized output for Detectron2 detection results."""
    # The primary detected regions (e.g., tables)
    primary_coordinates: Dict[int, List[BoundingBox]]
    # All other detected regions, categorized by label
    metadata: Dict[int, DetectionMetadata]
    page_dimensions: Dict[int, PageDimensions]
    # LayoutParser's raw results, if needed for debugging (can be excluded for production)
    model_results: Dict[int, Any] # lp.Layout is not easily serializable

    class Config:
        arbitrary_types_allowed = True


# ──────────────────────────────────────────────────────────────────────────
# 2. The Detector Class
# ──────────────────────────────────────────────────────────────────────────
class DetectronTableDetector:
    _DATASET_CONFIGS = {
        "PubLayNet": {"lp_uri": "lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config", "label_map": {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}},
        "TableBank": {"lp_uri": "lp://TableBank/faster_rcnn_R_50_FPN_3x/config", "label_map": {0: "Table"}},
        "HJDataset": {"lp_uri": "lp://HJDataset/retinanet_R_50_FPN_3x/config", "label_map": {1: "Page Frame", 2: "Row", 3: "Title Region", 4: "Text Region", 5: "Title", 6: "Subtitle", 7: "Other"}},
        "PrimaLayout": {"lp_uri": "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config", "label_map": {1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion", 6: "OtherRegion"}},
    }

    def __init__(self, dataset: str = "PubLayNet", confidence_threshold: Optional[float] = 0.5):
        if dataset not in self._DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset}'. Must be one of: {list(self._DATASET_CONFIGS.keys())}")
        
        self.dataset = dataset
        self.confidence_threshold = confidence_threshold
        cfg = self._DATASET_CONFIGS[dataset]
        self.model = lp.models.Detectron2LayoutModel(
            config_path=cfg["lp_uri"],
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.confidence_threshold],
            label_map=cfg["label_map"],
        )

    def _get_default_selector(self) -> str:
        """Determines the default primary object type for a dataset."""
        if self.dataset in ["PubLayNet", "TableBank"]: return "Table"
        if self.dataset == "PrimaLayout": return "TableRegion"
        if self.dataset == "HJDataset": return "Row"
        return "Text" # A safe fallback

    def _detect_on_page(
        self, np_img: np.ndarray, pdf_model: PDFModel, page_num: int, zoom: float, selector: Optional[str]
    ) -> Tuple[List[BoundingBox], DetectionMetadata, lp.Layout]:
        """Runs detection on a single numpy image and converts coordinates."""
        img_h, img_w, _ = np_img.shape
        layout_result = self.model.detect(np_img)
        
        primary_selector = selector or self._get_default_selector()
        
        primary_coords = []
        metadata = defaultdict(list)
        
        for block in layout_result:
            # Convert pixel coordinates to PDF-point coordinates
            page_bbox = pdf_model.img_bbox_to_page_bbox(
                block.coordinates, (img_w, img_h), page_num, zoom=zoom
            )
            # We will use page_bbox for this service, not stitched doc coordinates
            coords_tuple = tuple(map(float, page_bbox))

            if block.type == primary_selector:
                primary_coords.append(coords_tuple)
            else:
                metadata[block.type].append(coords_tuple)
        
        return primary_coords, dict(metadata), layout_result

    def run(self, pdf_model: PDFModel, zoom: float = 2.0, selector: Optional[str] = None) -> DetectronDetectionResult:
        """Processes an entire PDF document, page by page."""
        all_primary_coords = {}
        all_metadata = {}
        all_page_dims = {}
        all_model_results = {}
        
        for page_num in range(pdf_model.source_doc.page_count):
            _, np_img, img_w, img_h = pdf_model.page_to_numpy(page_num, zoom=zoom)
            
            primary, meta, layout = self._detect_on_page(np_img, pdf_model, page_num, zoom, selector)
            
            all_primary_coords[page_num] = primary
            all_metadata[page_num] = meta
            all_page_dims[page_num] = PageDimensions(image_width=img_w, image_height=img_h)
            all_model_results[page_num] = layout
            
        return DetectronDetectionResult(
            primary_coordinates=all_primary_coords,
            metadata=all_metadata,
            page_dimensions=all_page_dims,
            model_results=all_model_results,
        )

    def run_on_image(
        self, np_img: np.ndarray, pdf_model: PDFModel, page_num: int, zoom: float = 2.0, selector: Optional[str] = None
    ) -> DetectronDetectionResult:
        """Processes a single pre-rendered image."""
        img_h, img_w, _ = np_img.shape

        primary, meta, layout = self._detect_on_page(np_img, pdf_model, page_num, zoom, selector)
        
        return DetectronDetectionResult(
            primary_coordinates={page_num: primary},
            metadata={page_num: meta},
            page_dimensions={page_num: PageDimensions(image_width=img_w, image_height=img_h)},
            model_results={page_num: layout},
        )

    @classmethod
    def list_available_datasets(cls) -> List[str]:
        return list(cls._DATASET_CONFIGS.keys())