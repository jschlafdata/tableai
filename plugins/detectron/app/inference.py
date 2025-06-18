from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import layoutparser as lp
import numpy as np
from pydantic import BaseModel, field_serializer

# Assuming your refactored PDFModel is available
from tableai.pdf.pdf_model import PDFModel

# --- Pydantic models (unchanged) ---
BoundingBox = Tuple[float, float, float, float]
DetectionMetadata = Dict[str, List[BoundingBox]]
class PageDimensions(BaseModel): pass
class DetectronDetectionResult(BaseModel):
    primary_coordinates: Dict[int, List[BoundingBox]]
    metadata: Dict[int, DetectionMetadata]
    page_dimensions: Dict[int, PageDimensions]
    model_results: Dict[int, lp.Layout] # Keep the original type hint

    # This custom serializer tells Pydantic what to do when it encounters a `lp.Layout` object
    @field_serializer('model_results')
    def serialize_layout_results(self, layouts: Dict[int, lp.Layout]) -> Dict[int, Dict]:
        return {page: layout.to_dict() for page, layout in layouts.items()}

    class Config:
        arbitrary_types_allowed = True
# --- End Pydantic models ---

class DetectronTableDetector:
    """
    Detectron2-based detector with support for dynamic model configuration.
    """
    _DATASET_CONFIGS = {
        "PubLayNet": {"lp_uri": "lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config", "label_map": {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}},
        "TableBank": {"lp_uri": "lp://TableBank/faster_rcnn_R_50_FPN_3x/config", "label_map": {0: "Table"}},
        "HJDataset": {"lp_uri": "lp://HJDataset/retinanet_R_50_FPN_3x/config", "label_map": {1: "Page Frame", 2: "Row", 3: "Title Region", 4: "Text Region", 5: "Title", 6: "Subtitle", 7: "Other"}},
        "PrimaLayout": {"lp_uri": "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config", "label_map": {1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion", 6: "OtherRegion"}},
    }

    def __init__(self, dataset: str = "PubLayNet", extra_config_opts: Optional[Dict[str, Any]] = None):
        """
        Initializes the detector with dynamic configuration.

        Args:
            dataset: The name of the dataset model to load.
            extra_config_opts: A dictionary of Detectron2 configuration options to override.
                            e.g., {"MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.7, "MODEL.DEVICE": "cpu"}
        """
        if dataset not in self._DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset}'. Must be one of: {list(self._DATASET_CONFIGS.keys())}")
        
        self.dataset = dataset
        cfg = self._DATASET_CONFIGS[dataset]
        
        # Start with default confidence threshold
        config_dict = {"MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.5}
        
        # Merge user-provided overrides
        if extra_config_opts:
            config_dict.update(extra_config_opts)
            print(f"-> Initializing Detectron model '{dataset}' with custom config: {config_dict}")
        
        # Convert the dictionary to the list format required by layoutparser
        # e.g., ["KEY1", value1, "KEY2", value2]
        final_extra_config = []
        for key, value in config_dict.items():
            final_extra_config.extend([key, value])

        self.model = lp.models.Detectron2LayoutModel(
            config_path=cfg["lp_uri"],
            extra_config=final_extra_config,
            label_map=cfg["label_map"],
        )
        print(f"✓ Detectron model '{dataset}' loaded with config: {config_dict}")



    # ... The rest of the class methods (_get_default_selector, _detect_on_page, run, run_on_image)
    # ... do NOT need to be changed, as the model is fully configured at __init__ time.
    def _get_default_selector(self) -> str:
        if self.dataset in ["PubLayNet", "TableBank"]: return "Table"
        if self.dataset == "PrimaLayout": return "TableRegion"
        if self.dataset == "HJDataset": return "Row"
        return "Text"

    def _detect_on_page(self, np_img: np.ndarray, pdf_model: PDFModel, page_num: int, zoom: float, selector: Optional[str]) -> Tuple[List[BoundingBox], DetectionMetadata, lp.Layout]:
        img_h, img_w, _ = np_img.shape
        layout_result = self.model.detect(np_img)
        primary_selector = selector or self._get_default_selector()
        primary_coords, metadata = [], defaultdict(list)
        for block in layout_result:
            page_bbox = pdf_model.img_bbox_to_page_bbox(block.coordinates, (img_w, img_h), page_num, zoom=zoom)
            coords_tuple = tuple(map(float, page_bbox))
            if block.type == primary_selector:
                primary_coords.append(coords_tuple)
            else:
                metadata[block.type].append(coords_tuple)
        return primary_coords, dict(metadata), layout_result

    def run_on_image(self, np_img: np.ndarray, pdf_model: PDFModel, page_num: int, zoom: float = 2.0, selector: Optional[str] = None) -> DetectronDetectionResult:
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