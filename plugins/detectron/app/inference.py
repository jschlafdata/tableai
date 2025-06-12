import torch
import detectron2
import layoutparser as lp

from collections import defaultdict
from tableai.pdf.models import PDFModel
from typing import Dict, List, Tuple, Optional

class DetectronTableDetector:
    """
    Standalone Detectron2‐based layout detector (table or otherwise) that
    leverages PDFModel for rendering and coordinate transforms. You can choose
    among four pre-defined datasets: PubLayNet, TableBank, HJDataset, or PrimaLayout.
    Non‐table regions are returned under "metadata" per page.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 1.  DEFINE SUPPORTED DATASETS, THEIR LP URIs, AND LABEL MAPS
    # ──────────────────────────────────────────────────────────────────────────
    _DATASET_CONFIGS = {
        "PubLayNet": {
            "lp_uri":    "lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config",
            "label_map": {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        },
        "TableBank": {
            "lp_uri":    "lp://TableBank/faster_rcnn_R_50_FPN_3x/config",
            "label_map": {0: "Table"},
        },
        "HJDataset": {
            "lp_uri":    "lp://HJDataset/retinanet_R_50_FPN_3x/config",
            "label_map": {
                1: "Page Frame",
                2: "Row",
                3: "Title Region",
                4: "Text Region",
                5: "Title",
                6: "Subtitle",
                7: "Other",
            },
        },
        "PrimaLayout": {
            "lp_uri":    "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config",
            "label_map": {
                1: "TextRegion",
                2: "ImageRegion",
                3: "TableRegion",
                4: "MathsRegion",
                5: "SeparatorRegion",
                6: "OtherRegion",
            },
        },
    }

    # ──────────────────────────────────────────────────────────────────────────
    # 2.  INITIALIZER
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        dataset: str = "PubLayNet",
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            dataset: One of "PubLayNet", "TableBank", "HJDataset", or "PrimaLayout".
                     Defaults to "PubLayNet" if not specified.
            confidence_threshold: Minimum score for Detectron2 predictions.
        """
        if dataset not in self._DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Must be one of: "
                f"{list(self._DATASET_CONFIGS.keys())}"
            )
        self.dataset = dataset
        self.confidence_threshold = confidence_threshold

        cfg = self._DATASET_CONFIGS[dataset]
        self.lp_uri = cfg["lp_uri"]
        self.label_map = cfg["label_map"]

        self.model: Optional[lp.models.Detectron2LayoutModel] = None
        self._load_model()

    # ──────────────────────────────────────────────────────────────────────────
    # 3.  MODEL LOADER
    # ──────────────────────────────────────────────────────────────────────────
    def _load_model(self):
        """
        Initialize (or reinitialize) the LayoutParser Detectron2 model
        with the current `lp_uri` and `confidence_threshold`.
        """
        self.model = lp.models.Detectron2LayoutModel(
            self.lp_uri,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.confidence_threshold],
            label_map=self.label_map,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 4.  INTERNAL: EXTRACT BLOCKS OF A GIVEN TYPE
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _detected_blocks_of_type(
        model_result: lp.Layout, selector: str
    ) -> List[Tuple[float, float, float, float]]:
        """
        From a LayoutParser `Layout`, pick only those blocks whose `.type`
        equals `selector`. Return a list of bounding‐box tuples (x1, y1, x2, y2)
        in pixel coordinates.

        Args:
            model_result: LayoutParser Layout containing all detected blocks.
            selector: The block `.type` to filter by (e.g. "Table" or "TextRegion" etc.).

        Returns:
            A list of (x1, y1, x2, y2) tuples in image (pixel) coords.
        """
        coords = []
        for block in model_result:
            if block.type == selector:
                x1, y1, x2, y2 = block.coordinates
                coords.append((x1, y1, x2, y2))
        return coords

    # ──────────────────────────────────────────────────────────────────────────
    # 5.  MAIN: RUN DETECTION PAGE BY PAGE, CAPTUREing BOTH TABLES AND METADATA
    # ──────────────────────────────────────────────────────────────────────────
    def get_detectron_coords(
        self,
        pdf_model: PDFModel,
        zoom: float = 2.0,
        confidence_threshold: Optional[float] = None,
        selector: Optional[str] = None,
    ) -> None:
        """
        Run Detectron2 inference on each page of `pdf_model`, transform pixel
        boxes to PDF‐page boxes, then into combined‐PDF coords.  
        Tables (or selected blocks) go into `self.tbl_coordinates`; all other
        block types go into `self.metadata`.

        Populates:
          - self.model_results: Dict[page_number, LayoutParser Layout object]
          - self.tbl_coordinates: Dict[page_number, List[(x1,y1,x2,y2)] in combined‐PDF]
          - self.metadata: Dict[page_number, Dict[label, List[(x1,y1,x2,y2)] ]]
          - self.page_dimensions: Dict[page_number, {"image_width": int, "image_height": int}]
          - self.table_bboxes: flattened List of all combined–PDF table bounding boxes

        Args:
            pdf_model: Instance of PDFModel (pre‐loaded).
            zoom: Zoom factor for render.
            confidence_threshold: If given, overrides the stored threshold.
            selector: If provided, use this block‐type instead of the dataset’s default.
        """
        # Override threshold if needed
        if confidence_threshold is not None and confidence_threshold != self.confidence_threshold:
            self.confidence_threshold = confidence_threshold
            self._load_model()

        # Decide default selector for “table” or primary block
        default_selector = None
        if self.dataset == "PubLayNet":
            default_selector = "Table"
        elif self.dataset == "TableBank":
            default_selector = "Table"
        elif self.dataset == "HJDataset":
            default_selector = "Row"            # no "Table" label in HJDataset
        elif self.dataset == "PrimaLayout":
            default_selector = "TableRegion"

        sel = selector or default_selector

        self.model_results: Dict[int, lp.Layout] = {}
        self.tbl_coordinates: Dict[int, List[Tuple[float, float, float, float]]] = {}
        self.metadata: Dict[int, Dict[str, List[Tuple[float, float, float, float]]]] = {}
        self.page_dimensions: Dict[int, Dict[str, int]] = {}
        self.table_bboxes: List[Tuple[float, float, float, float]] = []

        for page_number, _ in enumerate(pdf_model.doc):
            # a) Render page via PDFModel → (_, np_img, img_w, img_h)
            _, np_img, img_w, img_h = pdf_model.page_to_numpy(page_number, zoom=zoom)
            self.page_dimensions[page_number] = {"image_width": img_w, "image_height": img_h}

            # b) Run Detectron2 model via LayoutParser
            layout_result = self.model.detect(np_img)
            self.model_results[page_number] = layout_result

            # c) Extract ALL block coordinates by type
            per_label_coords: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
            for block in layout_result:
                lbl = block.type
                x1, y1, x2, y2 = block.coordinates
                per_label_coords[lbl].append((x1, y1, x2, y2))

            # d) Separate “tables” vs “metadata”
            table_px = per_label_coords.pop(sel, [])
            combined_tables: List[Tuple[float, float, float, float]] = []
            for x1_px, y1_px, x2_px, y2_px in table_px:
                # pixel → PDF‐page coords
                page_bbox = pdf_model.img_bbox_to_page_bbox(
                    (x1_px, y1_px, x2_px, y2_px),
                    (img_w, img_h),
                    page_number,
                    zoom=zoom,
                )
                # PDF‐page coords → combined‐PDF coords
                combined_bbox = pdf_model.page_bbox_to_combined(page_number, page_bbox)
                tup = tuple(map(float, combined_bbox))
                combined_tables.append(tup)
                self.table_bboxes.append(tup)

            self.tbl_coordinates[page_number] = combined_tables

            # e) Convert all non‐table blocks into combined‐PDF coords and store under metadata
            combined_metadata: Dict[str, List[Tuple[float, float, float, float]]] = {}
            for lbl, coords_px in per_label_coords.items():
                combined_list = []
                for x1_px, y1_px, x2_px, y2_px in coords_px:
                    page_bbox = pdf_model.img_bbox_to_page_bbox(
                        (x1_px, y1_px, x2_px, y2_px),
                        (img_w, img_h),
                        page_number,
                        zoom=zoom,
                    )
                    combined_bbox = pdf_model.page_bbox_to_combined(page_number, page_bbox)
                    combined_list.append(tuple(map(float, combined_bbox)))
                combined_metadata[lbl] = combined_list

            self.metadata[page_number] = combined_metadata

    # ──────────────────────────────────────────────────────────────────────────
    # 6.  POST‐PROCESS (TABLES ONLY, WRAP IN DICT)
    # ──────────────────────────────────────────────────────────────────────────
    def process_detectron_coords(
        self, pdf_model: PDFModel, margins: Optional[Dict[str, float]] = None
    ) -> Dict[int, Dict]:
        """
        Take the raw table bounding boxes (self.tbl_coordinates) and create a nested dict:
          page_number → {
            "tables": {
                table_index: {
                    "rect": (x1,y1,x2,y2),
                    "confidence": 1.0,
                    "model_results": lp.Layout,
                    "image_dimensions": {"width": w, "height": h},
                },
                ...
            },
            "totals": {},       # reserved for future
          }
        (Non‐table regions are available in self.metadata, not duplicated here.)
        """
        if margins is None:
            margins = {}

        table_coordinates: Dict[int, Dict] = defaultdict(lambda: {"tables": {}, "totals": {}})

        for page_number in self.tbl_coordinates:
            coords = self.tbl_coordinates[page_number]
            layout_obj = self.model_results[page_number]
            img_w = self.page_dimensions[page_number]["image_width"]
            img_h = self.page_dimensions[page_number]["image_height"]

            for idx, bbox in enumerate(coords):
                table_coordinates[page_number]["tables"][idx] = {
                    "rect": bbox,
                    "confidence": 1.0,
                    "model_results": layout_obj,
                    "image_dimensions": {"width": img_w, "height": img_h},
                }

        return dict(table_coordinates)

    # ──────────────────────────────────────────────────────────────────────────
    # 7.  RUN (ORCHESTRATE EVERYTHING AND RETURN METADATA)
    # ──────────────────────────────────────────────────────────────────────────
    def run(
        self,
        pdf_model: PDFModel,
        zoom: float = 2.0,
        confidence_threshold: Optional[float] = None,
        margins: Optional[Dict[str, float]] = None,
        selector: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Execute Detectron2 detection and return a dictionary containing:
          - "tbl_coordinates": { page_number: [ (x1,y1,x2,y2), … ] }
          - "table_bboxes":   [ (x1,y1,x2,y2), … ]   # flattened
          - "metadata":       { page_number: { label: [ (x1,y1,x2,y2), … ], … } }
          - "processed_coordinates": <dict from process_detectron_coords>
          - "model_results":  { page_number: LayoutParser Layout }
          - "page_dimensions":{ page_number: { "image_width": w, "image_height": h } }

        Args:
            pdf_model: PDFModel (must already be loaded with doc/index).
            zoom: Zoom factor for rendering pages.
            confidence_threshold: If provided, override the model’s threshold.
            margins: Optional margins for downstream logic (not used here).
            selector: If provided, override the default block type (“Table” or dataset default).

        Returns:
            A dict containing raw + processed bounding boxes, metadata, etc.
        """
        # 1) Detect bounding boxes (fills self.tbl_coordinates & self.metadata)
        self.get_detectron_coords(
            pdf_model,
            zoom=zoom,
            confidence_threshold=confidence_threshold,
            selector=selector,
        )

        # 2) Process only tables into a structured dict
        processed = self.process_detectron_coords(pdf_model, margins=margins)

        return {
            "tbl_coordinates": self.tbl_coordinates,
            "table_bboxes": self.table_bboxes,
            "metadata": self.metadata,
            "page_dimensions": self.page_dimensions,
        }