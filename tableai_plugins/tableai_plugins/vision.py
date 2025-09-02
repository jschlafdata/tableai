import json
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

import numpy as np
import cv2

from ._yolo import YOLOTableDetector, YOLODetectionResult


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")

def _get_detector(dataset: str) -> YOLOTableDetector:
    return YOLOTableDetector()

@dataclass
class Detect:
    image_bytes: bytes
    pdf_bytes: bytes
    pdf_filename: str
    page_num: int
    dataset: str = "keremberke"
    zoom: float = 2.0
    model_overrides: Optional[Dict[str, Any]] = field(default_factory=dict)
    detector_factory: Callable[[str], YOLOTableDetector] = field(default=_get_detector, repr=False)

    async def detect_png(self) -> YOLODetectionResult:
        np_arr = np.frombuffer(self.image_bytes, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if np_img is None:
            raise HTTPException(status_code=400, detail="Could not decode the image bytes.")

        detector = self.detector_factory(self.dataset)
        return detector.run_on_image(
            np_img=np_img,
            page_num=self.page_num,
            zoom=self.zoom,
            model_name=self.dataset,
            model_overrides=self.model_overrides or {},
        )