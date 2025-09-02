import json
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
import cv2

from ._yolo import YOLOTableDetector, YOLODetectionResult


class MockUploadFile:
    def __init__(self, content: bytes, filename: str, content_type: str):
        self._content = content
        self.filename = filename
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._content


class YOLODetectionResult:
    def __init__(self, data):
        self.data = data
    def __repr__(self):
        return f"YOLODetectionResult(data={self.data})"

class YOLOTableDetector:
    def run_on_image(self, **kwargs):
        print("YOLO Detector running with args:")
        for k, v in kwargs.items():
            # Don't print the huge image array
            if isinstance(v, np.ndarray):
                print(f"  {k}: np.ndarray(shape={v.shape})")
            else:
                print(f"  {k}: {v}")
        return YOLODetectionResult(data={"tables_found": 1, "confidence": 0.95})

def _get_detector(dataset: str) -> YOLOTableDetector:
    print(f"Getting detector for dataset: {dataset}")
    return YOLOTableDetector()

class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


@dataclass
class Detect:
    """
    A data class to hold all necessary information for a single table detection task.
    The detection logic is implemented as a method on this class.
    """
    # --- Fields: The data needed for detection ---
    image_bytes: bytes
    pdf_bytes: bytes
    pdf_filename: str
    page_num: int
    dataset: str = "keremberke"
    zoom: float = 2.0
    model_overrides: Optional[Dict[str, Any]] = field(default_factory=dict)

    async def detect_png(self) -> YOLODetectionResult:
        """
        Detect tables in the configured PNG image bytes.

        This method uses the data stored in the instance fields (self.image_bytes,
        self.page_num, etc.) to perform the detection.
        """
        # The method now accesses its configuration via `self`
        overrides_dict = self.model_overrides or {}

        # Decode the image from the instance's byte data
        np_arr = np.frombuffer(self.image_bytes, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if np_img is None:
            raise HTTPException(status_code=400, detail="Could not decode the image bytes.")

        # Get the detector instance based on the instance's dataset
        detector = _get_detector(self.dataset)

        # Run detection using data from the instance
        result = detector.run_on_image(
            np_img=np_img,
            page_num=self.page_num,
            zoom=self.zoom,
            model_name=self.dataset,
            model_overrides=overrides_dict
        )
        return result
