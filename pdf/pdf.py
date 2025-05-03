from dataclasses import dataclass, asdict
import fitz
from data_loaders.ingest_files import FileReader
from typing import Optional, Union, List, Any
from pdf2image import convert_from_path
import numpy as np
import os
from PIL import Image


class Page:
    """Class representing a single page in a PDF"""
    
    def __init__(self, number: int,fitz_page: Any):
        self.number = number
        self.fitz_page = fitz_page
        self._coords = None
    
    @property
    def coords(self):
        x0,y0,x1,y1 = self.fitz_page.rect
        return XyXy(x0,y0,x1,y1, self.number)

@dataclass
class XyXy:
    """Class to handle coordinates in different formats"""
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int
    
    def scale(self, x_scale: float, y_scale: float) -> 'XyXy':
        """Scale coordinates by given factors"""
        return XyXy(
            x0=self.x0 * x_scale,
            y0=self.y0 * y_scale,
            x1=self.x1 * x_scale,
            y1=self.y1 * y_scale
        )
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def to_dict(self):
        return {
            self.page_num: {
            'page_box': [self.x0, self.y0, self.x1, self.y1],
            'height': self.height,
            'width': self.width
        }}


@dataclass
class Pdf:
    path: str
    stage: int
    PDF = Union[fitz.Document, None]
    _page_coords: Union[List, None] = None
    corrupt: bool = True
    width: int = 0
    height: int = 0

    def __post_init__(self):
        self.PDF = FileReader.pdf(self.path)
        self.width = self.PDF[0].rect.width
        self.height = self.PDF[0].rect.height
        if self.PDF:
            self.corrupt = False
    
    @property
    def count(self) -> int:
        """Number of pages in the PDF"""
        return len(self.PDF)
    
    @property
    def pages(self) -> List:
        """Access to all pages"""
        if self._page_coords is None:
            self._page_coords = [
                Page(number=i, fitz_page=self.PDF[i]).coords.to_dict()
                for i in range(self.count)
            ]
        return self._page_coords
    
    def __getitem__(self, idx: int) -> Page:
        """Allow indexing directly into the PDF to get pages"""
        return self.pages[idx]
    
    def __len__(self) -> int:
        """Allow using len() on the PDF object"""
        return self.count
    
    def close(self) -> None:
        """Close the fitz PDF document"""
        self.PDF.close()
    
    def to_dict(self):
        coords = [cd for cd in self.pages]
        return {
            'pages': self.count,
            'coords': coords,
        }
