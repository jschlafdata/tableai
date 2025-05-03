from enum import Enum
from typing import Any
from core.expectations import expects
import json
import json_repair as json_repair
import yaml
import fitz

class ValidJsonYaml(Enum):
    DICT = dict
    LIST = list

class ExpectedJsonYaml(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return any(isinstance(instance, t.value) for t in ValidJsonYaml)

class ExpectedJsonYaml(metaclass=ExpectedJsonYaml):
    pass

class DefaultDictHandler:
    @classmethod
    def handle_failure(cls, error, *args, file_path=None, **kwargs):
        # Return a default empty dict
        return {}

class JsonYamlHandler:
    @classmethod
    @expects(type=[ExpectedJsonYaml], handler=DefaultDictHandler)
    def handle_failure(cls, error, file_path):
        # Just attempt to load with json_repair. No try/except here.
        with open(file_path, 'r') as f:
            repaired = json_repair.load(f)
        return repaired

class ValidPdf(Enum):
    PDF = fitz.Document

# 2) Metaclass to check if an object is an instance of fitz.Document
class _ExpectedPdf(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return any(isinstance(instance, t.value) for t in ValidPdf)

# 3) The user-facing class that references the metaclass
class ExpectedPdf(metaclass=_ExpectedPdf):
    pass

# 4) A default handler if needed (for example, returning an empty dict on failure)
class DefaultPdfFallbackHandler:
    @classmethod
    def handle_failure(cls, error, *args, file_path=None, **kwargs):
        # Return a default "empty" response or raise an error as you prefer.
        # For consistency with your existing DefaultDictHandler, we return {}
        return {}

class PDFHandler:
    @classmethod
    @expects(type=[ExpectedPdf], handler=DefaultPdfFallbackHandler)
    def handle_failure(cls, error, file_path):
        """
        If the FileReader.pdf() method fails to produce a valid fitz.Document,
        we end up here. We’ll try again to open the PDF, then do basic
        validation (e.g., at least 1 page, page dimensions > 0).
        """
        doc = fitz.open(file_path)
        if doc.page_count < 1:
            # Return fallback or raise
            return {}

        for page in doc:
            if page.rect.width <= 0 or page.rect.height <= 0:
                return {}

        # If all checks pass, return the doc
        return doc

class FileReader:
    
    @staticmethod
    @expects(type=[ExpectedJsonYaml], handler=JsonYamlHandler)
    def json(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    @staticmethod
    @expects(type=[ExpectedJsonYaml], handler=JsonYamlHandler)
    def yaml(file_path):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    @staticmethod
    @expects(type=[ExpectedPdf], handler=PDFHandler)
    def pdf(file_path: str):
        """
        Primary PDF reading method. We open the PDF via fitz,
        validate that it has at least one page, etc.
        
        If this validation fails or we cannot open the document
        as a PDF, the `PDFHandler.handle_failure()` is invoked.
        """
        doc = fitz.open(file_path)

        # Basic validation
        if doc.page_count < 1:
            return None

        for page in doc:
            if page.rect.width <= 0 or page.rect.height <= 0:
                return None

        return doc