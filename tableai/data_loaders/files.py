from enum import Enum
from typing import Any, List, Optional, Dict
from tableai.expectations.base import expects
import json
import json_repair as json_repair
import yaml
import fitz
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlmodel import SQLModel, Field, create_engine, Session, select, text
from datetime import datetime
from pathlib import Path
import os
import shutil
from ocrmypdf import ocr, ExitCode

class PdfProcessingError(Exception):
    """Base class for PDF-related errors."""

class PdfHasNoPagesError(PdfProcessingError):
    def __init__(self, file_path: str):
        super().__init__(f"PDF '{file_path}' has no pages.")

class PdfHasNoSizeError(PdfProcessingError):
    def __init__(self, file_path: str):
        super().__init__(f"PDF '{file_path}' has a page with no width or height.")

class PdfHasNoTextError(PdfProcessingError):
    def __init__(self, file_path: str):
        super().__init__(f"PDF '{file_path}' has a page with no text.")

class ValidJsonYaml(Enum):
    DICT = dict
    LIST = list

class ExpectedJsonYaml(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return any(isinstance(instance, t.value) for t in ValidJsonYaml)

class ExpectedJsonYaml(metaclass=ExpectedJsonYaml):
    pass

class ValidPdf(Enum):
    PDF = fitz.Document

# 2) Metaclass to check if an object is an instance of fitz.Document
class _ExpectedPdf(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return any(isinstance(instance, t.value) for t in ValidPdf)

# 3) The user-facing class that references the metaclass
class ExpectedPdf(metaclass=_ExpectedPdf):
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


# 4) A default handler if needed (for example, returning an empty dict on failure)
class DefaultPdfFallbackHandler:
    @classmethod
    def handle_failure(cls, error, *args, file_path=None, **kwargs):
        # # Return a default "empty" response or raise an error as you prefer.
        # # For consistency with your existing DefaultDictHandler, we return {}
        # print('DEFAULT HANDLER ENGAGED!!')
        print(f"HANDLED PATH: {error}")
        return error


class PdfHandler:
    @classmethod
    def handle_failure(cls, error, file_path, recovery_path):
        custom_message=''
        file_name = str(file_path)
        file_id = file_name.removesuffix('.pdf')

        error_type = type(error).__name__
        error_message = str(error)
        print(f"HANDLER: recovery_path: {recovery_path}")

        error_base = {
            'ingestion_error': str(error_type), 
            'error_message': str(error_message),
            'file_path': str(file_path)
        }

        print(error_base)

        if isinstance(error, PdfHasNoTextError): 
            result = ocr(file_path, recovery_path, deskew=True, language='eng',force_ocr=True)
            if result == ExitCode.ok:
                ocr_repair = True 
                doc = fitz.open(recovery_path)
                page0 = doc[0]
                if not page0.get_text().strip():
                    error_base['message'] = 'OCR Completed, PDF Still does not have text.'
                    return DefaultPdfFallbackHandler.handle_failure(error_base, file_path=file_path)
                else:
                    return doc
            else:
                custom_message = 'OCR Repair Failed'
        error_base['message'] = custom_message
        return DefaultPdfFallbackHandler.handle_failure(error_base, file_path=file_path)

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
    @expects(type=[ExpectedPdf], handler=PdfHandler)
    def pdf(file_path: str, recovery_path: str = None, skip_recovery=False):
        doc = fitz.open(file_path)
        
        if doc.page_count < 1:
            raise PdfHasNoPagesError(file_path)
        
        page0 = doc[0]
        if page0.rect.width <= 0 or page0.rect.height <= 0:
            raise PdfHasNoSizeError(file_path)

        if not page0.get_text().strip():
            raise PdfHasNoTextError(file_path)

        return doc
