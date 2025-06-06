from __future__ import annotations

from enum import Enum
from typing import Any, Union, Generator, Tuple
from tableai.readers.expectations import expects
import json
from json_repair import loads as json_repair_loads
import yaml
import fitz
import argparse
import requests
import os, io, tempfile, contextlib, json, yaml, boto3, urllib, urllib.parse, pathlib

try:
    from ocrmypdf import ocr, ExitCode
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

class _Source(Enum):
    LOCAL  = "local"
    S3     = "s3"
    HTTP   = "http"
    BYTES  = "bytes"

    @classmethod
    def from_uri(cls, obj: Union[str, bytes, bytearray]) -> "_Source":
        if isinstance(obj, (bytes, bytearray, io.BufferedIOBase)):
            return cls.BYTES
        if obj.startswith("s3://"):
            return cls.S3
        scheme = urllib.parse.urlparse(obj).scheme
        if scheme in {"http", "https"}:
            return cls.HTTP
        return cls.LOCAL

# ───────────────────────────────────────────────────────────────
# 2.  Path / temp-file resolver
# ───────────────────────────────────────────────────────────────

class _PathResolver:
    def __init__(self, session=None, s3_client=None):
        self._temp_paths: list[str] = []
        # Use provided s3_client, session, or create default
        if s3_client:
            self._s3 = s3_client
        elif session:
            self._s3 = session.client("s3")
        else:
            self._s3 = boto3.client("s3")

    def _tmp(self, suffix=".pdf") -> str:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        self._temp_paths.append(tf.name)
        tf.close()
        return tf.name

    # ↓ return a real filesystem path containing the bytes
    def resolve(self, obj: Union[str, bytes, bytearray]) -> Tuple[str, _Source]:
        src = _Source.from_uri(obj)

        if src == _Source.LOCAL:
            return os.fspath(obj), src

        if src == _Source.BYTES:
            path = self._tmp()
            with open(path, "wb") as f:
                f.write(obj)
            return path, src

        if src == _Source.HTTP:
            path = self._tmp()
            headers = {"User-Agent": "tableai-file-reader/0.1"}
            with requests.get(obj, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            return path, src

        if src == _Source.S3:
            path = self._tmp()
            _, _, bucket_key = obj.partition("s3://")
            bucket, key = bucket_key.split("/", 1)
            self._s3.download_file(bucket, key, path)
            return path, src

        raise ValueError(f"Unknown source {src}")

    # clean tempfiles at the end of the context
    def cleanup(self):
        for p in self._temp_paths:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(p)


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
    def handle_failure(cls, error, *args, file_path=None, session=None, s3_client=None, **kwargs):
        # Return a default empty dict
        return {}

class JsonYamlHandler:
    @classmethod
    @expects(type=[ExpectedJsonYaml], handler=DefaultDictHandler)
    def handle_failure(cls, error, file_path, session=None, s3_client=None, **kwargs):
        # Just attempt to load with json_repair. No try/except here.
        with open(file_path, 'r') as f:
            repaired = json_repair_loads(f)
        return repaired

class DefaultPdfFallbackHandler:
    @classmethod
    def handle_failure(cls, error, *args, file_path=None, session=None, s3_client=None, **kwargs):
        return error

class PdfHandler:
    @classmethod
    def handle_failure(cls, error, file_path, session=None, s3_client=None, **kwargs):
        custom_message = ''
        file_name = str(file_path)
        error_type = type(error).__name__
        error_message = str(error)
        error_base = {
            'ingestion_error': error_type,
            'error_message': error_message,
            'file_path': str(file_path)
        }

        if isinstance(error, PdfHasNoTextError):
            # Write OCR output to a temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = ocr(file_path, tmp_path, deskew=True, language='eng', force_ocr=True)
                if result == ExitCode.ok:
                    try:
                        doc = fitz.open(tmp_path)
                        page0 = doc[0]
                        if not page0.get_text().strip():
                            error_base['message'] = 'OCR completed, PDF still does not have text.'
                            return DefaultPdfFallbackHandler.handle_failure(error_base, file_path=file_path, session=session, s3_client=s3_client)
                        else:
                            return doc
                    finally:
                        os.remove(tmp_path)  # Always clean up
                else:
                    custom_message = 'OCR Repair Failed'
            except Exception as e:
                custom_message = f'OCR repair raised: {e}'
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        error_base['message'] = custom_message
        return DefaultPdfFallbackHandler.handle_failure(error_base, file_path=file_path, session=session, s3_client=s3_client)

# ───────────────────────────────────────────────────────────────
# 3.  FileReader public API
# ───────────────────────────────────────────────────────────────
class FileReader:
    """
    Smart loader for JSON, YAML, PDF.

    Accepts *any* of:
        • local path    str | pathlib.Path
        • "s3://bucket/key"
        • "http(s)://…""
        • bytes / bytearray
    """

    # -------------- JSON/YAML -----------------
    @staticmethod
    def _load_text(func, obj, session=None, s3_client=None):
        res = _PathResolver(session=session, s3_client=s3_client)
        path, _ = res.resolve(obj)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return func(fh)
        finally:
            res.cleanup()

    @staticmethod
    @expects(type=[ExpectedJsonYaml], handler=JsonYamlHandler)
    def json(obj, session=None, s3_client=None):
        return FileReader._load_text(json.load, obj, session=session, s3_client=s3_client)

    @staticmethod
    @expects(type=[ExpectedJsonYaml], handler=JsonYamlHandler)
    def yaml(obj, session=None, s3_client=None):
        return FileReader._load_text(yaml.safe_load, obj, session=session, s3_client=s3_client)

    # -------------- PDF -----------------
    @staticmethod
    @expects(type=[ExpectedPdf], handler=PdfHandler)
    def pdf(obj, session=None, s3_client=None):
        res = _PathResolver(session=session, s3_client=s3_client)
        path, src = res.resolve(obj)

        try:
            doc = fitz.open(path)
            if doc.page_count < 1:
                raise PdfHasNoPagesError(path)
            page0 = doc[0]
            if page0.rect.width <= 0 or page0.rect.height <= 0:
                raise PdfHasNoSizeError(path)
            if not page0.get_text().strip():
                raise PdfHasNoTextError(path)
            return doc
        finally:
            # keep on LOCAL source so caller can reopen; cleanup otherwise
            if src != _Source.LOCAL:
                res.cleanup()