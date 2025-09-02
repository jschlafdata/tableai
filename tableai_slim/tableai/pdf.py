from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Any, AsyncIterator, Callable
import hashlib
import uuid
import fitz
from pathlib import Path

from ._exceptions import PdfHasNoPagesError
from ._exceptions import PdfHasNoSizeError
from ._exceptions import PdfPathDoesNotExist
from ._exceptions import PdfProcessingError
from ._exceptions import PdfHasNoTextError

from ._async_iter import aislice
from ._async_iter import amap
from ._async_iter import atee

from ._utils import ext_page_metadata
from ._utils import ext_pdf_metadata

import dateparser


@dataclass(frozen=True)
class DocumentIdentity:
    source_path: Optional[str]
    id: str
    content_hash: str
    source_type: str = "local"

    @staticmethod
    def sha256_bytes(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    @classmethod
    def from_doc(cls, source_path: Optional[str], raw_bytes: bytes, source_type="local"):
        sid = cls.sha256_bytes((source_path or str(uuid.uuid4())).encode("utf-8"))
        return cls(source_path=source_path, id=sid, content_hash=cls.sha256_bytes(raw_bytes), source_type=source_type)

@dataclass
class Page:
    number: int
    width: float
    height: float
    rotation: int = 0
    _fitz_page: Optional[Any] = field(default=None, repr=False)

@dataclass
class PDFDocument:
    identity: DocumentIdentity
    error: Optional[PdfProcessingError] = None
    _fitz_doc: fitz.Document = field(default=None, repr=False)

    @property
    def page_count(self) -> int:
        return self._fitz_doc.page_count
    
    async def iter_page_metadata(self, limit: int = 5) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield page-specific metadata for at most `limit` pages (front of document).
        Keeps the pass cheap so other steps (classify, etc.) can start early.
        """
        count = 0
        async for pg in self.iter_pages():
            if count >= limit:
                break
            yield ext_page_metadata(pg)
            count += 1

    async def profile(self, limit: int = 5) -> Dict[str, Any]:
        """
        Convenience: collect document-level metadata once + the first `limit` page metas.
        """
        doc_meta = ext_pdf_metadata(self._fitz_doc)  # your existing function (fitz.Document -> dict)
        pages = []
        async for pm in self.iter_page_metadata(limit=limit):
            pages.append(pm)
        return {"document": doc_meta, "pages": pages}
        
    def iter_pages(self) -> AsyncIterator[Page]:
        """Fresh async stream of Page objects (lazy)."""
        async def gen():
            for i in range(self._fitz_doc.page_count):
                p = self._fitz_doc[i]
                yield Page(
                    number=i,
                    width=float(p.rect.width),
                    height=float(p.rect.height),
                    rotation=int(getattr(p, "rotation", 0)),
                    _fitz_page=p,
                )
        return gen()

    async def head(self, n: int = 5) -> List[Page]:
        """Collect first n pages (async islice)."""
        pages = []
        async for pg in aislice(self.iter_pages(), 0, n):
            pages.append(pg)
        return pages

    async def map_pages(self, fn: Callable[[Page], Any], *, batch_size: int = 0) -> AsyncIterator[Any]:
        """
        Map a function (sync or async) across pages.
        Use batch_size > 0 for parallel async funcs.
        """
        async for result in amap(fn, self.iter_pages(), batch_size=batch_size):
            yield result

    async def tee_pages(self, count: int = 2, *, maxsize: int = 0):
        """Split a single pass of pages into `count` independent branches."""
        return await atee(self.iter_pages(), count=count, maxsize=maxsize)

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None, data: Optional[bytes] = None) -> "PDFDocument":
        error = None
        if path is None and data is None:
            raise ValueError("Provide either path or data")

        if path is not None:
            p = Path(path)
            if not p.exists():
                raise PdfPathDoesNotExist(str(p))
            doc = fitz.open(p)
            raw = doc.tobytes(garbage=4, clean=True)
            ident = DocumentIdentity.from_doc(str(p), raw, source_type="local")
        else:
            doc = fitz.open(stream=data, filetype="pdf")
            raw = data or doc.tobytes(garbage=4, clean=True)
            ident = DocumentIdentity.from_doc(None, raw, source_type="bytes")

        if doc.page_count == 0:
            error = PdfHasNoPagesError(str(path) if path else "<bytes>")
        first = doc[0]
        if first.rect.width == 0 or first.rect.height == 0:
            error = PdfHasNoSizeError(str(path) if path else "<bytes>")

        return cls(identity=ident, _fitz_doc=doc, error=error)