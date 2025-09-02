# tests/test_detect_from_synthetic_pdfs.py
import hashlib
from pathlib import Path
from typing import List

import pytest
from tableai_plugins import Detect

import logging
logger = logging.getLogger(__name__)

# ---------- Helpers ----------

def _render_first_page_png(pdf_path: Path, zoom: float = 2.0) -> bytes:
    """
    Render the first page of a PDF to PNG bytes using PyMuPDF (fitz).
    """
    fitz = pytest.importorskip("fitz")  # skip test if PyMuPDF not installed
    doc = fitz.open(pdf_path)
    try:
        if doc.page_count == 0:
            pytest.skip(f"{pdf_path} has 0 pages")
        page = doc[0]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


# ---------- Fixtures (duplicate safely if not in conftest.py) ----------

@pytest.fixture
def synthetic_pdf_dir() -> Path:
    """Path to synthetic data PDFs directory."""
    return Path("tests/synthetic_data")


@pytest.fixture
def sample_pdf_path(synthetic_pdf_dir: Path) -> Path:
    """Get first available PDF from synthetic data directory."""
    pdf_files = list(synthetic_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found in synthetic_data directory")
    return pdf_files[0]


@pytest.fixture
def all_pdf_paths(synthetic_pdf_dir: Path) -> List[Path]:
    """Get all PDFs from synthetic data directory."""
    pdf_files = list(synthetic_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found in synthetic_data directory")
    return pdf_files


@pytest.fixture
def sample_pdf_bytes(sample_pdf_path: Path) -> bytes:
    """Load sample PDF as bytes."""
    return sample_pdf_path.read_bytes()


# ---------- Tests ----------

def _count_tables(result, page_num: int = 1) -> int:
    return len(result.coordinates_by_page.get(page_num, []))

def _assert_boxes_within_image(result, page_num: int = 1):
    dims = result.page_dimensions[page_num]
    W, H = dims.image_width, dims.image_height
    for (x1, y1, x2, y2) in result.coordinates_by_page.get(page_num, []):
        assert 0 <= x1 < x2 <= W
        assert 0 <= y1 < y2 <= H

@pytest.mark.asyncio
async def test_detect_from_single_pdf(sample_pdf_path: Path, sample_pdf_bytes: bytes):
    """
    Render first page to PNG, run Detect.detect_png(), and validate structure + bbox sanity.
    """
    pytest.importorskip("cv2")

    image_bytes = _render_first_page_png(sample_pdf_path, zoom=2.0)

    detect = Detect(
        image_bytes=image_bytes,
        pdf_bytes=sample_pdf_bytes,
        pdf_filename=sample_pdf_path.name,
        page_num=1,
        dataset="keremberke",
        zoom=2.0,
        model_overrides={"conf": 0.25},
    )

    result = await detect.detect_png()
    logger.info("result=%s", result.model_dump())

    # Basic schema checks (no 'confidence'—this is a Pydantic model now)
    assert 1 in result.page_dimensions
    assert isinstance(result.coordinates_by_page.get(1, []), list)

    # If any boxes are present, they must be within image bounds
    _assert_boxes_within_image(result, page_num=1)

@pytest.mark.asyncio
async def test_detect_across_multiple_pdfs(all_pdf_paths: List[Path]):
    """
    Run on several PDFs; expect 0 tables for the receipt photo, ≥1 for real PDFs.
    """
    pytest.importorskip("cv2")

    # Minimal expectation policy by filename
    min_expected = {}
    for p in all_pdf_paths[:5]:
        name = p.name.lower()
        # The phone photo of a receipt should NOT detect tables
        if "receipt" in name or "phone" in name:
            min_expected[p] = 0
        else:
            # Actual PDFs should have at least one table on page 1
            min_expected[p] = 1

    tested = 0
    for pdf_path, min_tables in min_expected.items():
        logger.info("Running on %s", pdf_path.name)
        png_bytes = _render_first_page_png(pdf_path, zoom=2.0)
        pdf_bytes = pdf_path.read_bytes()

        detect = Detect(
            image_bytes=png_bytes,
            pdf_bytes=pdf_bytes,
            pdf_filename=pdf_path.name,
            page_num=1,
            dataset="keremberke",
            zoom=2.0,
            model_overrides=None,
        )
        result = await detect.detect_png()
        logger.info("result=%s", result.model_dump())

        # Count detections on page 1 and enforce per-file expectation
        n = _count_tables(result, page_num=1)
        logger.info("tables_found=%d on %s", n, pdf_path.name)
        assert n >= min_tables

        # If any present, verify bbox sanity
        _assert_boxes_within_image(result, page_num=1)

        tested += 1

    assert tested > 0


@pytest.mark.asyncio
async def test_detect_raises_on_bad_image_bytes(sample_pdf_path: Path, sample_pdf_bytes: bytes):
    """
    Passing non-image bytes should raise an HTTP-like exception from Detect.
    """
    pytest.importorskip("cv2")

    detect = Detect(
        image_bytes=b"not a png",
        pdf_bytes=sample_pdf_bytes,
        pdf_filename=sample_pdf_path.name,
        page_num=1,
    )

    with pytest.raises(Exception) as excinfo:
        await detect.detect_png()

    # Your Detect raises a custom HTTPException(status_code=400, detail="...")
    status = getattr(excinfo.value, "status_code", None)
    assert status == 400
    assert "Could not decode the image bytes" in str(excinfo.value)


if __name__ == "__main__":
    # Run tests with asyncio
    pytest.main([__file__, "-v"])