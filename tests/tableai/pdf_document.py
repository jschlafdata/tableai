"""
Test suite for PDFDocument functionality using synthetic data PDFs.
"""
import pytest
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# Assuming the following imports based on your structure
from tableai import PDFDocument, DocumentIdentity, Page
from tableai.exceptions import PdfProcessingError, PdfPathDoesNotExist, PdfHasNoPagesError, PdfHasNoSizeError


# Test fixtures
@pytest.fixture
def synthetic_pdf_dir():
    """Path to synthetic data PDFs directory."""
    return Path("tests/synthetic_data")


@pytest.fixture
def sample_pdf_path(synthetic_pdf_dir):
    """Get first available PDF from synthetic data directory."""
    pdf_files = list(synthetic_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found in synthetic_data directory")
    return pdf_files[0]


@pytest.fixture
def all_pdf_paths(synthetic_pdf_dir):
    """Get all PDFs from synthetic data directory."""
    pdf_files = list(synthetic_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found in synthetic_data directory")
    return pdf_files


@pytest.fixture
def sample_pdf_bytes(sample_pdf_path):
    """Load sample PDF as bytes."""
    with open(sample_pdf_path, 'rb') as f:
        return f.read()


# Test DocumentIdentity
class TestDocumentIdentity:
    def test_sha256_bytes(self):
        """Test SHA256 hash calculation."""
        test_bytes = b"test content"
        expected_hash = hashlib.sha256(test_bytes).hexdigest()
        assert DocumentIdentity.sha256_bytes(test_bytes) == expected_hash
    
    def test_from_doc_with_path(self, sample_pdf_path):
        """Test creating DocumentIdentity from document with path."""
        with open(sample_pdf_path, 'rb') as f:
            raw_bytes = f.read()
        
        identity = DocumentIdentity.from_doc(
            source_path=str(sample_pdf_path),
            raw_bytes=raw_bytes,
            source_type="local"
        )
        
        assert identity.source_path == str(sample_pdf_path)
        assert identity.source_type == "local"
        assert identity.id == DocumentIdentity.sha256_bytes(str(sample_pdf_path).encode("utf-8"))
        assert identity.content_hash == DocumentIdentity.sha256_bytes(raw_bytes)
    
    def test_from_doc_without_path(self, sample_pdf_bytes):
        """Test creating DocumentIdentity from bytes without path."""
        identity = DocumentIdentity.from_doc(
            source_path=None,
            raw_bytes=sample_pdf_bytes,
            source_type="bytes"
        )
        
        assert identity.source_path is None
        assert identity.source_type == "bytes"
        assert identity.content_hash == DocumentIdentity.sha256_bytes(sample_pdf_bytes)
        # ID should be generated from UUID when no path is provided
        assert identity.id is not None


# Test PDFDocument loading
class TestPDFDocumentLoading:
    def test_load_from_path(self, sample_pdf_path):
        """Test loading PDF from file path."""
        pdf_doc = PDFDocument.load(path=sample_pdf_path)
        
        assert pdf_doc is not None
        assert pdf_doc.identity.source_path == str(sample_pdf_path)
        assert pdf_doc.identity.source_type == "local"
        assert pdf_doc.page_count > 0
        assert pdf_doc.error is None
    
    def test_load_from_bytes(self, sample_pdf_bytes):
        """Test loading PDF from bytes."""
        pdf_doc = PDFDocument.load(data=sample_pdf_bytes)
        
        assert pdf_doc is not None
        assert pdf_doc.identity.source_path is None
        assert pdf_doc.identity.source_type == "bytes"
        assert pdf_doc.page_count > 0
        assert pdf_doc.error is None
    
    def test_load_invalid_path(self):
        """Test loading PDF with invalid path."""
        with pytest.raises(PdfPathDoesNotExist):
            PDFDocument.load(path="nonexistent/path.pdf")
    
    def test_load_no_input(self):
        """Test loading PDF without path or data."""
        with pytest.raises(ValueError, match="Provide either path or data"):
            PDFDocument.load()


# Test Page metadata
class TestPageMetadata:
    @pytest.mark.asyncio
    async def test_iter_pages(self, sample_pdf_path):
        """Test iterating through pages."""
        pdf_doc = PDFDocument.load(path=sample_pdf_path)
        
        pages = []
        async for page in pdf_doc.iter_pages():
            pages.append(page)
        
        assert len(pages) == pdf_doc.page_count
        
        # Check first page properties
        if pages:
            first_page = pages[0]
            assert isinstance(first_page, Page)
            assert first_page.number == 0
            assert first_page.width > 0
            assert first_page.height > 0
            assert hasattr(first_page, 'rotation')
    
    @pytest.mark.asyncio
    async def test_iter_page_metadata(self, sample_pdf_path):
        """Test iterating page metadata with limit."""
        pdf_doc = PDFDocument.load(path=sample_pdf_path)
        
        metadata_list = []
        async for metadata in pdf_doc.iter_page_metadata(limit=3):
            metadata_list.append(metadata)
        
        assert len(metadata_list) <= 3
        assert len(metadata_list) <= pdf_doc.page_count
        
        # Check metadata structure
        for meta in metadata_list:
            assert isinstance(meta, dict)
            assert 'page_number' in meta
            assert 'width' in meta
            assert 'height' in meta
            assert 'rotation' in meta
    
    @pytest.mark.asyncio
    async def test_head(self, sample_pdf_path):
        """Test getting first n pages."""
        pdf_doc = PDFDocument.load(path=sample_pdf_path)
        
        n = min(3, pdf_doc.page_count)
        head_pages = await pdf_doc.head(n=n)
        
        assert len(head_pages) == n
        for i, page in enumerate(head_pages):
            assert isinstance(page, Page)
            assert page.number == i


# Test profile function
class TestProfile:
    @pytest.mark.asyncio
    async def test_profile_structure(self, sample_pdf_path):
        """Test PDF profile structure and content."""
        pdf_doc = PDFDocument.load(path=sample_pdf_path)
        profile = await pdf_doc.profile(limit=5)
        
        # Check top-level structure
        assert isinstance(profile, dict)
        assert 'document' in profile
        assert 'pages' in profile
        
        # Check document metadata
        doc_meta = profile['document']
        assert isinstance(doc_meta, dict)
        assert 'format' in doc_meta
        assert 'raw' in doc_meta
        
        # Check pages metadata
        pages_meta = profile['pages']
        assert isinstance(pages_meta, list)
        assert len(pages_meta) <= 5
        assert len(pages_meta) <= pdf_doc.page_count
        
        # Validate each page metadata
        for page_meta in pages_meta:
            assert 'page_number' in page_meta
            assert 'width' in page_meta
            assert 'height' in page_meta
            assert page_meta['width'] > 0
            assert page_meta['height'] > 0
    
    @pytest.mark.asyncio
    async def test_profile_with_different_limits(self, sample_pdf_path):
        """Test profile with different limit values."""
        pdf_doc = PDFDocument.load(path=sample_pdf_path)
        
        # Test with limit=1
        profile_1 = await pdf_doc.profile(limit=1)
        assert len(profile_1['pages']) <= 1
        
        # Test with limit greater than page count
        profile_large = await pdf_doc.profile(limit=100)
        assert len(profile_large['pages']) == min(100, pdf_doc.page_count)


# Test map_pages functionality
class TestMapPages:
    @pytest.mark.asyncio
    async def test_map_pages_sync_function(self, sample_pdf_path):
        """Test mapping a synchronous function across pages."""
        pdf_doc = PDFDocument.load(path=sample_pdf_path)
        
        def get_page_dimensions(page: Page) -> Dict[str, float]:
            return {
                'number': page.number,
                'area': page.width * page.height
            }
        
        results = []
        async for result in pdf_doc.map_pages(get_page_dimensions):
            results.append(result)
        
        assert len(results) == pdf_doc.page_count
        for i, result in enumerate(results):
            assert result['number'] == i
            assert result['area'] > 0
    
    @pytest.mark.asyncio
    async def test_map_pages_async_function(self, sample_pdf_path):
        """Test mapping an asynchronous function across pages."""
        pdf_doc = PDFDocument.load(path=sample_pdf_path)
        
        async def async_get_page_info(page: Page) -> Dict[str, Any]:
            await asyncio.sleep(0.001)  # Simulate async operation
            return {
                'number': page.number,
                'is_landscape': page.width > page.height
            }
        
        results = []
        async for result in pdf_doc.map_pages(async_get_page_info, batch_size=2):
            results.append(result)
        
        assert len(results) == pdf_doc.page_count


# Test tee_pages functionality
class TestTeePages:
    @pytest.mark.asyncio
    async def test_tee_pages(self, sample_pdf_path):
        """Test splitting page iteration into multiple branches."""
        pdf_doc = PDFDocument.load(path=sample_pdf_path)
        
        # Create two independent iterators
        iter1, iter2 = await pdf_doc.tee_pages(count=2)
        
        # Consume from both iterators
        pages1 = []
        pages2 = []
        
        async for page in iter1:
            pages1.append(page.number)
        
        async for page in iter2:
            pages2.append(page.number)
        
        assert pages1 == pages2
        assert len(pages1) == pdf_doc.page_count


# Test multiple PDFs
class TestMultiplePDFs:
    @pytest.mark.asyncio
    async def test_all_synthetic_pdfs(self, all_pdf_paths):
        """Test loading and profiling all synthetic PDFs."""
        for pdf_path in all_pdf_paths:
            pdf_doc = PDFDocument.load(path=pdf_path)
            
            # Basic assertions
            assert pdf_doc.page_count > 0
            assert pdf_doc.error is None
            
            # Test profile
            profile = await pdf_doc.profile(limit=2)
            assert 'document' in profile
            assert 'pages' in profile
            
            # Test iteration
            page_count = 0
            async for page in pdf_doc.iter_pages():
                page_count += 1
                if page_count > 5:  # Limit iteration for performance
                    break
            
            assert page_count > 0


# Test error conditions
class TestErrorConditions:
    @pytest.mark.asyncio
    async def test_empty_pdf_handling(self, tmp_path):
        """Test handling of PDFs with no pages (if such a PDF can be created)."""
        # This test would require creating or having a PDF with no pages
        # which is an edge case that the code checks for
        pass  # Placeholder for when we have such test data
    
    @pytest.mark.asyncio
    async def test_zero_size_page_handling(self, tmp_path):
        """Test handling of PDFs with zero-size pages."""
        # This test would require a PDF with pages that have zero width/height
        pass  # Placeholder for when we have such test data


# Integration tests
class TestIntegration:
    @pytest.mark.asyncio
    async def test_complete_workflow(self, sample_pdf_path):
        """Test a complete workflow using multiple PDFDocument features."""
        # Load PDF
        pdf_doc = PDFDocument.load(path=sample_pdf_path)
        
        # Get profile
        profile = await pdf_doc.profile(limit=3)
        assert profile is not None
        
        # Get first few pages
        head_pages = await pdf_doc.head(n=2)
        assert len(head_pages) <= 2
        
        # Map a function across pages
        async def extract_dimensions(page: Page):
            return {'num': page.number, 'w': page.width, 'h': page.height}
        
        dimensions = []
        async for dim in pdf_doc.map_pages(extract_dimensions):
            dimensions.append(dim)
            if len(dimensions) >= 3:
                break
        
        assert len(dimensions) > 0
        
        # Verify consistency
        for i, dim in enumerate(dimensions):
            assert dim['num'] == i


# Performance tests (optional, can be marked as slow)
class TestPerformance:
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_pdf_iteration(self, all_pdf_paths):
        """Test performance with larger PDFs."""
        import time
        
        for pdf_path in all_pdf_paths:
            pdf_doc = PDFDocument.load(path=pdf_path)
            
            if pdf_doc.page_count < 10:
                continue  # Skip small PDFs for performance testing
            
            start_time = time.time()
            page_count = 0
            async for _ in pdf_doc.iter_pages():
                page_count += 1
            
            elapsed = time.time() - start_time
            pages_per_second = page_count / elapsed if elapsed > 0 else 0
            
            # Assert reasonable performance (adjust threshold as needed)
            assert pages_per_second > 10  # At least 10 pages per second


if __name__ == "__main__":
    # Run tests with asyncio
    pytest.main([__file__, "-v"])