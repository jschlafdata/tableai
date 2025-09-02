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

class PdfPathDoesNotExist(PdfProcessingError):
    def __init__(self, file_path: str):
        super().__init__(f"PDF path '{file_path}' does not exist.")
