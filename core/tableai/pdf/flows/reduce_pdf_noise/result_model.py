from tableai.pdf.flows.generic_results import FlowResult, FlowResultStage, section_field
from pydantic import ConfigDict, BaseModel, Field
from typing import Optional, Dict, Any

class NoiseDetectionResult(FlowResult):
    """
    Noise detection workflow result using the generic framework.
    """
    # === Overview Section ===
    overview: str = section_field(
        section="overview",
        default="""
# Overview -- Flow & Process Goal, and Expected Result Output:
Detects and removes recurring noise regions (headers, footers, processing statements) from PDF documents.
This sophisticated multi-stage pipeline combines geometric analysis, pattern recognition, and spatial 
reasoning to identify content that appears across multiple pages but isn't the main document content.
Examples include "CARD PROCESSING STATEMENT" headers, merchant numbers, page footers, etc.
After obtaining a list of noise regions for each virtual page, those are used to generate a list
of "content" regions, meaning probable regions with data or tables, and not header or footer information.
The resulting inverse list is used to render the original pdf with all data regions combined in
one single base64 image without the noise.
        """.strip(),
        description="High-level description of the noise detection process and its purpose"
    )
    
    goal: str = section_field(
        section="overview",
        default="""
# Primary Goal of Result
The goal and purpose of the new rendered image without noise is the following:
- Creates an image that can be used for inference models like YOLO or Detectron making it easier for those models to detect multi page spanning tables.
- Makes it easier for future multi modal vision inference to extract high quality and accurate table and data extractions.
- The result should only exclude data irrelevant to extracting tables from the pdfs, otherwise it is a bad process flow.
        """.strip(),
        description="Primary objectives and intended use cases for the noise-free result"
    )
    
    # === Configuration Section ===
    process_optional_parameters: str = section_field(
        section="configuration",
        default="""
# Should the result output from the process flow using the default parameters indicate a poor result, these 
parameters can be reconfigured to improve the process.
All possible parameter values that can be updated or refined to impact result output.
---

```python
noise_params = NoiseRegionParams(
    # === Core Geometric Detection Parameters ===
    HEADER_BOUND=75,                    # Y-coordinate boundary for header candidates (reduced from 100)
    FOOTER_BOUND=85,                    # Y-coordinate boundary for footer candidates (increased from 75)
    MIN_OCCURRENCES=2,                  # Minimum times a line must appear to be considered recurring (reduced from 3)
    
    # === Spatial Tolerance Parameters ===
    WHITESPACE_TOLERANCE=15,            # Tolerance for horizontal whitespace detection (increased from 10)
    TOUCHING_TOLERANCE=3.0,             # Tolerance for merging spatially adjacent regions (reduced from 5.0)
    
    # === FitzSearchIndex Query Configuration ===
    BLOCK_GROUP_QUERY_INCLUDED_VALUES=[  # Fields to include in the groupby transformation
        "bbox", 
        "path", 
        "normalized_text", 
        "bbox(rel)",
        "font_size",                    # Added: Include font size information
        "page"                          # Added: Include page number
    ],
    BLOCK_GROUP_QUERY_LABEL='Custom.Header.Footer.Analysis',  # Custom query label for tracking
    BLOCK_GROUPBY_KEYS=["block", "font_size"],  # Group by both block AND font size
    
    # === Universal Parameter ===
    query_label="CustomNoiseDetection_Run_2024"  # Universal label for all operations
)
```
        """.strip(),
        description="Optional parameters for result refinement and post-processing"
    )
    
    # === Statistics Section ===
    noise_regions_count: int = section_field(
        section="statistics", 
        default=0, 
        description="Number of noise regions detected"
    )
    
    content_regions_count: int = section_field(
        section="statistics", 
        default=0, 
        description="Number of content regions identified"
    )
    
    pages_analyzed: int = section_field(
        section="statistics", 
        default=0, 
        description="Total pages included in analysis"
    )
    
    # === Images Section ===
    result_image: Optional[str] = section_field(
        section="images",
        default=None,
        description="Base64-encoded final result image (annotated PDF with noise/content regions highlighted)"
    )
    
    original_image: Optional[str] = section_field(
        section="images",
        default=None, 
        description="Base64-encoded original PDF sample image for comparison"
    )
    
    annotated_image: Optional[str] = section_field(
        section="images",
        default=None, 
        description="Base64-encoded annotated PDF sample (limited pages)"
    )
    
    # === Results Section ===
    noise_regions: Optional[list] = section_field(
        section="results", 
        default=None, 
        description="Raw noise region bounding boxes"
    )
    
    content_regions: Optional[list] = section_field(
        section="results", 
        default=None, 
        description="Raw content region bounding boxes"
    )
    
    noise_regions_by_page: Optional[Dict[int, Any]] = section_field(
        section="metadata", 
        default=None, 
        description="Noise regions organized by page"
    )
    pdf_model: Optional[Any] = Field(default=None, exclude=False)
    
    # Enhanced display methods using PDFModel
    def show_result_with_highlights(self, **kwargs):
        """Display the result using PDFModel with noise/content region highlights."""
        
        # Create highlight boxes for noise and content regions
        highlight_boxes = {}
        
        if self.noise_regions:
            highlight_boxes["Noise Regions"] = {
                "boxes": self.noise_regions,
                "color": "red"
            }
        
        if self.content_regions:
            highlight_boxes["Content Regions"] = {
                "boxes": self.content_regions,
                "color": "blue"
            }
        
        # Use PDFModel's rich display functionality
        self.pdf_model.show(highlight_boxes=highlight_boxes, **kwargs)
    
    def show_crop_content_regions(self, **kwargs):
        """Display only the content regions as cropped images."""
        if not self.pdf_model or not self.content_regions:
            print("PDFModel or content regions not available")
            return
        
        self.pdf_model.show(crop_boxes=self.content_regions, **kwargs)

    @classmethod
    def load_from_stage(
        cls, 
        stage: 'FlowResultStage',
    ) -> "NoiseDetectionResult":
        """
        A factory method that creates a NoiseDetectionResult instance
        from a flexible FlowResultStage object and the final flow state.
        """
        if not isinstance(stage, FlowResultStage):
            raise TypeError("Input 'stage' must be a FlowResultStage object.")
            
        # Get all the data from the stage object.
        stage_data = stage.model_dump()
                
        # Create a new instance of this class, populating it with the
        # data from the stage. Pydantic handles the field mapping.
        instance = cls(**stage_data)
        
        # Inject the pdf_model from the stage for helper methods to work.
        if hasattr(stage, 'pdf_model'):
            instance.pdf_model = stage.pdf_model
            
        return instance