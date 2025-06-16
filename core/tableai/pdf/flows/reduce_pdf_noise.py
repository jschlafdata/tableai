from datetime import datetime
from collections import defaultdict
from tableai.pdf.coordinates import (
    Geometry,
    CoordinateMapping
)
from pydantic import BaseModel, Field 
from typing import Optional
from tableai.pdf.trace import TraceLog, TraceableWorkflow
from tableai.pdf.models import PDFModel
from tableai.pdf.generic_models import (
    GenericFunctionParams,
    GroupOps
)
from tableai.pdf.query import (
    TextNormalizer, 
    WhitespaceGenerator, 
    groupby,
    regroup_by_key,
    QueryParams
)

from tableai.pdf.query_funcs import (
    horizontal_whitespace,
    group_vertically_touching_bboxes, 
    HorizontalWhitespaceParams, 
    GroupTouchingBoxesParams
)

from tableai.pdf.flows.generic import (
    FlowResultBase, 
    section_field
)

from typing import List, Tuple, Any, Optional, Tuple, Dict, Union, Callable

HeaderFooterParams = GenericFunctionParams.create_custom_model(
    "HeaderFooterParams",
    custom_fields={
        'HEADER_BOUND': {
            'type': int,
            'default': 75,
            'description': "Y-coordinate boundary for the top of a page. Anything with y0_rel < this value is a header candidate."
        },
        'FOOTER_BOUND': {
            'type': int,
            'default': 75,
            'description': "Boundary from the bottom of a page. Anything with y1_rel > (page_height - this value) is a footer candidate."
        },
        'MIN_OCCURRENCES': {
            'type': int,
            'default': 2,
            'description': "The minimum number of times a line must appear across all pages to be confirmed as a header/footer."
        },
        'BLOCK_GROUP_QUERY_INCLUDED_VALUES': { 'type': list, 'default': ["bbox", "path", "normalized_text", "bbox(rel)"], 'description': "Fields to return from the LineIndex groupby transformation." },
        'BLOCK_GROUP_QUERY_LABEL': { 'type': str, 'default': 'Header.Footer.Blocks', 'description': "Query label for the initial base line index query returning blocks grouped by line." },
        'BLOCK_GROUPBY_KEYS': { 'type': list, 'default': ["block"], 'description': "Keys passed to the groupby transform function." },
    }
)

# For group_vertically_touching_bboxes
RecurringLinesParams = GenericFunctionParams.create_custom_model(
    "RecurringLinesParams", {
        'MIN_OCCURRENCES': { 'type': int, 'default': 2, 'description': "How many times a line must be seen to be considered recurring." }
    }
)

# For the high-level find_combined_noise_regions orchestrator
NoiseRegionParams = GenericFunctionParams.create_custom_model(
    "NoiseRegionParams", {
        'HEADER_BOUND': { 'type': int, 'default': 75, 'description': "Y-coord boundary for header candidates." },
        'FOOTER_BOUND': { 'type': int, 'default': 75, 'description': "Y-coord boundary for footer candidates." },
        'MIN_OCCURRENCES': { 'type': int, 'default': 2, 'description': "Min times a line must appear to be 'recurring'." },
        'WHITESPACE_TOLERANCE': { 'type': int, 'default': 10, 'description': "Tolerance passed to horizontal_whitespace." },
        'TOUCHING_TOLERANCE': { 'type': float, 'default': 5.0, 'description': "Tolerance for merging noise regions." },
        'BLOCK_GROUP_QUERY_INCLUDED_VALUES': { 'type': list, 'default': ["bbox", "path", "normalized_text", "bbox(rel)"], 'description': "Fields to return from the LineIndex groupby transformation." },
        'BLOCK_GROUP_QUERY_LABEL': { 'type': str, 'default': 'Header.Footer.Blocks', 'description': "Query label for the initial base line index query returning blocks grouped by line." },
        'BLOCK_GROUPBY_KEYS': { 'type': list, 'default': ["block"], 'description': "Keys passed to the groupby transform function." },
        'TEXT_NORMALIZER_PATTERNS': { 
            'type': dict, 
            'default': {
                r'page\s*\d+\s*of\s*\d+': 'page xx of xx',
                r'page\s*\d+': 'page xx',
                r'\b\d{4}[-/]\d{2}[-/]\d{2}\b': 'YYYY-MM-DD',  # Normalize dates
                r'\$[\d,]+\.\d{2}': '$X.XX'  # Normalize currency
            }, 
            'description': "Regex patterns for text normalization. Keys are patterns, values are replacements." 
        },
        'TEXT_NORMALIZER_DESCRIPTION': { 
            'type': str, 
            'default': 'Normalizes text using a set of regex substitutions. This is used in the index to create index items with the key=[]', 
            'description': "Description for the TextNormalizer component." 
        },
        'WHITESPACE_MIN_GAP': { 
            'type': float, 
            'default': 5.0, 
            'description': "Minimum gap in pixels to consider as significant whitespace region." 
        },
        'GENERATE_IMAGES': { 
            'type': bool, 
            'default': True, 
            'description': "Whether to generate base64 sample images for original and annotated PDFs." 
        },
        'IMAGE_ZOOM': { 
            'type': float, 
            'default': 1.0, 
            'description': "Zoom factor for generated sample images." 
        },
        'IMAGE_SPACING': { 
            'type': int, 
            'default': 5, 
            'description': "Spacing in pixels between cropped regions in sample images." 
        },
        'IMAGE_PAGE_LIMIT': { 
            'type': int, 
            'default': 1, 
            'description': "Maximum virtual page number to include in sample images (0-based, so 1 = first 2 pages)." 
        },
        'IMAGE_BOX_WIDTH': { 
            'type': int, 
            'default': 4, 
            'description': "Width of highlight box outlines in sample images." 
        },
        'IMAGE_FONT_SIZE': { 
            'type': int, 
            'default': 16, 
            'description': "Font size for highlight box labels in sample images." 
        },
        'NOISE_BOX_COLOR': { 
            'type': str, 
            'default': 'red', 
            'description': "Color for noise region highlight boxes." 
        },
        'INVERSE_BOX_COLOR': { 
            'type': str, 
            'default': 'blue', 
            'description': "Color for inverse (content) region highlight boxes." 
        },
    }
)


class NoiseDetectionResult(FlowResultBase):
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
    
    # === LineTextIndex Query Configuration ===
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
    
    # === Metadata Section ===
    parameters_used: Optional[Dict[str, Any]] = section_field(
        section="metadata", 
        default=None, 
        description="Actual parameters used in processing"
    )
    
    image_config: Optional[Dict[str, Any]] = section_field(
        section="metadata", 
        default=None, 
        description="Configuration used for image generation"
    )
    
    noise_regions_by_page: Optional[Dict[int, Any]] = section_field(
        section="metadata", 
        default=None, 
        description="Noise regions organized by page"
    )
    
    @property
    def tracer_metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary suitable for workflow tracer logging."""
        return {
            'sample_images': {
                'result_pdf_image': self.result_image,
                'original_pdf_sample': self.original_image,
                'annotated_pdf_sample': self.annotated_image,
                'noise_regions_count': self.noise_regions_count,
                'inverse_regions_count': self.content_regions_count,
                'noise_regions_by_page': self.noise_regions_by_page,
                'inverse_noise_regions': self.content_regions,
                'pages_included': self.pages_analyzed,
                'metadata': self.image_config or {}
            },
            'image_config': self.image_config or {}
        }
    
    # Enhanced display methods using PDFModel
    def show_result_with_highlights(self, **kwargs):
        """Display the result using PDFModel with noise/content region highlights."""
        if not self.pdf_model:
            print("PDFModel not available, falling back to basic display")
            self.display_images()
            return
        
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


def _generate_noise_detection_images(pdf_model, noise_regions, params):
    """
    Generates sample images for noise detection results using the generic framework.
    
    Args:
        pdf_model: PDFModel instance
        noise_regions: Final noise region bounding boxes
        params: NoiseRegionParams instance
    
    Returns:
        NoiseDetectionResult: Complete result object with images and metadata
    """
    def process_noise_regions(pdf_model, noise_regions):
        """Single-pass processing with proper defaultdict initialization."""
        noise_regions_by_page = defaultdict(lambda: {'bboxes': []})
        
        for region in noise_regions:
            # Use VPM directly instead of LineTextIndex wrapper
            page_info = pdf_model.line_index.get_virtual_page_wh(region)
            pg_num = page_info['page_number']
            
            page_data = noise_regions_by_page[pg_num]
            page_data['page_width'] = page_info['page_width']
            page_data['page_height'] = page_info['page_height']
            page_data['bboxes'].append(region)
        
        return dict(noise_regions_by_page)
    
    # Process noise regions to get page structure
    noise_regions_by_page = process_noise_regions(pdf_model, noise_regions)
    inverse_noise_regions = Geometry.inverse_page_blocks(noise_regions_by_page)
    
    # Generate original PDF sample
    original_pdf_sample = pdf_model.get_page_base64(
        zoom=params.IMAGE_ZOOM,
        spacing=params.IMAGE_SPACING,
        page_limit=params.IMAGE_PAGE_LIMIT
    )
    
    # Generate annotated PDF sample
    annotated_pdf_sample = pdf_model.get_page_base64(
        highlight_boxes={
            'NoiseBounds': {
                'color': params.NOISE_BOX_COLOR, 
                'boxes': noise_regions
            },
            'InverseNoiseBounds': {
                'color': params.INVERSE_BOX_COLOR, 
                'boxes': inverse_noise_regions
            },
        },
        zoom=params.IMAGE_ZOOM,
        spacing=params.IMAGE_SPACING,
        page_limit=params.IMAGE_PAGE_LIMIT,
        box_width=params.IMAGE_BOX_WIDTH,
        font_size=params.IMAGE_FONT_SIZE
    )

    # Generate final result image (full PDF without page limit)
    result_pdf_image = pdf_model.get_page_base64(
        highlight_boxes={
            'InverseNoiseBounds': {
                'color': params.INVERSE_BOX_COLOR, 
                'boxes': inverse_noise_regions
            }
        },
        zoom=params.IMAGE_ZOOM,
        spacing=params.IMAGE_SPACING,
        page_limit=None  # Include all pages for final result
    )
    
    # KEY CHANGE: Create result with pdf_model reference and let framework handle the rest
    return NoiseDetectionResult(
        # NEW: Pass the PDFModel for rich display functionality
        pdf_model=pdf_model,
        
        # Core result images (framework handles these automatically)
        result_image=result_pdf_image,
        original_image=original_pdf_sample,
        annotated_image=annotated_pdf_sample,
        
        # Statistics (automatically organized into sections)
        noise_regions_count=len(noise_regions),
        content_regions_count=len(inverse_noise_regions),
        pages_analyzed=len(noise_regions_by_page),
        
        # Results (automatically organized into sections)
        noise_regions=noise_regions,
        content_regions=inverse_noise_regions,
        
        # Metadata (automatically organized into sections)
        parameters_used={
            'HEADER_BOUND': params.HEADER_BOUND,
            'FOOTER_BOUND': params.FOOTER_BOUND,
            'MIN_OCCURRENCES': params.MIN_OCCURRENCES,
            'WHITESPACE_TOLERANCE': params.WHITESPACE_TOLERANCE,
            'TOUCHING_TOLERANCE': params.TOUCHING_TOLERANCE,
            'query_label': params.query_label
        },
        image_config={
            'zoom': params.IMAGE_ZOOM,
            'spacing': params.IMAGE_SPACING,
            'page_limit': params.IMAGE_PAGE_LIMIT,
            'box_width': params.IMAGE_BOX_WIDTH,
            'font_size': params.IMAGE_FONT_SIZE,
            'noise_color': params.NOISE_BOX_COLOR,
            'inverse_color': params.INVERSE_BOX_COLOR
        },
        noise_regions_by_page=noise_regions_by_page
    )


def find_combined_noise_regions(
    pdf_model: 'PDFModel',
    trace: TraceLog,
    params: Optional[NoiseRegionParams] = None,
    **kwargs
) -> List[Tuple[float, ...]]:
    """
    Detects and removes recurring noise regions (headers, footers, processing statements) from PDF documents.
    
    This sophisticated multi-stage pipeline combines geometric analysis, pattern recognition, and spatial 
    reasoning to identify content that appears across multiple pages but isn't the main document content.
    Examples include "CARD PROCESSING STATEMENT" headers, merchant numbers, page footers, etc.
    
    Returns merged bounding boxes of all detected noise regions for masking during content extraction.
    """
    p = (params or NoiseRegionParams()).model_copy(update=kwargs)

    # Create configurable TextNormalizer and WhitespaceGenerator
    line_index_config = {
        'text_normalizer': TextNormalizer(
            patterns=p.TEXT_NORMALIZER_PATTERNS
        ),
        'whitespace_generator': WhitespaceGenerator(
            min_gap=p.WHITESPACE_MIN_GAP
        )
    }
    
    # Convert to serializable format for tracing
    line_index_trace_config = {
        'text_normalizer': line_index_config['text_normalizer'].to_dict(),
        'whitespace_generator': line_index_config['whitespace_generator'].to_dict()
    }
    
    line_index = trace.run_and_log_step(
        "Step 0: LineTextIndex Configuration & Initialization",
        function=lambda: pdf_model.line_index,  # Use existing line_index (already configured)
        params=line_index_trace_config,
        description="Configures the LineTextIndex with TextNormalizer and WhitespaceGenerator models. TextNormalizer standardizes recurring text patterns (like page numbers) while WhitespaceGenerator detects full-width vertical whitespace regions that often accompany header/footer content."
    )

    # Step 1: Foundation Data Gathering
    base_q = QueryParams(
        key="normalized_text",
        groupby=groupby(
            *p.BLOCK_GROUPBY_KEYS,
            include=p.BLOCK_GROUP_QUERY_INCLUDED_VALUES,
            query_label=p.BLOCK_GROUP_QUERY_LABEL
        )
    )
    grouped = trace.run_and_log_step(
        "Step 1: Foundation Data Gathering - Group all text lines by block identifier",
        function=lambda: line_index.query(params=base_q),
        log_function=line_index.query,
        params=base_q,
        description="""
        Creates the foundation dataset by grouping all text elements in the PDF 
        by their block identifiers. This establishes the base structure for subsequent noise detection analysis.
        """
    )

    # Step 2: Geometric Header/Footer Detection
    process_params = {
        'aggregations': {
            'merged_bbox': GroupOps.merge_bboxes,
            'merged_bbox_rel': lambda g: GroupOps.merge_bboxes(g, key='group_bboxes_rel'),
            'full_text': GroupOps.concat_text
        },
        'filters': [
            lambda d: (
                d.get('merged_bbox_rel') is not None and
                d.get('physical_page_bounds') is not None and
                (d['merged_bbox_rel'][1] < p.HEADER_BOUND or
                 d['merged_bbox_rel'][3] > (d['physical_page_bounds']['height'] - p.FOOTER_BOUND))
            )
        ],
        'include': ['page','member_count','physical_page_bounds']
    }
    
    processed = trace.run_and_log_step(
        "Step 2: Geometric Header/Footer Detection - Filter blocks by position",
        function=lambda: grouped.group.process(**process_params),
        log_function=grouped.group.process,
        params=process_params,
        description="""
        Applies geometric filtering to identify text blocks positioned in 
        header zones (y < HEADER_BOUND) or footer zones (y > page_height - FOOTER_BOUND). 
        Aggregates bounding boxes and merges spatial data to understand element positioning.
        """
    )

    # Step 3: Recurring Pattern Confirmation
    reg_args = {'key':'full_text','min_count':p.MIN_OCCURRENCES,'return_list':True}
    rec = trace.run_and_log_step(
        "Step 3: Recurring Pattern Confirmation - Validate by text content frequency",
        function=lambda: regroup_by_key(processed, **reg_args),
        log_function=regroup_by_key,
        params=reg_args,
        description="""
        Takes geometrically-suspected header/footer blocks and groups them by actual text content. 
        Filters for elements appearing at least MIN_OCCURRENCES times across pages, 
        confirming which geometric suspects are truly recurring noise patterns.
        """
    )
    hf_boxes = [x['merged_bbox'] for x in rec if x.get('merged_bbox')]

    # Step 4: Horizontal Whitespace Integration
    ws_p = HorizontalWhitespaceParams(y_tolerance=p.WHITESPACE_TOLERANCE)
    whitespace = trace.run_and_log_step(
        "Step 4: Horizontal Whitespace Integration - Detect full-width whitespace regions",
        function=lambda: horizontal_whitespace(line_index, params=ws_p).pluck('bbox'),
        log_function=horizontal_whitespace,
        params=ws_p,
        description="""
        Identifies horizontal whitespace regions that span the page width. 
        These often accompany header/footer content and help define complete noise 
        regions that include both text and surrounding whitespace.
        """
    )

    # Step 5: Page Margin Retrieval
    margin_bboxes = pdf_model.virtual_page_metadata['page_content_areas']['margin_bboxes']
    trace.run_and_log_step(
        "Step 5: Page Margin Retrieval - Extract document margin boundaries",
        function=lambda: margin_bboxes,
        description="""
        Retrieves pre-computed page margin bounding boxes that define the document's 
        content boundaries. These margins often contain or border noise elements and are crucial 
        for complete noise region detection.
        """
    )

    # Step 6: Header/Footer + Whitespace Consolidation
    touch_p = GroupTouchingBoxesParams(tolerance=p.TOUCHING_TOLERANCE)
    tg1 = trace.run_and_log_step(
        "Step 6: Header/Footer + Whitespace Consolidation - Merge spatially adjacent regions",
        function=lambda: group_vertically_touching_bboxes(whitespace, hf_boxes, params=touch_p),
        log_function=group_vertically_touching_bboxes,
        params=touch_p,
        description="""
        Groups confirmed header/footer text with horizontally adjacent whitespace regions. 
        Uses vertical proximity detection to merge text and whitespace that 
        form cohesive noise regions (e.g., header text + surrounding white space).
        """
    )
    merged1 = [Geometry.merge_all_boxes(g) for g in tg1]

    # Step 7: Final Spatial Integration with Margins
    tg2 = trace.run_and_log_step(
        "Step 7: Final Spatial Integration - Merge with page margins",
        function=lambda: group_vertically_touching_bboxes(margin_bboxes, merged1, params=touch_p),
        log_function=group_vertically_touching_bboxes,
        params=touch_p,
        description="""
        Performs final spatial integration by merging the consolidated 
        header/footer+whitespace regions with page margins. This captures complete 
        noise regions that extend to or include margin areas.
        """
    )

    # Step 8: Final Consolidation
    final = [Geometry.merge_all_boxes(g) for g in tg2]
    trace.run_and_log_step(
        "Step 8: Final Consolidation - Produce merged noise region bounding boxes",
        function=lambda: final,
        description="""
        Consolidates all grouped regions into final merged bounding boxes representing complete 
        noise regions. These bounding boxes can be used to mask out recurring headers, 
        footers, and processing statements during document content extraction.
        """
    )

    process_result = trace.run_and_log_step(
        "Step 9: Generate Sample Images - Create original and annotated PDF previews",
        function=lambda: _generate_noise_detection_images(pdf_model, final, p),
        description="""
        Generates base64-encoded sample images showing:
        1. Original PDF pages within page limit
        2. Annotated PDF with noise regions (red) and content regions (blue) highlighted
        These images provide visual confirmation of the noise detection results.
        """
    )
    
    # Add images to trace metadata
    # trace.add_metadata(process_result.tracer_metadata)
    return process_result