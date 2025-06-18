from tableai.pdf.coordinates import Geometry
from tableai.pdf.query_funcs_wrappers import (
    merge_all_bboxes, 
    concat_text
)
from tableai.pdf.generic_params import GenericFunctionParams, QueryParams
from tableai.pdf.flows.generic_flow import GenericPDFFlowContext
from tableai.pdf.flows.generic_params import FlowParams
from tableai.pdf.generic_results import ResultSet, ChainResult
from tableai.pdf.flows.generic_results import FlowResultStage
from tableai.pdf.generic_types import BoundingBoxes, TextBlocks, BBox, BBoxList, GroupFunction
from tableai.pdf.flows.generic_dependencies import FlowDependencies
from tableai.pdf.flows.reduce_pdf_noise.result_model import NoiseDetectionResult
from tableai.pdf.flows.query_helper import QueryBuilder, GenericQueryBase
from tableai.pdf.flows.generic_context import NodeContext, RunContext, StepInput
from collections import defaultdict
from pydantic import ConfigDict, BaseModel
from tableai.pdf.generic_tools import BaseChain, regroup_by_key
from dataclasses import dataclass
from tableai.pdf.flows.trace import trace_call
from typing import Optional, Any, List, Callable, Dict, Set, Tuple

import argparse
import asyncio
import pprint
import base64
import os

# For the high-level find_combined_noise_regions orchestrator
ImageProcessingParams = GenericFunctionParams.create_custom_model(
    "ImageProcessingParams", {
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


pdf_flow_runner = GenericPDFFlowContext(
    # Pass the flow's definition via the new FlowParams object
    flow_params=FlowParams(
        deps_type=FlowDependencies,
        result_type=NoiseDetectionResult,
        overview="Detects and removes recurring noise regions...",
        goal="To produce a clean list of 'content' regions."
    ),
)


NodeFlow0 = QueryBuilder.build(
    key="normalized_text",
    groupby_keys=["block"],
    include_fields=["bbox", "path", "normalized_text", "bbox(rel)"],
)
@pdf_flow_runner.flow.step(NodeContext[NodeFlow0](
        description="""
        Creates the foundation dataset by grouping all text elements in the PDF 
        by their block identifiers. This establishes the base structure for subsequent noise detection analysis.
        Simple index query to get fitz span data grouped by [block].
        """
    )
)
async def group_text_lines(ctx: RunContext, input: StepInput[GenericQueryBase]) -> ResultSet:
    """Groups all text lines by block identifier."""
    query_index = ctx.deps.pdf_model.query_index
    
    query_config = input.data
    extra_config = input.config
    
    final_query = query_config.build_query()
    
    # =============================================================
    # THE CORE CHANGE: Execute the critical call THROUGH the tracer.
    # =============================================================
    return await trace_call(
        ctx,
        target_object=query_index,
        method_name='query',
        params=final_query  
    )


@pdf_flow_runner.flow.step(
    NodeContext[group_text_lines](
        expected_return_type=ChainResult,
        input_model=HeaderFooterParams,
        description="""
        Applies geometric filtering to identify text blocks positioned in 
        header zones (y < HEADER_BOUND) or footer zones (y > page_height - FOOTER_BOUND). 
        Aggregates bounding boxes and merges spatial data to understand element positioning.
        """
    )
)
# 2. The function's return type hint can be BaseChain, as it returns the unexecuted chain object.
async def header_footer_filter(ctx: RunContext, input: StepInput[ResultSet]) -> BaseChain:
    """
    Takes the grouped ResultSet and returns a configured chain for geometric filtering.
    The Flow Runner will automatically execute this chain.
    """
    grouped_data = input.data
    config = input.config
    if input.input_model:
        HEADER_BOUND = getattr(input.input_model, 'HEADER_BOUND', 100)
        FOOTER_BOUND = getattr(input.input_model, 'FOOTER_BOUND', 100)
    else:
        HEADER_BOUND, FOOTER_BOUND = 100, 100
    
    # This logic is UNCHANGED. It correctly returns the GroupChain object.
    grp = (grouped_data.group.chain
        .include(['page', 'member_count', 'physical_page_bounds'])
        .agg({
            'merged_bbox': merge_all_bboxes('group_bboxes'),
            'merged_bbox_rel': merge_all_bboxes('group_bboxes_rel'),
            'full_text': concat_text('group_text')
        })
        .filter(lambda d: (
            d.get('merged_bbox_rel') is not None and
            d.get('physical_page_bounds') is not None and
            (d['merged_bbox_rel'][1] < HEADER_BOUND or
             d['merged_bbox_rel'][3] > (d['physical_page_bounds']['height'] - FOOTER_BOUND))
        ))
    )
    return grp



@pdf_flow_runner.flow.step(
    NodeContext[header_footer_filter](
        expected_return_type=ChainResult,
        description="""
        Takes geometrically-suspected header/footer blocks and groups them by actual text content. 
        Filters for elements appearing at least MIN_OCCURRENCES times across pages, 
        confirming which geometric suspects are truly recurring noise patterns.
        """, 
        input_model=HeaderFooterParams
    ),
)
# 2. The function's return type hint can be BaseChain, as it returns the unexecuted chain object.
async def filter_recurring_block_pattern(ctx: RunContext, input: StepInput[ResultSet]) -> BoundingBoxes:
    """
    Takes the grouped ResultSet and returns a configured chain for geometric filtering.
    The Flow Runner will automatically execute this chain.
    """
    grouped_data = input.data
    config = input.config
    
    if input.input_model:
        MIN_OCCURRENCES = getattr(input.input_model, 'MIN_OCCURRENCES', 2)
        print(f"MIN_OCCURRENCES: {MIN_OCCURRENCES}")

    # regroup_params = 
    group_filtered = regroup_by_key(data=grouped_data, key='full_text', min_count=MIN_OCCURRENCES, return_list=True)
    hf_boxes = [x['merged_bbox'] for x in group_filtered if x.get('merged_bbox')]
    return hf_boxes

@dataclass
class DefaultContext:
    description: str = 'A description for this flow has not been set yet.'

class GenericInput(BaseModel):
    pass

HorizontalWhitespaceParams = GenericFunctionParams.create_custom_model(
    "HorizontalWhitespaceParams", {
        'page_number': { 'type': Optional[int], 'default': None, 'description': "Optional page number to search within." },
        'y_tolerance': { 'type': int, 'default': 10, 'description': "Minimum vertical gap to be considered whitespace." }
    }
)
@pdf_flow_runner.flow.step(
    NodeContext[DefaultContext](
        page_number=0, y_tolerance=10, query_label='get_fw_whitespace', input_model=HorizontalWhitespaceParams
    )
)
async def get_large_whitespace_blocks(ctx: RunContext, input: StepInput[GenericInput]) -> BoundingBoxes:
    """Finds full-width whitespace blocks using a self-initializing parameter model."""
    query_index = ctx.deps.pdf_model.query_index
    input_model = input.input_model
    config = input.config
    transform_func = lambda rows: [r for r in rows if r["gap"] >= input_model.y_tolerance]

    query_params = QueryParams(
        page=input_model.page_number,
        key="full_width_v_whitespace",
        transform=transform_func,
        query_label=config.get('query_label', '')
    )
    return query_index.query(params=query_params).pluck('bbox')


def group_vertically_touching_bboxes(
    whitespace_blocks: List[Tuple[float, ...]], 
    header_footer_blocks: List[Tuple[float, ...]],
    y_tolerance = 10,
    **kwargs
) -> List[List[Tuple[float, ...]]]:
    """Groups vertically touching bboxes using a self-initializing parameter model."""
    bboxes = whitespace_blocks + header_footer_blocks
    
    # Handle edge case of an empty list
    if not bboxes:
        return []

    # --- OPTIMIZATION: Convert to a set for fast O(1) lookups later ---
    # This is the key to making the final filtering step efficient.
    header_footer_set: Set[Tuple[float, ...]] = set(header_footer_blocks)

    # 1. Sort bboxes by their top coordinate (y0)
    sorted_bboxes = sorted(bboxes, key=lambda b: b[1])

    # 2. Initialize the first group
    all_groups: List[List[Tuple[float, ...]]] = []
    current_group: List[Tuple[float, ...]] = [sorted_bboxes[0]]

    # 3. Iterate and group based on vertical proximity
    for i in range(1, len(sorted_bboxes)):
        current_bbox = sorted_bboxes[i]
        previous_bbox_in_chain = current_group[-1]
        
        gap = current_bbox[1] - previous_bbox_in_chain[3]  # current.y0 - previous.y1

        if gap <= y_tolerance:
            current_group.append(current_bbox)
        else:
            all_groups.append(current_group)
            current_group = [current_bbox]

    # 4. Add the last group
    if current_group:
        all_groups.append(current_group)
        
    # --- CORRECTED FILTERING LOGIC ---
    # Use a list comprehension that checks for membership in the efficient set.
    # For each 'group', keep it if 'any' 'bbox' in that group exists in the 'header_footer_set'.
    return [
        group for group in all_groups 
        if any(bbox in header_footer_set for bbox in group)
    ]

GroupTouchingBoxesParams = GenericFunctionParams.create_custom_model(
    "GroupTouchingBoxesParams", {
        'y_tolerance': { 'type': float, 'default': 2.0, 'description': "Max vertical distance between boxes to be considered 'touching'." }
    }
)
@pdf_flow_runner.flow.step(
    NodeContext[get_large_whitespace_blocks, filter_recurring_block_pattern](
        description="""
        Groups confirmed header/footer text with horizontally adjacent whitespace regions. 
        Uses vertical proximity detection to merge text and whitespace that 
        form cohesive noise regions (e.g., header text + surrounding white space).
        ...
        This function also gets margin bboxes accessed from metadata on the core pdf model.
        """,
        input_model=GroupTouchingBoxesParams
    )
)
async def run_group_vertically_touching_bboxes(ctx: RunContext, input: StepInput[ResultSet]) -> List[List[Tuple[float, ...]]]:
    pdf_model = ctx.deps.pdf_model
    pdf_margin_bboxes = pdf_model.virtual_page_metadata['page_content_areas']['margin_bboxes']
    
    grouped_data = input.data
    whitespace_blocks = input.data['get_large_whitespace_blocks']
    header_footer_blocks_cleaned = input.data['filter_recurring_block_pattern']

    merged_heder_footer_blocks = group_vertically_touching_bboxes(
        whitespace_blocks=whitespace_blocks, 
        header_footer_blocks=header_footer_blocks_cleaned,
        y_tolerance=input.input_model.y_tolerance,
    )
    merged1 = [Geometry.merge_all_boxes(g) for g in merged_heder_footer_blocks]

    final_merged_bboxes = group_vertically_touching_bboxes(
        whitespace_blocks=pdf_margin_bboxes, 
        header_footer_blocks=merged1,
        y_tolerance=input.input_model.y_tolerance,
    )
    final = [Geometry.merge_all_boxes(g) for g in final_merged_bboxes]
    return final

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
            # Use VPM directly instead of FitzSearchIndex wrapper
            page_info = pdf_model.query_index.get_virtual_page_wh(region)
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
    return FlowResultStage(
        pdf_model=pdf_model,
        result_image=result_pdf_image,
        original_image=original_pdf_sample,
        annotated_image=annotated_pdf_sample,
        noise_regions_count=len(noise_regions),
        content_regions_count=len(inverse_noise_regions),
        pages_analyzed=len(noise_regions_by_page),
        noise_regions=noise_regions,
        content_regions=inverse_noise_regions,
        noise_regions_by_page=noise_regions_by_page
    )

@pdf_flow_runner.flow.step(NodeContext[run_group_vertically_touching_bboxes](
        description="""
        Generates base64-encoded sample images showing:
        1. Original PDF pages within page limit
        2. Annotated PDF with noise regions (red) and content regions (blue) highlighted
        These images provide visual confirmation of the noise detection results.
        """,
        input_model=ImageProcessingParams
    )
)
async def render_images_from_output_coordinates(ctx: RunContext, input: StepInput[ResultSet]) -> FlowResultStage:
    pdf_model = ctx.deps.pdf_model
    noise_regions = input.data
    return _generate_noise_detection_images(pdf_model, noise_regions, params=input.input_model)



def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the PDF Noise Detection Flow on a specified document.",
        formatter_class=argparse.RawTextHelpFormatter
    )
        
    parser.add_argument(
        "pdf_path",
        type=str,
        help="The full path to the PDF file to process."
    )

    return parser.parse_args()

async def main(args):
    """
    The main asynchronous function to configure and run the flow.
    """
    print("=============================================")
    print("    PDF NOISE DETECTION WORKFLOW STARTING    ")
    print("=============================================")

    final_result = await pdf_flow_runner.run(pdf_path=args.pdf_path)
    print(final_result)

if __name__ == "__main__":
    # This block only runs when you execute the script directly,
    # e.g., `python your_script_name.py /path/to/file.pdf`
    
    # Parse the command-line arguments
    cli_args = setup_arg_parser()
    
    # Run the main async function with the parsed arguments
    asyncio.run(main(cli_args))