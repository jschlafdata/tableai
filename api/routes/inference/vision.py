from fastapi import (
    APIRouter,
    Query, 
    Depends,
    HTTPException,
    Request
)

import json
from typing import List, Any, Dict, Union, Optional
import uuid
from pathlib import Path
from pydantic import BaseModel
from collections import defaultdict

### Application imports ###
from api.service.dependencies import ensure_initialized
from tableai.core.serialize import serialize
from backend.models.backend import (
    DropboxSyncRecord, 
    FileExtractionResult, 
    PDFClassifications, 
    ClassificationLabel, 
    FileNodeRecord,
    LLMInferenceTableStructures
)

from tableai.extract.process import DropboxRegisterService
from tableai.nodes.file_node import DirectoryFileNode
from tableai.extract.table_headers import TableHeaderExtractor

from api.models.requests import (
    PDFExtractRequest, 
    LLMTableStructureRequest,
    LLMInferenceResultRequest
)

from tableai.extract.index_search import (
     FitzTextIndex, 
     LineTextIndex
)

from tableai.inference.prompts.table_structure import (
    TABLE_STRUCTURE_PROMPT,
    TABLE_STRUCTURE_PROMPT_V2,
    TABLE_STRUCTURE_PROMPT_V3,
    TABLE_STRUCTURE_PROMPT_V4,
    TABLE_STRUCTURE_PROMPT_V5
)
from tableai.inference.vision_call import VisionInferenceClient
from tableai.extract.helpers import Stitch

### ------------------ ###

router = APIRouter()

import fitz

from tableai.extract.query_engine import QueryEngine
from tableai.extract.helpers import Map, Stitch
from collections import defaultdict

from tableai.inference.pdf_vision import VisionInferenceClient
from tableai.nodes.pdf_node import (
    PDFModel,
    Source
)
from api.models.requests import (
    PdfVisionModelRequest,
    VisionModelOptions
)

from tableai.extract.table_headers import TableHeaderExtractor

class TableColumnsEntry(BaseModel):
    headers_hierarchy: bool
    columns: List[str]

class TableCountWithColumnsDict(BaseModel):
    number_of_tables: int
    tables: Dict[str, TableColumnsEntry]

class TableWithNameAndTotals(BaseModel):
    table_name: Optional[str]
    columns: List[str]
    totals_row_label: Optional[str]
    bottom_right_cell_value: Optional[str]
    table_breakers: Optional[List[str]]

class TableCountWithNamesAndTotalsDict(BaseModel):
    number_of_tables: int
    tables: Dict[str, TableWithNameAndTotals]

@router.post("/find_table_columns")
def get_extraction(
    req: PdfVisionModelRequest,
    request_obj: Request,
    api_service: 'APIService' = Depends(ensure_initialized)
):
    db = api_service.db
    node = db.run_op(FileNodeRecord, "get", filter_by={'uuid': req.file_id})
    register_service = DropboxRegisterService(db)
    if node:
        print(f"RUNNING promptId: {req.promptId}")
        if req.promptId == 1:
            node_stage_paths = json.loads(node[0].stage_paths_json)
            abs_path = node_stage_paths.get(str(0), {}).get('abs_path', None)
            if abs_path:
                pdf_model = PDFModel(path=Path(abs_path), source=Source.LOCAL)
                if req.page_limit:
                    pdf_model.set_limit(req.page_limit)

                # Use the received request model directly
                client = VisionInferenceClient(
                    username=api_service.service_config.ollama_api_user,
                    password=api_service.service_config.ollama_api_key,
                )
                results = client.process_pdf(req, pdf_model, TableCountWithColumnsDict)
                if results:
                    result_dict = [res.dict() for res in results if hasattr(res, 'dict') and res.dict()]
                    abs_path_3 = node_stage_paths.get(str(2), {}).get('abs_path', None)
                    if abs_path_3:
                        header_extractor = TableHeaderExtractor(abs_path_3)
                        header_results, core_results = header_extractor.process(result_dict)

                        try:
                            multi_page_doc, table_idx_metadata = unqiue_tables_to_pages(abs_path_3, core_results['relative_bounds'])
                            directory_file_node = register_service.get_node(file_id=req.file_id)
                            directory_file_node.add_stage(3)

                            directory_file_node.store_metadata(3, {3: table_idx_metadata}, db)
                            stage3_outpath = directory_file_node.stage_paths[3]['abs_path']
                            multi_page_doc.save(stage3_outpath)
                
                            index = FitzTextIndex.from_document(multi_page_doc)
                            all_pages_text_index = index.query(**{"blocks[*].lines[*].spans[*].text": "*"}, restrict=["text", "font", "bbox"])
                            line_index = LineTextIndex(all_pages_text_index, page_metadata=index.page_metadata)
                            engine = QueryEngine(line_index)
                            numbers = engine.get("Numbers")
                            percentages = engine.get("Percentages")
                            paragraphs = engine.get("Paragraphs")
                            dates = engine.get("Dates")
                            toll_free = engine.get("Toll.Free.#")
                            horizontal_whitespace = engine.get("Horizontal.Whitespace")
                        except:
                            return {
                                'inference_result': results, 
                                'header_bounds': header_results,
                                'core_results': core_results,
                            }
                        # return header_results
                        if header_results:
                            return {
                                'inference_result': results, 
                                'header_bounds': header_results,
                                'core_results': core_results,
                                'file_node_path': abs_path_3,
                                'stage3': [
                                    numbers, 
                                    percentages,
                                    paragraphs, 
                                    dates, 
                                    toll_free, 
                                    horizontal_whitespace
                                ]
                            }
                return {'inference_result': results}

        if req.promptId == 2:
            node_stage_paths = json.loads(node[0].stage_paths_json)
            abs_path = node_stage_paths.get(str(3), {}).get('abs_path', None)
            print(f'RUNNING ABS PATH: {abs_path}')
            if abs_path:
                pdf_model = PDFModel(path=Path(abs_path), source=Source.LOCAL)
                if req.page_limit:
                    pdf_model.set_limit(req.page_limit)

                # Use the received request model directly
                client = VisionInferenceClient(
                    username=api_service.service_config.ollama_api_user,
                    password=api_service.service_config.ollama_api_key,
                )
                results = client.process_pdf(req, pdf_model, TableCountWithNamesAndTotalsDict)
                print(f"RESULTS: {results}")
                return {'inference_result': results, 'request_params': req.dict(), 'abs_path': abs_path}
    else:
        return {'inference_result': {}}



def unqiue_tables_to_pages(file_path, relative_bounds):

    # relative_bounds = response.json()['core_results']['relative_bounds']

    table_idx_bounds = defaultdict(list)
    table_idx_bounds_seen = defaultdict(set)  # <-- This tracks unique bboxes per table
    table_idx_metadata = defaultdict(list)

    for table in relative_bounds:
        tidx = table['table_index']
        bbox = table['table_metadata'].get('refined_table_bounds', {}).get('bbox', None)
        if bbox is not None:
            bbox_tuple = tuple(bbox)
            if bbox_tuple not in table_idx_bounds_seen[tidx]:
                table_idx_bounds[tidx].append(bbox)
                table_idx_bounds_seen[tidx].add(bbox_tuple)
        # No deduplication needed here for metadata, but keep as is:
        table_idx_metadata[tidx].append({
            'table_index': table['table_index'],
            'headers_bbox': table['bbox'],
            'columns': table['columns'],
            'hierarchy': table['hierarchy'],
            'bounds_index': table['bounds_index']
        })

    print(f"table_idx_bounds: {table_idx_bounds}")

    table_docs = {}
    for tbl_idx, bounds in table_idx_bounds.items():
        d = Stitch.clip_and_place_pdf_regions(
            input_pdf_path=file_path,
            regions=bounds,
            source_page_number=0,
            layout='vertical',
            page_width=None,
            page_height=None,
            margin=20,
            gap=2,
            center_horizontally=True
        )
        if d:
            table_docs[tbl_idx] = d
            width = d[0].rect.width
            height = d[0].rect.height
            for item in table_idx_metadata[tbl_idx]:
                item['page_metadata'] = {0: {'width': width, 'height': height}}

    output_pdf = fitz.open()  # Empty PDF to collect all table pages

    for idx in sorted(table_docs.keys()):
        doc = table_docs[idx]
        # Insert the first (and only) page from each table_doc
        output_pdf.insert_pdf(doc, from_page=0, to_page=0)

    return output_pdf, table_idx_metadata