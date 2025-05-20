from fastapi import (
    APIRouter,
    Query, 
    Depends,
    HTTPException,
    Request
)

import json
from typing import List, Any, Dict

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

from tableai.inference.prompts.table_structure import (
    TABLE_STRUCTURE_PROMPT,
    TABLE_STRUCTURE_PROMPT_V2,
    TABLE_STRUCTURE_PROMPT_V3,
    TABLE_STRUCTURE_PROMPT_V4,
    TABLE_STRUCTURE_PROMPT_V5
)
from tableai.inference.vision_call import VisionInferenceClient

### ------------------ ###

router = APIRouter()

import fitz
from tableai.extract.index_search import  (
     FitzTextIndex, 
     LineTextIndex, 
     groupby, 
     filterby, 
     chain_transform,
     render_api, 
     search_normalized_text,
     normalize_word,
     normalize_recurring_text, 
     patterns, 
     try_convert_float, 
     identify_currency_symbols,
     try_convert_date,
     find_paragraph_blocks,
     try_convert_percent,
     merge_result_group_bounds,
     expand_bounds,
     PDFTools
 )

from tableai.extract.query_engine import QueryEngine
from tableai.extract.helpers import Map, Stitch
from collections import defaultdict

class ProcessStages:
    def __init__(self, req, db, node):
        self.req = req
        self.db = db
        self.node = node
        self.process_results = {}
        self._runit()
        self._storeit()
    
    def _runit(self):
        for stage in [0,1,2]:
            self.node_stage_process(stage)
    
    def _storeit(self):
        print(self.process_results.keys())
        for index, (stage, metadata) in enumerate(self.process_results.items()):
            print(f"storing metadata from stage run: {index}")
            if index == 0:
                pass 
            else:
                self.directory_file_node.store_metadata(index, {stage: metadata}, self.db)

    def node_stage_process(self, stage):
        current_stage = f"stage{stage}"
        print(f"RUNNING STAGE: {current_stage}")
        if stage == 0:
            node_stage_paths = json.loads(self.node[0].stage_paths_json)
            abs_path = node_stage_paths.get(str(stage), {}).get('abs_path', None)
        else:
            node_stage_paths = self.directory_file_node.stage_paths
            abs_path = node_stage_paths.get(stage, {}).get('abs_path', None)
        print(f"RUNNING abs_path: {abs_path}")
        if abs_path:
            doc = fitz.open(abs_path)
            index = FitzTextIndex.from_document(doc)
            all_pages_text_index = index.query(**{"blocks[*].lines[*].spans[*].text": "*"}, restrict=["text", "font", "bbox"])
            line_index = LineTextIndex(all_pages_text_index, page_metadata=index.page_metadata)
            engine = QueryEngine(line_index)

            if stage == 0:
                self.header_footer_bounds = engine.get("Header.Footer.Blocks")
                register_service = DropboxRegisterService(self.db)
                if self.header_footer_bounds:
                    footer_header_bounds = self.header_footer_bounds.get('results', {}).get('pages', {})
                    self.stage1_pdf_metadata = self.header_footer_bounds['pdf_metadata']
                else:
                    footer_header_bounds = {}
                self.directory_file_node = register_service.get_node(file_id=self.req.file_id)
                self.directory_file_node.add_stage(1)
                redacted_doc = Stitch.redact_pdf_regions(
                    self.directory_file_node.doc, 
                    footer_header_bounds
                )
                print(f"CURRENT LABEL: {register_service._get_node_label(self.directory_file_node)}")
                Stitch.combine_pages_into_one(redacted_doc, self.directory_file_node)

            if stage == 1:
                header_footer_bounds = self.translate_header_bounds(self.header_footer_bounds)
                pdf_metadata=header_footer_bounds['pdf_metadata']
                line_index.set_search_result_bound_restrictions(header_footer_bounds)
                engine.line_index = line_index
                self.header_footer_bounds = header_footer_bounds  


            page_indicators = engine.get("Page.Indicators")
            numbers = engine.get("Numbers")
            percentages = engine.get("Percentages")
            paragraphs = engine.get("Paragraphs")
            dates = engine.get("Dates")
            toll_free = engine.get("Toll.Free.#")
            horizontal_whitespace = engine.get("Horizontal.Whitespace")

            search_results_list = [
                page_indicators, 
                numbers, 
                percentages,
                paragraphs, 
                dates, 
                toll_free, 
                horizontal_whitespace
            ]
            if stage == 0:
                search_results_list.append(self.header_footer_bounds)
            if stage ==1:
                ignore_blocks, stage1_data_blocks_dict, stage1_data_blocks, width, height = self.combine_results_and_relabel(
                    [horizontal_whitespace, self.header_footer_bounds],
                    new_label='ignore_blocks', 
                    description='Whitespace or Header regions with redactions.', 
                    pdf_metadata=pdf_metadata,
                    data_region_y_margin=1
                )
                search_results_list.append(ignore_blocks)
                search_results_list.append(stage1_data_blocks_dict)
                self.directory_file_node.add_stage(2)

                output_pdf_path = self.directory_file_node.stage_paths.get(2, {}).get('abs_path', None)
                stage2_doc = Stitch.clip_and_place_pdf_regions(
                    input_pdf_path=abs_path,
                    regions=stage1_data_blocks,
                    source_page_number=0,
                    layout='vertical',
                    page_width=width,
                    page_height=height,
                    margin=20,
                    gap=10,
                    center_horizontally=True
                )
                stage2_doc.save(output_pdf_path)
            self.process_results[current_stage] = search_results_list
    
    def translate_header_bounds(self, hf_bounds):
        all_items = []
        width, height = Map.get_virtual_page_size(hf_bounds['pdf_metadata'])
        for pg_num, items in hf_bounds['results']['pages'].items():
            for res in items:
                bbox = res['bbox']
                res['bbox'] = Map.translate_bbox_to_virtual_page(bbox, int(pg_num), hf_bounds['pdf_metadata'])
                res['page'] = '0'
                res['meta'] = {'width': width, 'height': height}
                all_items.append(res)

        # Overwrite the original dict
        hf_bounds['results']['pages'] = {'0': all_items}
        hf_bounds['pdf_metadata'] = {0: {'width': width, 'height': height}}
        return hf_bounds

    def combine_results_and_relabel(self, item_groups, new_label='', description='', pdf_metadata=None, data_region_y_margin=1):
        if pdf_metadata is None:
            pdf_metadata = {}
        else:
            width = pdf_metadata[0]['width']
            height = pdf_metadata[0]['height']

        blocks = defaultdict(list)
        recurring_blocks_list=[]
        for item_group in item_groups:
            for pg_num, items in item_group['results']['pages'].items():
                for res in items:
                    blocks[pg_num].append(res)
                    recurring_blocks_list.append(res['bbox'])
        inverse_blocks = Map.create_inverse_blocks(width, height, recurring_blocks_list)
        inverse_results = []
        for bbox in inverse_blocks:
            x0,y0,x1,y1 = bbox
            new_bbox = (x0,y0-data_region_y_margin,x1, y1+data_region_y_margin)
            inverse_results.append({
                'bbox': new_bbox,
                'page': '0',
                'region': 'inverse',
                'value': 'Data Regions',  # Or ''
            })
        combined_results = {
            'results': {'pages': dict(blocks)},
            'pdf_metadata': pdf_metadata,
            'query_label': new_label,
            'description': description,
        }
        inverse_combined_results = {
            'results': {'pages': {'0': inverse_results}},
            'pdf_metadata': pdf_metadata,
            'query_label': 'Data.Regions',
            'description': 'Page regions with tables or valid data.',
        }
        return combined_results, inverse_combined_results, inverse_results, width, height

@router.post("/doc_query")
def get_extraction(
        req: PDFExtractRequest,
        request_obj: Request,
        api_service: 'APIService' = Depends(ensure_initialized)
    ):
    db = api_service.db
    print(f"processing file id: {req.file_id}")
    node = db.run_op(FileNodeRecord, "get", filter_by={'uuid': req.file_id})
    if node:
        stage_processor = ProcessStages(req, db, node)
        return stage_processor.process_results
    else:
        return {}

import uuid

@router.post("/vision/structure")
def get_extraction(
        req: LLMTableStructureRequest,
        request_obj: Request,
        api_service: 'APIService' = Depends(ensure_initialized)
    ):
    db = api_service.db
    node = db.run_op(FileNodeRecord, "get", filter_by={'uuid': req.file_id})
    if node:
        node_stage_paths = json.loads(node[0].stage_paths_json)
        abs_path = node_stage_paths.get(str(req.stage), {}).get('abs_path', None)
        mount_path = node_stage_paths.get(str(req.stage), {}).get('mount_path', None)

        client = VisionInferenceClient(
            model_library={
                'mistralVL': ('mistral-small3.1:24b', 'ollama-medium-ollama.portal-ai.tools'),
                'llamaVL':   ('llama3.2-vision', 'ollama-embeddings-ollama.portal-ai.tools')
            },
            model_choice='mistralVL',
            username=api_service.service_config.ollama_api_user,
            password=api_service.service_config.ollama_api_key,
            page_limit=None
        )

        print(f"CLASSIFICAITON: {req.classification_label}")

        results, timings = client.process_pdf(
            pdf_path=abs_path,
            prompt=TABLE_STRUCTURE_PROMPT_V5,
        )

        inference_model = LLMInferenceTableStructures(
            run_uuid=str(uuid.uuid4()),
            uuid=req.file_id,
            stage=req.stage,
            classification_label=req.classification_label,
            prompt=TABLE_STRUCTURE_PROMPT_V5, 
            prompt_version=5, 
            prompt_name="TABLE_STRUCTURE_PROMPT", 
            response=json.dumps(results)
        )

        db.run_op(
            LLMInferenceTableStructures, 
            operation="merge", 
            data=inference_model
        )


@router.post("/vision/table_headers")
def get_extraction(
        req: LLMInferenceResultRequest,
        request_obj: Request,
        api_service: 'APIService' = Depends(ensure_initialized)
    ):
    db = api_service.db
    node = db.run_op(FileNodeRecord, "get", filter_by={'uuid': req.file_id})

    inference_results = db.run_op(LLMInferenceTableStructures, "get", filter_by={
        'classification_label': req.classification_label
    }) 
    prompt_versions = [(x.prompt_version, x.created_at, x) for x in inference_results]

    # Filter for the requested prompt_version, if provided
    if req.prompt_version:
        filtered = [x for x in prompt_versions if x[0] == req.prompt_version]
    else:
        filtered = prompt_versions

    if not filtered:
        # handle no results
        return {"error": "No prompt results found."}

    # Select the entry with the latest created_at timestamp
    latest = max(filtered, key=lambda x: x[1])
    latest_prompt_result = latest[2]

    print(f"Detecting tables via output from prompt: {latest_prompt_result.prompt}\n\n----\nVersion: {latest_prompt_result.prompt_version}")
    table_structures = json.loads(latest_prompt_result.response)
    # return table_structures

    if node:
        node_stage_paths = json.loads(node[0].stage_paths_json)
        abs_path = node_stage_paths.get(str(req.stage), {}).get('abs_path', None)
        tbl_header_extractor = TableHeaderExtractor(abs_path)
        flat_list, y_meta = tbl_header_extractor.process(table_structures)
        return flat_list
    else:
        return {}