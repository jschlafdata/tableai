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

from api.models.requests import (
    PDFExtractRequest, 
    LLMTableStructureRequest
)

from tableai.inference.prompts.table_structure import (
    TABLE_STRUCTURE_PROMPT,
    TABLE_STRUCTURE_PROMPT_V2
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
     render_api, 
     search_normalized_text,
     normalize_word,
     normalize_recurring_text, 
     patterns, 
     try_convert_float, 
     identify_currency_symbols,
     try_convert_date,
     PDFTools
 )


@router.post("/doc_query")
def get_extraction(
        req: PDFExtractRequest,
        request_obj: Request,
        api_service: 'APIService' = Depends(ensure_initialized)
    ):
    db = api_service.db
    node = db.run_op(FileNodeRecord, "get", filter_by={'uuid': req.file_id})
    if node:
        node_stage_paths = json.loads(node[0].stage_paths_json)
        abs_path = node_stage_paths.get(str(req.stage), {}).get('abs_path', None)
        mount_path = node_stage_paths.get(str(req.stage), {}).get('mount_path', None)
        if abs_path:
            doc = fitz.open(abs_path)
            index = FitzTextIndex.from_document(doc)
            all_pages_text_index = index.query(**{"blocks[*].lines[*].spans[*].text": "*"}, restrict=["text", "font", "bbox"])
            line_index = LineTextIndex(all_pages_text_index, page_metadata=index.page_metadata)
            
            
            page_patterns = line_index.query(
                key="text",
                transform=lambda rows: render_api(
                    query_label="pageIndicators",
                    description="find common page spans",
                    pdf_metadata={},
                    include=[]
                )(filterby(
                    lambda t: t and t.lower() == "page xx of xx",
                    field="normalized_value",
                    test=bool
                )(rows))
            )

            floats = line_index.query(
                key="text",
                transform=lambda rows: render_api(
                    query_label="floats",
                    description="Return all blocks that successfully evaluate as Floats or Ints",
                    pdf_metadata={}
                )(filterby(try_convert_float, "value", test=lambda x: x is not None)(rows))
            )

            dates = line_index.query(
                key="text",
                transform=lambda rows: render_api(
                    query_label="dates",
                    description="Return all date fields.",
                    pdf_metadata={}
                )(filterby(try_convert_date, "value", test=bool)(rows))
            )

            toll_free = line_index.query(
                key="text",
                transform=lambda rows: render_api(
                    query_label="toll_free",
                    description="Return all toll-free patterns.",
                    pdf_metadata={}
                )(filterby(lambda t: patterns(t, pattern_name="toll_free"), field="value", test=bool)(rows))
            )

            return [page_patterns, floats, dates, toll_free]
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
            username=api_service.service_config.ollama_api_key,
            password=api_service.service_config.ollama_api_key
        )

        print(f"CLASSIFICAITON: {req.classification_label}")

        results, timings = client.process_pdf(
            pdf_path=abs_path,
            prompt=TABLE_STRUCTURE_PROMPT_V2,
        )

        inference_model = LLMInferenceTableStructures(
            run_uuid=str(uuid.uuid4()),
            uuid=req.file_id,
            stage=req.stage,
            classification_label=req.classification_label,
            prompt=TABLE_STRUCTURE_PROMPT_V2, 
            response=json.dumps(results)
        )

        db.run_op(
            PDFClassifications, 
            operation="merge", 
            data=inference_model
        )
