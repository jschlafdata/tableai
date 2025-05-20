from fastapi import (
    APIRouter,
    Query, 
    Depends,
    HTTPException,
    Request
)

import json
from typing import List, Any, Dict, Union
import uuid
from pathlib import Path
from pydantic import BaseModel

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

class ColumnHierarchy(BaseModel):
    level0: str
    # Accept arbitrary additional sub-levels (Pydantic v2 style)
    model_config = {"extra": "allow"}

class TableCountWithColumnsDict(BaseModel):
    number_of_tables: int
    headers_heirarchy: bool
    columns: Dict[str, List[Union[str, dict]]]


@router.post("/find_table_columns")
def get_extraction(
    req: PdfVisionModelRequest,
    request_obj: Request,
    api_service: 'APIService' = Depends(ensure_initialized)
):
    db = api_service.db
    node = db.run_op(FileNodeRecord, "get", filter_by={'uuid': req.file_id})
    if node:
        node_stage_paths = json.loads(node[0].stage_paths_json)
        abs_path = node_stage_paths.get(str(req.stage), {}).get('abs_path', None)
        if abs_path:
            pdf_model = PDFModel(path=Path(abs_path), source=Source.LOCAL)
            pdf_model.set_limit(req.page_limit)

            # Use the received request model directly
            client = VisionInferenceClient(
                username=api_service.service_config.ollama_api_user,
                password=api_service.service_config.ollama_api_key,
            )
            results = client.process_pdf(req, pdf_model, TableCountWithColumnsDict)
            return results
    else:
        return {}


    # db = api_service.db
    # node = db.run_op(FileNodeRecord, "get", filter_by={'uuid': req.file_id})
    # if node:
    #     node_stage_paths = json.loads(node[0].stage_paths_json)
    #     abs_path = node_stage_paths.get(str(req.stage), {}).get('abs_path', None)
    #     mount_path = node_stage_paths.get(str(req.stage), {}).get('mount_path', None)

    #     client = VisionInferenceClient(
    #         model_library={
    #             'mistralVL': ('mistral-small3.1:24b', 'ollama-medium-ollama.portal-ai.tools'),
    #             'llamaVL':   ('llama3.2-vision', 'ollama-embeddings-ollama.portal-ai.tools')
    #         },
    #         model_choice='mistralVL',
    #         username=api_service.service_config.ollama_api_user,
    #         password=api_service.service_config.ollama_api_key,
    #         page_limit=None
    #     )

    #     print(f"CLASSIFICAITON: {req.classification_label}")

    #     results, timings = client.process_pdf(
    #         pdf_path=abs_path,
    #         prompt=TABLE_STRUCTURE_PROMPT_V5,
    #     )

    #     inference_model = LLMInferenceTableStructures(
    #         run_uuid=str(uuid.uuid4()),
    #         uuid=req.file_id,
    #         stage=req.stage,
    #         classification_label=req.classification_label,
    #         prompt=TABLE_STRUCTURE_PROMPT_V5, 
    #         prompt_version=5, 
    #         prompt_name="TABLE_STRUCTURE_PROMPT", 
    #         response=json.dumps(results)
    #     )

    #     db.run_op(
    #         LLMInferenceTableStructures, 
    #         operation="merge", 
    #         data=inference_model
    #     )