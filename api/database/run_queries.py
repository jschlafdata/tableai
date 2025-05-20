from fastapi import (
    APIRouter,
    Query, 
    Depends,
    HTTPException
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

router = APIRouter()

@router.get("/records")
async def list_records(
    included_models: List[str] = Query(default=["FileExtractionResult", "DropboxSyncRecord"]),
    merge: bool = Query(default=False),
    include_metadata: bool = Query(default=False),
    api_service: 'APIService' = Depends(ensure_initialized)
) -> Any:
    
    db = api_service.db
    combined = merge_dicts(
        await get_dropbox_sync(db),
        await get_file_node_records(db)
    )
    result_records = []
    for k,v in combined.items():
        result_records.append({'file_id': k, **v})
    return result_records

def merge_dicts(*dicts: dict) -> dict:
    merged = {}
    for d in dicts:
        for k, v in d.items():
            if k in merged and isinstance(v, dict) and isinstance(merged[k], dict):
                merged[k].update(v)
            else:
                merged[k] = v
    return merged

async def get_file_extraction_result(db):
    extract_records = db.run_op(FileExtractionResult, operation="get")
    return {
        r.file_id: {
            "classification_label": r.classification_label,
            "updated_at": r.updated_at
        }
        for r in extract_records
    }

async def get_dropbox_sync(db):
    records = db.run_op(DropboxSyncRecord, operation="get")
    if records:
        return {
                r.dropbox_safe_id: {**json.loads(r.metadata_json)}
                for r in records
            }
    else:
        return {}

async def get_file_node_records(db):
    records = db.run_op(FileNodeRecord, operation="get")
    dbx_register_service = DropboxRegisterService(db)
    if records:
        return  {
            r.uuid: {
                'completed_stages': json.loads(r.completed_stages_json), 
                'stage_paths': json.loads(r.stage_paths_json),
                'classification': dbx_register_service._get_node_label(DirectoryFileNode.from_record(r))
                } for r in records
            }
    else:
        return {}

@router.get("/classify/existing_labels")
async def run_extraction(
    api_service: 'APIService' = Depends(ensure_initialized)
): 
    records = api_service.db.run_op(ClassificationLabel, operation="get")
    return records

@router.get("/classify/labels")
async def run_extraction(
    api_service: 'APIService' = Depends(ensure_initialized)
): 
    query = """select b.* 
                from 
                dropboxsyncrecord a
                join pdfclassifications b 
                on a.dropbox_safe_id = b.file_id"""
    return api_service.db.run_raw(query)


@router.get("/classify/samples")
def get_classification_samples(
    limit: int = 10,
    api_service: 'APIService' = Depends(ensure_initialized)
):
    """
    Return sample file_ids grouped by classification.
    Useful for UI to preview documents before assigning labels.
    """
    query = """select b.* 
            from 
            dropboxsyncrecord a
            join pdfclassifications b 
            on a.dropbox_safe_id = b.file_id"""
    records = api_service.db.run_raw(query)
    grouped = {}
    for r in records:
        if not r.get('classification'):
            continue
        grouped.setdefault(r['classification'], []).append(r['file_id'])

    # Limit to N samples per group
    return {k: v[:limit] for k, v in grouped.items()}


@router.get("/stage0_summary", response_model=List[Dict])
def get_stage0_summary(
    api_service: 'APIService' = Depends(ensure_initialized)
    ):
    query = """
    SELECT 
        uuid as file_id, 
        REPLACE(uuid, 'id_', 'id:') AS dropbox_id,
        json_extract(extraction_metadata_json, '$.stage0.recovery_path') as recovery_path,
        json_extract(extraction_metadata_json, '$.stage0.meta_tag') as classification_id,
        json_extract(extraction_metadata_json, '$.stage0.recovered') as recovered_pdf,
        json_extract(extraction_metadata_json, '$.stage0.abs_path') as stage0_path,
        json_extract(extraction_metadata_json, '$.stage0') as stage0_metadata,
        json_extract(completed_stages_json, '$[0]') as stage0_complete,
        lower((
            SELECT GROUP_CONCAT(value, '/')
            FROM json_each(FileNodeRecord.source_directories_json)
            ) || '/' || file_name) AS path_lower,
        *
    FROM FileNodeRecord
    WHERE source_type = 'file'
    """
    return api_service.db.run_raw(query)


@router.get("/inference_results/table_structure")
def get_inference_results(
    api_service: 'APIService' = Depends(ensure_initialized)
):
    db = api_service.db
    records = db.run_op(LLMInferenceTableStructures, operation="get")
    return records

@router.get("/extract_metadata")
def list_synced_records(
    api_service: 'APIService' = Depends(ensure_initialized)
):
    db = api_service.db
    records = db.run_op(DropboxSyncRecord, operation="get")
    core_records = [
        {**json.loads(r.metadata_json), **{'dropbox_safe_id': r.dropbox_safe_id}} for r in records
    ]
    extract_records = db.run_op(FileExtractionResult, operation="get")
    extract_meta = [
        {
            "file_id": r.file_id,
            "label": r.classification_label,
            "updated_at": r.updated_at
        }
        for r in extract_records
    ]
    extract_metadata = {v['file_id']: v for v in extract_meta}
    core_metadata = {v['dropbox_safe_id']:v for v in core_records if v['dropbox_safe_id'] in extract_metadata.keys()}
    combined_meta = {k: {**extract_metadata[k], **core_metadata[k]} for k in extract_metadata.keys()}
    return combined_meta


@router.get("/extractions/{file_id}")
def get_extraction(
    file_id: str,
    api_service: 'APIService' = Depends(ensure_initialized)
    ):
    db = api_service.db
    print(file_id)
    record = db.run_op(
        FileExtractionResult,
        operation="get",
        filter_by={"file_id": file_id}
    )
    print(type(record))
    if not record:
        raise HTTPException(status_code=404, detail="Not found")
    records = json.loads(record[0].extracted_json)
    class_label = {'classification_label': record[0].classification_label}
    # return records
    recs = {**records, **class_label}
    recs['metadata'] = recs['extraction_metadata']
    return [recs]


@router.get("/filters/metadata", response_model=Dict[str, Any])
def get_filter_metadata(
    api_service: 'APIService' = Depends(ensure_initialized)
):
    db = api_service.db
    filters = {
        "subDirectories": set(),
        "categoryFilters": {},
        "availableClassifications": set(),
    }

    enriched_files = []

    # 1. Load file records
    records = db.run_op(FileNodeRecord, "get")

    classifications = db.run_op(PDFClassifications, "get", columns=["file_id", "classification"])
    _labels = db.run_op(ClassificationLabel, "get")
    label_map = {l.classification: l.label for l in _labels}
    classification_dict = {c.file_id: label_map.get(c.classification, 'NONE') for c in classifications if c.classification != ''}
    for record in records:
        try:
            directories = json.loads(record.source_directories_json)
        except Exception:
            directories = []

        try:
            categories = json.loads(record.source_categories_json)
        except Exception:
            categories = {}

        try:
            extraction_metadata = json.loads(record.extraction_metadata_json)
        except Exception:
            extraction_metadata = {}

        try:
            completed_stages = json.loads(record.completed_stages_json)
        except Exception:
            completed_stages = []

        try:
            stage_paths = json.loads(record.stage_paths_json)
        except Exception:
            stage_paths = {}

        # Update filter sets
        if directories:
            filters["subDirectories"].add(directories[0])

        for key, val in categories.items():
            if key not in filters["categoryFilters"]:
                filters["categoryFilters"][key] = set()
            filters["categoryFilters"][key].add(val)

        # Construct enriched file record
        enriched_files.append({
            "uuid": record.uuid,
            "source": record.source,
            "source_type": record.source_type,
            "file_name": record.file_name,
            "dropbox_safe_id": record.uuid,
            "local_path": record.local_path,
            "input_dir": record.input_dir,
            "output_dir": record.output_dir,
            "path_categories": categories,
            "directories": directories,
            "completed_stages": completed_stages,
            "stage_paths": stage_paths,
            "classification": classification_dict.get(record.uuid, 'all'),
            "fastapi_url": extraction_metadata.get("stage0", {}).get("fastapi_url"),
            "pdf_metadata": extraction_metadata.get("pdf_metadata", {}),
            "stage0": extraction_metadata.get("stage0", {}),
            "stage1": extraction_metadata.get("stage1", {}),
            "stage2": extraction_metadata.get("stage2", {}),
            "stage3": extraction_metadata.get("stage3", {})
        })

    # 2. Add classifications from both sources
    filters["availableClassifications"].update(c.classification for c in classifications)

    labels = db.run_op(ClassificationLabel, "get", columns=["classification"])
    filters["availableClassifications"].update(l.classification for l in labels)

    # 3. Return all compiled metadata and options
    return {
        "files": enriched_files,
        "filters": {
            "subDirectory": "all",
            "searchQuery": "",
            "fileIdQuery": "",
            "selectedFileIndex": -1,
            "selectedStage": "stage0",
            "categoryFilters": {
                key: sorted(values) for key, values in filters["categoryFilters"].items()
            },
            "classificationFilter": "all",
        },
        "uiOptions": {
            "availableClassifications": sorted(filters["availableClassifications"]),
            "showClassificationFilter": True
        },
        "metadataOptions": {
            "subDirectories": sorted(filters["subDirectories"]),
            "categoryKeys": list(filters["categoryFilters"].keys())
        }
    }