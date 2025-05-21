# backend/models/backend.py

from sqlmodel import SQLModel, Field
from datetime import datetime, timezone
from typing import Optional


class DropboxSyncRecord(SQLModel, table=True):
    __tablename__ = "dropbox_sync_record"

    dropbox_id: str = Field(primary_key=True)
    dropbox_safe_id: str
    path_lower: str
    local_path: str
    size: int
    server_modified: datetime
    synced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata_json: str = Field(default="{}")


class BadPDFSync(SQLModel, table=True):
    __tablename__ = "bad_pdf_sync"

    file_id: str = Field(primary_key=True)
    error_message: str


class DropboxSyncError(SQLModel, table=True):
    __tablename__ = "dropbox_sync_error"

    file_id: str = Field(primary_key=True)
    error: str
    timestamp: datetime


class PDFClassifications(SQLModel, table=True):
    __tablename__ = "pdf_classifications"

    file_id: str = Field(primary_key=True)
    classification: str


class ClassificationLabel(SQLModel, table=True):
    __tablename__ = "classification_label"

    classification: str = Field(primary_key=True)
    label: str


class FileExtractionResult(SQLModel, table=True):
    __tablename__ = "file_extraction_result"

    file_id: str = Field(primary_key=True)
    classification_label: str
    extracted_json: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LLMInferenceTableStructures(SQLModel, table=True):
    __tablename__ = "llm_inference_table_structures"

    run_uuid: str = Field(primary_key=True, nullable=False)
    uuid: str
    stage: int
    classification_label: str
    prompt: str
    prompt_version: Optional[int] = None
    prompt_name: str = ''
    response: str = Field(default="{}")
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class FileNodeRecord(SQLModel, table=True):
    __tablename__ = "file_node_record"

    uuid: str = Field(primary_key=True)
    source: str
    source_id: str
    source_name: str
    source_file_name: str
    source_type: str
    local_path: str
    input_dir: str
    output_dir: str
    file_name: str
    file_type: str
    name: str
    auto_label: Optional[str] = ''

    completed_stages_json: str = Field(default="{}")
    stage_paths_json: str = Field(default="{}")
    extraction_metadata_json: str = Field(default="{}")
    source_directories_json: str = Field(default="[]")
    source_categories_json: str = Field(default="{}")
    source_metadata_json: str = Field(default="{}")

    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_sync: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


MODEL_REGISTRY = {
    "BadPDFSync": BadPDFSync,
    "DropboxSyncRecord": DropboxSyncRecord,
    "PDFClassifications": PDFClassifications,
    "ClassificationLabel": ClassificationLabel,
    "FileExtractionResult": FileExtractionResult,
    "LLMInferenceTableStructures": LLMInferenceTableStructures,
    "FileNodeRecord": FileNodeRecord,
}
