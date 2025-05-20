import os
from dataclasses import dataclass, field, asdict, fields
from typing import List, Union
import uuid
from pathlib import Path
import fitz
import json
from typing import Dict, List, Optional
from sqlmodel import SQLModel, Field

### Application imports ###
from tableai.core.time import TimeModel, UtcTimeUtil
from tableai.data_loaders.files import FileReader
from backend.models.backend import FileNodeRecord, PDFClassifications, DropboxSyncRecord
from tableai.core.serialize import _deserialize_field, _serialize_field, safe_json
from tableai.core.paths import PathManager
### ------------------- ###

@dataclass(kw_only=True)
class DirectoryFileNode(TimeModel):
    """
    Represents a single file discovered by traversing a directory tree.
    Inherits timestamp handling from TimeModel (created_at, updated_at, last_sync).
    """

    uuid: str
    source: str
    source_id: str
    source_name: str
    source_file_name: str
    source_type: str
    local_path: Path
    input_dir: Path
    output_dir: Path
    file_name: str
    file_type: str
    name: str
    auto_label: str = ''
    current_stage: int = 0
    recovered: bool = False
    corrupted: bool = False
    stage_paths: dict = field(default_factory=dict)
    extraction_metadata: dict = field(default_factory=dict)
    source_directories: Optional[List] = None
    source_categories: dict = field(default_factory=dict)
    source_metadata: dict = field(default_factory=dict)
    completed_stages: list = field(default_factory=list)
    base_llm_config: dict = field(default_factory=dict)

    # Attributes merged from PdfFuncs
    error_message: dict = field(default_factory=dict)
    meta_tag: str = ""  # meta_tag from PDF metadata

    def __post_init__(self):
        super().__post_init__()
        # Any additional initialization logic here
    
    @property
    def doc(self) -> Optional[fitz.Document]:
        """
        If you want to keep a property that returns a PyMuPDF doc,
        you can load it on the fly. Alternatively, you might store
        it after registration. Here we attempt to load if needed.
        """
        try:
            return fitz.open(self.current_pdf_path)
        except:
            return None

    @property
    def meta(self):
        return self.to_dict()

    @property
    def current_pdf_path(self) -> str:
        """
        Returns the absolute path of the PDF for the current stage.
        """
        if self.current_stage in self.stage_paths:
            return self.stage_paths[self.current_stage]["abs_path"]
        # Fallback to local path if no stage path is found
        return str(self.local_path)

    def store_metadata(self, stage_number, data: Dict, db=None):
        stage = f"stage{stage_number}"
        self.extraction_metadata[stage] = {
            **self.stage_paths.get(stage_number, {}), 
            **data
        }
        if not self.corrupted:
            self.completed_stages.append(stage_number)
            self.completed_stages = list(set(self.completed_stages))

        if db:
            # Persist to DB
            db.run_op(FileNodeRecord, operation="merge", data=self.to_record())

    def load_pdf(self):
        """
        Loads the PDF from the current stage path (or local path),
        returning either a fitz.Document or a dict (if corrupted).
        """
        abs_path = self.stage_paths[self.current_stage]["abs_path"]
        recovery_path = self.stage_paths[self.current_stage]["recovery_path"]
        return FileReader.pdf(abs_path, recovery_path=recovery_path)

    def add_stage(self, stage_number: int):
        stage_file_path = PathManager.get_stage_pdf_path(self.uuid, self.file_type, stage_number)
        recovery_path = PathManager.get_recovery_path(self.uuid, self.file_type, stage_number)

        # Ensure stage and recovery dirs exist
        stage_file_path.parent.mkdir(parents=True, exist_ok=True)
        recovery_path.parent.mkdir(parents=True, exist_ok=True)

        file_name = f"{self.uuid}.{self.file_type}"

        self.stage_paths[stage_number] = {
            "abs_path": str(stage_file_path),
            "rel_path": PathManager.get_rel_path(stage_file_path),
            "recovery_path": str(recovery_path),
            "mount_path": str(os.path.join(PathManager.get_fastapi_mount_path(stage_number), file_name))
        }

        return str(stage_file_path)

    @property
    def fastapi_url(self):
        base = PathManager.get_fastapi_mount_path(self.current_stage, self.recovered)
        return str(Path(base) / Path(f"{self.uuid}.{self.file_type}"))

    @classmethod
    def from_sync_record(cls, record: DropboxSyncRecord, source: str = 'dropbox', current_stage=0) -> "DirectoryFileNode":
        metadata = json.loads(record.metadata_json)
        llm_config_dir = PathManager.llm_dir
        llm_config_file = llm_config_dir / "pdf_schema.yml"
        base_llm_config = FileReader.yaml(llm_config_file)

        return cls(
            uuid=record.dropbox_safe_id,
            current_stage=current_stage,
            source=source,
            source_id=record.dropbox_id,
            source_name=Path(metadata.get("path_display")).stem,
            source_file_name=metadata.get("file_name"),
            source_type=metadata.get("type"),
            local_path=Path(record.local_path),
            input_dir=Path(record.local_path).parent,
            output_dir=Path(PathManager.output_dir), 
            file_name=metadata.get("file_name"),
            file_type=Path(metadata.get("file_name")).suffix.lstrip('.'),
            name=metadata.get("name"),
            source_directories=metadata.get("directories"),
            source_categories=metadata.get("path_categories", {}),
            auto_label=metadata.get("auto_label", ''),
            source_metadata=metadata,
            base_llm_config=base_llm_config
        )

    def register(self, db):
        """
        Register a newly discovered file (stage 0) in the database, 
        attempting to load its PDF, setting corruption flags, etc.
        """
        if self.current_stage == 0 and self.source_type == 'file':
            self.add_stage(0)
            # Attempt to load the PDF, track if it is corrupted
            self.register_pdf()

            # If not corrupted, fetch metadata
            metadata = self.fetch_pdf_metadata()
            metadata['llm_table_configs'] = self.base_llm_config
            metadata['fastapi_url'] = self.fastapi_url

            self.store_metadata(0, metadata)
            # If the file is not corrupted, mark stage 0 as complete
            if not self.corrupted:
                self.completed_stages.append(0)

            # Persist to DB
            db.run_op(FileNodeRecord, operation="merge", data=self.to_record())

            # Insert classification info into DB
            db.run_op(
                PDFClassifications,
                operation="merge",
                data={
                    "file_id": self.uuid,
                    "classification": self.meta_tag
                }
            )

    def register_pdf(self):
        """
        Logic that attempts to load the PDF and sets self.corrupted/self.recovered 
        as appropriate. Merged from PdfFuncs.register.
        """
        # For stage/current_stage, get the recovery path:
        recovery_path = self.stage_paths[self.current_stage]["recovery_path"]
        pdf_result = FileReader.pdf(str(self.local_path), recovery_path)

        if isinstance(pdf_result, dict):
            # If it's a dict, the file is corrupted/unreadable
            self.corrupted = True
            self.error_message = pdf_result
            return

        if isinstance(pdf_result, fitz.Document):
            # If it's a valid doc, check if a recovery file was created
            self.recovered = Path(recovery_path).exists()
            self.corrupted = False
            pdf_result.close()

    def fetch_pdf_metadata(self):
        """
        Fetch high-level PDF metadata (pages, coords, meta_tag, etc.).
        If corrupted, return details from error_message.
        """
        if self.corrupted:
            # Return minimal info about the corruption
            return {
                "recovery_path": self.stage_paths[self.current_stage]["recovery_path"], 
                "corrupt_flag": self.corrupted,
                "recovered": self.recovered,
                "error_message": self.error_message
            }
        else:
            # doc is passed by the decorator (fitzAwrap)
            md = self.doc.metadata  # merged Info + XMP
            meta_fields = {
                key: md.get(key, None)
                for key in ["creator", "title", "author", "subject", "keywords"]
            }
            pdf_trailer = self.doc.pdf_trailer()
            if isinstance(pdf_trailer, dict) and "Root" in pdf_trailer:
                version = pdf_trailer.get("Version", None)
            else:
                version = None

            # Build a unique-ish hash of core PDF metadata (optional usage)
            meta_hash = '|'.join(
                f"{k}|{''.join(v.split())}" for k, v in meta_fields.items() if v
            ).strip()
            self.meta_tag = meta_hash

            coords = []
            pages = len(self.doc)
            for i, page in enumerate(self.doc):
                rect = page.rect
                coords.append({
                    i: {
                        "page_box": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "width": rect.width,
                        "height": rect.height
                    }
                })

            width = self.doc[0].rect.width if pages > 0 else None
            height = self.doc[0].rect.height if pages > 0 else None

            return {
                "pages": pages,
                "coords": coords,
                "meta_tag": self.meta_tag if self.meta_tag else 'ALL',
                "recovered": self.recovered,
                "recovery_path": self.stage_paths[self.current_stage]["recovery_path"],
                "pdf_version": version,
                "width": width,
                "height": height
            }

    def to_dict(self) -> dict:
        base = {
            "uuid": self.uuid,
            "source": self.source,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_file_name": self.source_file_name,
            "source_type": self.source_type,
            "local_path": str(self.local_path),
            "file_name": self.file_name,
            "file_type": self.file_type,
            "name": self.name,
            "stage_paths": self.stage_paths,
            "extraction_metadata": self.extraction_metadata,
            "source_directories": self.source_directories,
            "source_categories": self.source_categories,
            "source_metadata": self.source_metadata,
            "completed_stages": self.completed_stages,
            "base_llm_config": self.base_llm_config,
            "recovered": self.recovered,
            "corrupted": self.corrupted,
            "error_message": self.error_message,
            "meta_tag": self.meta_tag,
            "auto_label": self.auto_label
        }
        return {**base, **super().to_dict()}

    def to_record(self) -> FileNodeRecord:
        raw = asdict(self)
        record_kwargs = {}

        for f in FileNodeRecord.__fields__:
            if f.endswith("_json"):
                key = f.replace("_json", "")
                record_kwargs[f] = json.dumps(raw.get(key, {}))
            elif f in raw:
                record_kwargs[f] = _serialize_field(raw[f])
            else:
                record_kwargs[f] = None

        # Manually include time fields if present
        for time_field in ["created_at", "updated_at", "last_sync"]:
            if time_field in raw:
                record_kwargs[time_field] = raw[time_field]

        return FileNodeRecord(**record_kwargs)

    @classmethod
    def from_record(cls, record: FileNodeRecord) -> "DirectoryFileNode":

        if record:
            record_dict = record.dict()
            init_kwargs = {}

            for f in fields(cls):
                val = record_dict.get(f.name)
                if val is None and f.name + "_json" in record_dict:
                    val = _deserialize_field(f.name, record_dict[f.name + "_json"])
                else:
                    val = _deserialize_field(f.name, val)

                # Restore integer keys for stage_paths
                if f.name == "stage_paths" and isinstance(val, dict):
                    try:
                        val = {int(k): v for k, v in val.items()}
                    except (ValueError, TypeError):
                        pass  # Skip conversion if keys can't be casted

                init_kwargs[f.name] = val
            
            if "time_util" not in init_kwargs or init_kwargs["time_util"] is None:
                init_kwargs["time_util"] = UtcTimeUtil()

            return cls(**init_kwargs)
        else:
            return None