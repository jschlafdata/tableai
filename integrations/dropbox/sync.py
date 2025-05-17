from dropbox.files import FileMetadata, FolderMetadata
import os, json, re
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlmodel import SQLModel, Field, create_engine, Session, select
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import os
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor

### Aplication imports ###
from backend.models.backend import DropboxSyncRecord, FileNodeRecord, DropboxSyncError, FileExtractionResult
from integrations.dropbox.auth import DropboxAuth
from integrations.dropbox.files import DropboxMetadataConverter
### ------------------ ###

BATCH_SIZE = 100

class DropboxSync(DropboxAuth):
    def __init__(self, db, api_service):
        super().__init__()
        self.db = db
        self.api = api_service
        self.dbx_client = self.get_client()
        self.PUBLIC_DIR = Path(__file__).parent.parent.parent / ".synced" / "dropbox"
        self.synced_files: List[str] = []
        self.records_to_store: List[SQLModel] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.api_semaphore = asyncio.Semaphore(5)

    def flush_records(self):
        if len(self.records_to_store) >= BATCH_SIZE:
            print(f"[flush] Writing {len(self.records_to_store)} records to DB")
            self.db.run_op(DropboxSyncRecord, operation="merge_many", data=self.records_to_store)
            self.records_to_store.clear()
            if self.api:
                self.api._cache_data("sync_partial", self.synced_files[:])

    async def safe_api_call(self, func, *args):
        async with self.api_semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, func, *args)

    async def download_file(self, dbx_path: str, local_dir: Path):
        os.makedirs(self.PUBLIC_DIR, exist_ok=True)

        ignore_match = False
        if self.ignore:
            try:
                ignore_match = re.search(self.ignore, dbx_path) is not None
                if ignore_match:
                    print(f"ignoring path: {dbx_path}\n regex: {self.ignore}")
            except re.error as e:
                print(f"Invalid regex pattern '{self.ignore}': {e}")
        if not ignore_match:
            try:
                metadata, result = await self.safe_api_call(self.dbx_client.files_download, dbx_path)
            except Exception as e:
                self.log_sync_error(dbx_path, str(e))
                return
            
            dropbox_id = metadata.id
            extension = Path(metadata.name).suffix.lstrip('.') or 'bin'

            if (not self.file_types or extension in self.file_types) and not ignore_match:
                filename = f"{dropbox_id}.{extension}"
                target_dir = self.PUBLIC_DIR / Path(extension)
                target_dir.mkdir(parents=True, exist_ok=True)
                filename = filename.replace(":", "_")
                target = target_dir / filename

                with open(target, "wb") as f:
                    f.write(result.content)

                rel = str(target.relative_to(self.PUBLIC_DIR))
                self.synced_files.append(rel)

                record = self.create_record(metadata, str(target))
                self.records_to_store.append(record)
                self.flush_records()

    def parse_path_categories(self, path_lower: str) -> dict:
        categories = {}

        pattern_parts = self.path_categories.strip("/").split("/")
        path_parts = path_lower.strip("/").split("/")

        # Slide over the path to find a matching segment of same length
        for i in range(len(path_parts) - len(pattern_parts) + 1):
            window = path_parts[i:i + len(pattern_parts)]
            match = {}
            for pat_part, actual_part in zip(pattern_parts, window):
                if pat_part.startswith("{") and pat_part.endswith("}"):
                    var_name = pat_part[1:-1]
                    match[var_name] = actual_part
                elif pat_part != actual_part:
                    break  # not a match, continue sliding
            else:
                # All parts matched
                return match

        return categories

    def create_record(self, metadata_obj, local_path: str) -> SQLModel:
        entry_dict = DropboxMetadataConverter.to_dict(metadata_obj)
        dropbox_id = entry_dict["dropbox_id"]
        path_lower = entry_dict.get("path_lower", "")
        server_modified_str = entry_dict.get("server_modified")
        server_modified_dt = datetime.fromisoformat(server_modified_str) if server_modified_str else datetime.utcnow()
        size_val = entry_dict.get("size", 0)
        path_parts = Path(path_lower).parts
        directories = [x for x in path_parts[:-1] if x != '/']
        file_name = path_parts[-1] if path_parts else ""
        name = Path(file_name).stem

        # self.path_categories THIS HAS BEEN ADDED

        path_categories = {}
        if self.path_categories:
            # parse_path_categories is the helper we defined above
            path_categories = self.parse_path_categories(path_lower)

        extra_meta = {
            'directories': directories,
            'name': name,
            'file_name': file_name,
            'path_categories': path_categories,
            'auto_label': self.auto_label or ''
        }

        entry_dict = {**extra_meta, **entry_dict}
        return DropboxSyncRecord(
            dropbox_id=dropbox_id,
            dropbox_safe_id=dropbox_id.replace(":", "_"), 
            path_lower=path_lower,
            local_path=local_path,
            size=size_val,
            server_modified=server_modified_dt,
            synced_at=datetime.utcnow(),
            metadata_json=json.dumps(entry_dict, default=str),
        )

    async def download_folder(self, dbx_folder: str, local_folder: Path):
        folder_meta = await asyncio.to_thread(self.dbx_client.files_get_metadata, dbx_folder)
        self.records_to_store.append(self.create_record(folder_meta, str(local_folder)))

        resp = await asyncio.to_thread(self.dbx_client.files_list_folder, dbx_folder)
        tasks = []

        for entry in resp.entries:
            if isinstance(entry, FolderMetadata):
                subname = entry.name
                tasks.append(self.download_folder(entry.path_lower, local_folder / subname))
            elif isinstance(entry, FileMetadata):
                existing = self.db.run_op(DropboxSyncRecord, operation="get", filter_by={"dropbox_id": entry.id})
                if (
                    not existing
                    or existing[0].server_modified < entry.server_modified
                    or self.force_refresh
                ):
                    tasks.append(self.download_file(entry.path_lower, local_folder))
        await asyncio.gather(*tasks)

    def log_sync_error(self, file_id: str, error: str):
        # local import to avoid circular
        error_entry = DropboxSyncError(
            file_id=file_id,
            error=error,
            timestamp=datetime.utcnow()
        )
        self.db.run_op(DropboxSyncError, operation="merge", data=error_entry)

    async def run(self, sync_request):
        self.file_types = sync_request.file_types
        self.ignore = sync_request.ignore
        self.path_categories = sync_request.path_categories
        self.auto_label = sync_request.auto_label
        self.force_refresh = sync_request.force_refresh
        print(f"FORCING: {self.force_refresh}")
        print(f"IGNORING: {self.ignore}")
        await self._run_async(sync_request.paths)

    async def _run_async(self, paths: List[str]):
        tasks = []

        for p in paths:
            dbx_path = "" if p in ("", "/", "root") else "/" + p.lstrip("/")
            meta = await asyncio.to_thread(self.dbx_client.files_get_metadata, dbx_path)

            if isinstance(meta, FolderMetadata):
                parts = [seg for seg in p.split("/") if seg]
                local_base = self.PUBLIC_DIR.joinpath(*parts) if parts else self.PUBLIC_DIR
                tasks.append(self.download_folder(dbx_path, local_base))
            else:
                existing = self.db.run_op(DropboxSyncRecord, operation="get", filter_by={"dropbox_id": meta.id})
                if not existing or existing[0].server_modified < meta.server_modified or self.force_refresh == True:
                    parent_parts = [seg for seg in Path(p).parent.parts if seg]
                    local_parent = self.PUBLIC_DIR.joinpath(*parent_parts)
                    tasks.append(self.download_file(dbx_path, local_parent))

        await asyncio.gather(*tasks)

        if self.records_to_store:
            self.db.run_op(DropboxSyncRecord, operation="merge_many", data=self.records_to_store)
            self.records_to_store.clear()


class DropboxMetadata(DropboxAuth):
    def __init__(self):
        super().__init__()
        self.dbx_client = self.get_client()

    def fetch(self, db_path):
        print(f"requesting: {db_path}")
        res = self.dbx_client.files_list_folder(db_path)
        # Process entries using the DropboxMetadataConverter
        out = []
        for entry in res.entries:
            # Convert the entry to a dictionary using our dataclass
            entry_dict = DropboxMetadataConverter.to_dict(entry)                
            # Add type field based on the entry class
            if isinstance(entry, FolderMetadata):
                entry_dict["type"] = "folder"
            elif isinstance(entry, FileMetadata):
                entry_dict["type"] = "file"
            if entry_dict.get("is_downloadable") == False:
                # Skip this entry
                continue
            out.append(entry_dict)
        return out


class Validate:
    def __init__(self, db: 'DBManager'):
        self.db = db

    def validate_local_files(self):
        records = self.db.run_op(DropboxSyncRecord, "get")
        removed_count = 0

        for record in records:
            if not os.path.exists(record.local_path):
                self.db.run_op(
                    DropboxSyncRecord,
                    "delete",
                    filter_by={"dropbox_id": record.dropbox_id}
                )

        records = self.db.run_op(FileNodeRecord, "get")
        removed_count = 0

        for record in records:
            if not os.path.exists(record.local_path):
                self.db.run_op(
                    FileNodeRecord,
                    "delete",
                    filter_by={"uuid": record.uuid}
                )
        
        records = self.db.run_op(FileExtractionResult, "get")
        removed_count = 0

        for record in records:
            extracted_json = json.loads(record.extracted_json)
            local_path = extracted_json.get('local_path')
            if not os.path.exists(local_path):
                self.db.run_op(
                    FileExtractionResult,
                    "delete",
                    filter_by={"file_id": record.file_id}
                )
                removed_count += 1

        print(f"Validation complete. {removed_count} stale record(s) removed from the database.")
    
    def remove_synced_item(self, dropbox_id: str, remove_local: bool = True):
        """
        Remove a synced file/folder from local disk (optional) and from DB.
        
        :param dropbox_id: The dropbox_id (primary key) of the record in DropboxSyncRecord.
        :param remove_local: If True, also remove the file/folder from local disk.
        """
        record = self.db.run_op(
            DropboxSyncRecord,
            operation="get",
            filter_by={"dropbox_id": dropbox_id}
        )
        if not record:
            print(f"No record found for dropbox_id='{dropbox_id}'. Nothing to remove.")
            return

        local_path = record.local_path
        if remove_local:
            # Clean up local disk
            if os.path.isfile(local_path):
                os.remove(local_path)
            elif os.path.isdir(local_path):
                shutil.rmtree(local_path)

        self.db.run_op(
            DropboxSyncRecord,
            operation="delete",
            filter_by={"dropbox_id": dropbox_id}
        )

        print(f"Removed record for dropbox_id='{dropbox_id}' from DB. "
              f"{'Local file/folder also deleted.' if remove_local else 'Local file/folder retained.'}")