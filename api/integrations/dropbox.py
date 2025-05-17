from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends, 
    HTTPException,
    Request,
    FastAPI
)
import asyncio
import uuid 
import logging 
from urllib.parse import unquote
import json
from typing import List, Dict, Any

from api.tasks.on_task_end import PostHooks
from api.models.requests import DropboxSyncRequest, DropboxRegisterRequest, DropboxProcessRequest
from api.models.tasks import TaskStatus, RateLimiter
from api.service.dependencies import get_db, ensure_initialized
from backend.models.backend import DropboxSyncRecord, FileExtractionResult, FileNodeRecord
from integrations.dropbox.sync import DropboxMetadata, DropboxSync, Validate
from integrations.dropbox.auth import DropboxAuth
from api.tasks.wrappers import task_runner
from api.service.manager import APIService
from tableai.extract.process import DropboxRegisterService

router = APIRouter()

class DropboxManager(APIService):
    def __init__(self, api_service: 'APIService', app: FastAPI = None):
        super().__init__(DB=api_service.db)
        self.db = api_service.db
        self.api = api_service
        self.app = app
        self.tasks = api_service.tasks
        self.logger = logging.getLogger("DropboxManager")
        self.dbx_client = DropboxAuth().get_client()

    @task_runner(cache_prefix="folders", result_key="folder_items", eager=True, force_refresh_arg="force_refresh")
    async def process_list_folder(self, task_id: str, path: str, force_refresh: bool = True):
        """
        Background task to list items in a Dropbox folder.
        Uses caching if not force_refresh.
        """
        # 2) Not cached or forced refresh → call Dropbox
        db_path = "" if path in ("", "/", "root") else f"/{path.lstrip('/')}"
        metadata_runner = DropboxMetadata()
        return metadata_runner.fetch(db_path=db_path)

    @task_runner(
            cache_prefix="sync", 
            result_key="synced_files", 
            eager=False, 
            post_run=[
                PostHooks.node_sync,
                PostHooks.mount_sync_folders
            ]
    )
    async def process_sync_items(self, task_id: str, sync_request: DropboxSyncRequest):
        validator = Validate(db=self.db)
        validator.validate_local_files()

        dbx_sync_engine = DropboxSync(db=self.db, api_service=self.api)
        await dbx_sync_engine.run(sync_request)

        dbx_register_service = DropboxRegisterService(instance=self.db)

        # Gather all "synced records" from your DB
        synced_records = self.db.run_op(DropboxSyncRecord, operation="get")

        # Filter by directories if provided
        filtered_records = []
        if sync_request.paths:
            for record in synced_records:
                # If any requested directory is in record.path_lower
                if any(d in record.path_lower for d in sync_request.paths):
                    print(f"registering path: {record.path_lower}")
                    filtered_records.append(record)

            # De-duplicate if both directories and file_ids are used
            filtered_records = list({r.dropbox_safe_id: r for r in filtered_records}.values())

            # Register them
            results = []
            if filtered_records:
                results = dbx_register_service.add(
                    node_list=filtered_records,
                    force=sync_request.force_refresh,
                    stage=0
                )
        return dbx_sync_engine.synced_files
    
    @task_runner(
        cache_prefix="register",
        result_key="registered_items",
        eager=False,
        force_refresh_arg="force"
    )
    async def process_register_items(self, task_id: str, req: DropboxRegisterRequest):
        """
        Background task to register items in the local DB,
        using the DropboxRegisterService logic.
        """

        dbx_register_service = DropboxRegisterService(instance=self.db)
        if req.force_refresh == True:
            # Gather all "synced records" from your DB
            synced_records = self.db.run_op(DropboxSyncRecord, operation="get")

            # Filter by directories if provided
            filtered_records = []
            if req.directories:
                dir_set = [d.lower() for d in req.directories]
                for record in synced_records:
                    # If any requested directory is in record.path_lower
                    if any(d in record.path_lower for d in dir_set):
                        filtered_records.append(record)

            # Filter by file_ids if provided
            if req.file_ids:
                filtered_records += [
                    r for r in synced_records
                    if r.dropbox_safe_id in req.file_ids
                ]

            # De-duplicate if both directories and file_ids are used
            filtered_records = list({r.dropbox_safe_id: r for r in filtered_records}.values())
            if filtered_records:
                results = dbx_register_service.add(
                    node_list=filtered_records,
                    force=req.force_refresh,
                    stage=req.stage
                )
        elif req.force_refresh == False:
            print(f"requested directories: {req.directories}")
            proced_file_nodes = self.db.run_op(FileNodeRecord, operation="get")
            results = [
                r for r in proced_file_nodes
                if any(
                    '/'.join(json.loads(r.source_directories_json)).startswith(l)
                    for l in req.directories
                )
            ]
            print(f"loaded_source_dirs: {len(results)}")

        if results:
            extraction_records = self.db.run_op(FileExtractionResult, operation="get")
            existing_ids = [node.file_id for node in extraction_records]
            for res in results:
                file_id = res.uuid
                # Only re-process if forced or not yet processed
                if req.force_refresh or file_id not in existing_ids:
                    dbx_register_service.process(file_id=file_id)

        return {}



@router.post("/sync")
async def sync_items(
    request: DropboxSyncRequest,
    request_obj: Request,
    api_service: 'APIService' = Depends(ensure_initialized)
) -> TaskStatus:
    
    print(f"FORCING: {request.force_refresh}")

    dbx_manager = DropboxManager(api_service, request_obj.app)
    
    # Call task function, it will return task_id
    task_id = await dbx_manager.process_sync_items(sync_request=request)

    return api_service.tasks[task_id]

@router.get("/records/{dropbox_id}")
def get_metadata_record(
    dropbox_id: str,
    api_service: 'APIService' = Depends(ensure_initialized)
):
    filename = unquote(dropbox_id)
    db = api_service.db
    results = db.run_op(
        DropboxSyncRecord,
        operation="get",
        filter_by={"dropbox_id": filename}
    )
    if not results:
        raise HTTPException(404, detail="Record not found")
    return json.loads(results[0].metadata_json)


@router.get("/folders/{path:path}")
async def list_folder(
    path: str,
    force_refresh: bool = False,
    api_service: 'APIService' = Depends(ensure_initialized)
):
    dbx_manager = DropboxManager(api_service)
    return await dbx_manager.process_list_folder(path=path, force_refresh=force_refresh)


@router.get("/sync/history", response_model=List[DropboxSyncRecord])
def get_sync_history(
    api_service: 'APIService' = Depends(ensure_initialized)
):
    db = api_service.db
    return db.run_op(DropboxSyncRecord, operation="get")


@router.post("/process")
async def register_items(
    req: DropboxRegisterRequest,
    request_obj: Request,
    api_service: 'APIService' = Depends(ensure_initialized)
) -> TaskStatus:
    """
    Enqueue a background task to register Dropbox items into the local DB.
    """
    dbx_manager = DropboxManager(api_service, request_obj.app)

    # Kick off the background task
    # task_runner returns a task_id
    task_id = await dbx_manager.process_register_items(req=req)

    # Return the TaskStatus so client can track progress
    return api_service.tasks[task_id]