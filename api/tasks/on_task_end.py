# api/hooks.py
from pathlib import Path 
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from tableai.extract.process import DropboxRegisterService
from tableai.core.paths import PathManager
from backend.models.backend import DropboxSyncRecord, FileExtractionResult, FileNodeRecord
from integrations.dropbox.sync import Validate
from api.service.manager import APIService

class PostHooks:
    @staticmethod
    def mount_sync_folders(app: FastAPI = None, api_service: APIService = None):
        for mount_path, dir_path in PathManager.all_mount_configs().items():
            already_mounted = any(
                getattr(r, "path", None) == mount_path and isinstance(r.app, StaticFiles)
                for r in app.routes
            )
            if dir_path.exists() and not already_mounted:
                app.mount(mount_path, StaticFiles(directory=dir_path), name=mount_path.strip("/"))
                print(f"[✔] Mounted {mount_path} → {dir_path}")
            else:
                print(f"[ℹ] Mount skipped: exists={dir_path.exists()}, already mounted={already_mounted}, path={mount_path}")

    @staticmethod
    def node_sync(app: FastAPI=None, api_service: APIService=None):
        node_manager = DropboxRegisterService(instance=api_service.db)
        validate_sync = Validate(api_service.db)
        validate_sync.validate_local_files()

        dbx_sync_records = api_service.db.run_op(DropboxSyncRecord, operation="get")
        file_node_records = api_service.db.run_op(FileNodeRecord, operation="get")

        synced_nodes = {node.dropbox_safe_id: node for node in dbx_sync_records}
        file_nodes = {node.uuid: node for node in file_node_records}
        non_processed_nodes = [node for _id, node in synced_nodes.items() if _id not in  file_nodes.keys()]
        node_manager._sync_post_classification()

        validate_sync = Validate(api_service.db)
        validate_sync.validate_local_files()

        print(f"FileNodeRecord Count: {len(file_node_records)}")
        print(f"DropboxSyncRecord Count: {len(dbx_sync_records)}")
        print("Merged and synced all nodes!")