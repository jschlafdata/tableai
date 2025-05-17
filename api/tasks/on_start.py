

from pathlib import Path 
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.service.manager import APIService
from tableai.core.paths import PathManager

class PreHooks:
    @staticmethod
    def mount_sync_folders(app: FastAPI = None, api_service: APIService = None):
        for mount_path, dir_path in PathManager.all_mount_configs().items():
            print(f"Initializing sync dir: {dir_path}")
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
            already_mounted = any(
                getattr(r, "path", None) == mount_path and isinstance(r.app, StaticFiles)
                for r in app.routes
            )
            if dir_path.exists() and not already_mounted:
                app.mount(mount_path, StaticFiles(directory=dir_path), name=mount_path.strip("/"))
                print(f"[✔] Mounted {mount_path} → {dir_path}")
            else:
                print(f"[ℹ] Mount skipped: exists={dir_path.exists()}, already mounted={already_mounted}, path={mount_path}")
