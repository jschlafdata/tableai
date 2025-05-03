import asyncio
import logging
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from dropbox.files import FileMetadata, FolderMetadata

from api.app.config import ServiceConfig, Settings
from api.app.requests import FilterRequest, DataRequest
from api.app.tasks import TaskStatus, RateLimiter
from api.app.constants import ALLOWED_ORIGINS
from integrations.dropbox.auth import DropboxAuth
import os, json, re

from sqlmodel import SQLModel, Field, create_engine, Session, select

DATABASE_URL = "sqlite:///./sync_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

class SyncRecord(SQLModel, table=True):
    path_lower: str               = Field(primary_key=True)
    local_path: str
    size: int
    server_modified: datetime
    synced_at: datetime           = Field(default_factory=datetime.utcnow)
    metadata_json: str            = Field(default="{}")  # renamed from `metadata`


class SyncRequest(BaseModel):
    paths: List[str]


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_service")


class APIService:
    """Main API service implementation"""

    def __init__(self):
        self.logger = logging.getLogger("api_service.main")
        self.resource_providers = {}
        self.resource_cache = {}
        self.settings = Settings()
        self.tasks: Dict[str, TaskStatus] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cache = {}
        self.cache_ttl = timedelta(hours=config.cache_ttl_hours)
        self._initialized = False
        self._initializing = False
        self._shutdown = False

    async def initialize(self):
        """Initialize the service"""
        if self._initialized or self._initializing:
            return

        try:
            self._initializing = True
            self.logger.info("Starting service initialization...")
            
            # Validate directories
            self.settings._validate_directories()
            
            # Initialize core components
            self.logger.info("Initializing service components...")
            await self._initialize_components()
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self.cleanup_old_tasks())
            
            self._initialized = True
            self._initializing = False
            self.logger.info("Service initialization completed successfully")
            
        except Exception as e:
            self._initializing = False
            self.logger.error(f"Service initialization failed: {str(e)}", exc_info=True)
            raise

    async def _initialize_components(self):
        """Initialize core service components"""
        # Initialize any core service components here
        # This is a placeholder for your own initialization logic
        await asyncio.sleep(0.1)  # Simulate some async work
        
    async def cleanup_old_tasks(self):
        """Periodic cleanup of old tasks"""
        self.logger.info("Starting cleanup task")
        while not self._shutdown:
            try:
                await asyncio.sleep(config.cleanup_interval)
                if self._shutdown:
                    break
                    
                current_time = asyncio.get_event_loop().time()
                to_remove = [
                    task_id for task_id, task in self.tasks.items()
                    if task.status in ["completed", "failed", "cancelled"]
                    and hasattr(task, 'start_time')
                    and current_time - task.start_time > config.cleanup_interval
                ]
                
                for task_id in to_remove:
                    self.logger.debug(f"Removing old task: {task_id}")
                    del self.tasks[task_id]
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {str(e)}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retrying

    def _get_cached_data(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < self.cache_ttl:
                return entry['data']
        return None

    def _cache_data(self, key: str, data: Any):
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }

    async def get_task_metrics(self) -> Dict[str, Any]:
        """Get current task metrics"""
        total = len(self.tasks)
        processing = len([t for t in self.tasks.values() if t.status == "processing"])
        completed = len([t for t in self.tasks.values() if t.status == "completed"])
        failed = len([t for t in self.tasks.values() if t.status == "failed"])
        
        return {
            "total": total,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0
        }

    async def ensure_initialized(self):
        """Ensure service is initialized before processing requests"""
        if not self._initialized:
            await self.initialize()

    # async def initialize_provider(self, category: str):
    #     """Initialize a provider for a specific category"""
    #     try:
    #         await self.ensure_initialized()
            
    #         self.logger.info(f"Initializing provider for category: {category}")
            
    #         # This is a placeholder for your provider initialization logic
    #         # In a real implementation, you would initialize your data provider
            
    #         # Return a dummy provider
    #         return f"Provider_{category}_{uuid.uuid4()}"
            
    #     except Exception as e:
    #         self.logger.error(f"Error initializing provider: {str(e)}", exc_info=True)
    #         raise

    async def shutdown(self):
        """Graceful shutdown of the service"""
        self.logger.info("Initiating service shutdown...")
        self._shutdown = True
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

config = ServiceConfig()
api_service = APIService()
app = FastAPI(
    title="Generic API Service",
    description="Template for a general-purpose API service",
    default_response_class=ORJSONResponse,
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.middleware("http")
async def ensure_service_initialized(request: Request, call_next):
    """Middleware to ensure service is initialized before processing requests"""
    if not request.url.path in ["/health"]:  # Allow health checks during initialization
        await api_service.ensure_initialized()
    return await call_next(request)

@app.get("/health")
async def health_check():
    """Service health check"""
    try:
        initialized = api_service._initialized
        initializing = api_service._initializing
        metrics = await api_service.get_task_metrics() if initialized else None
        
        status = "healthy" if initialized else "initializing" if initializing else "starting"
        
        return {
            "status": status,
            "initialized": initialized,
            "initializing": initializing,
            "components": {
                "file_system": True,
                "providers": len(api_service.resource_providers) if initialized else 0,
                "cache_size": len(api_service.resource_cache) if initialized else 0,
                "tasks": metrics
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/data")
async def create_data_request(
    request: DataRequest,
    background_tasks: BackgroundTasks
) -> TaskStatus:
    """Create a new data processing request"""
    await api_service.ensure_initialized()

    task_id = str(uuid.uuid4())
    api_service.tasks[task_id] = TaskStatus(
        id=task_id,
        status="pending",
        progress=0,
        start_time=asyncio.get_event_loop().time()
    )

    background_tasks.add_task(
        api_service.process_data_request,
        task_id,
        request.category,
        request.resource_id,
        request.force_refresh,
        request.filters
    )

    return api_service.tasks[task_id]


auth = DropboxAuth()
dbx = auth.get_client()


@app.get("/folders/{path:path}")
async def list_folder(path: str):
    # Normalize root
    db_path = "" if path in ("", "/", "root") else f"/{path.lstrip('/')}"
    try:
        res = dbx.files_list_folder(db_path)
        out = []
        for e in res.entries:
            # Base fields
            row = {
                "id":         e.id,
                "name":       e.name,
                "path_lower": e.path_lower,
            }

            if isinstance(e, FolderMetadata):
                row.update({
                    "type":            "folder",
                    "size":            None,
                    "server_modified": None,
                    # placeholder: you could fetch sharing info on folders here
                    "shared_with":     [],
                })
            elif isinstance(e, FileMetadata):
                row.update({
                    "type":            "file",
                    "size":            e.size,                        # bytes
                    "server_modified": e.server_modified.isoformat(),  # string
                    # placeholder: you could fetch sharing info on files here
                    "shared_with":     [],
                })
            out.append(row)
        return out

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


class SyncRequest(BaseModel):
    paths: List[str]  # list of Dropbox “path_lower” values, e.g. ["invoices", "reports/April.pdf"]

def sanitize(name: str) -> str:
    """
    Replace any character not in [a-zA-Z0-9._-] with underscore,
    collapse multiple underscores, strip leading/trailing.
    """
    s = re.sub(r'[^0-9A-Za-z.\-_]', '_', name)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


@app.post("/sync")
def sync_items(request: SyncRequest):
    PUBLIC_DIR = Path(__file__).parent.parent.parent / "frontend" / "ui" / "public" / "synced"
    synced_files: List[str] = []

    def download_file(dbx_path: str, local_dir: Path):
        os.makedirs(local_dir, exist_ok=True)
        metadata, result = dbx.files_download(dbx_path)
        # slugify the filename
        filename = sanitize(metadata.name)
        target = local_dir / filename

        with open(target, "wb") as f:
            f.write(result.content)

        rel = str(target.relative_to(PUBLIC_DIR))
        synced_files.append(rel)

        # record/update history
        record = SyncRecord(
            path_lower=dbx_path,
            local_path=str(target),
            size=metadata.size,
            server_modified=metadata.server_modified,
            synced_at=datetime.utcnow(),
            metadata_json=json.dumps({"original_name": metadata.name})
        )
        with Session(engine) as session:
            session.merge(record)
            session.commit()

    def download_folder(dbx_folder: str, local_folder: Path):
        os.makedirs(local_folder, exist_ok=True)
        resp = dbx.files_list_folder(dbx_folder)
        for entry in resp.entries:
            if isinstance(entry, FolderMetadata):
                # slugify subfolder name
                subname = sanitize(entry.name)
                download_folder(entry.path_lower, local_folder / subname)
            elif isinstance(entry, FileMetadata):
                # skip if up-to-date
                with Session(engine) as session:
                    prev = session.get(SyncRecord, entry.path_lower)
                if prev and prev.server_modified >= entry.server_modified:
                    continue
                download_file(entry.path_lower, local_folder)
        with Session(engine) as session:
           session.merge(
               SyncRecord(
                   path_lower=dbx_folder,
                   local_path=str(local_folder),
                   size=0,
                   server_modified=datetime.utcnow(),
                   synced_at=datetime.utcnow(),
                   metadata_json=json.dumps({"is_folder": True})
               )
           )
           session.commit()

    try:
        for p in request.paths:
            # normalize incoming path, then slugify each segment for local dir
            dbx_path = "" if p in ("", "/", "root") else "/" + p.lstrip("/")
            meta = dbx.files_get_metadata(dbx_path)

            if isinstance(meta, FolderMetadata):
                # build a sanitized local path under PUBLIC_DIR
                parts = [sanitize(seg) for seg in p.split("/") if seg]
                local_base = PUBLIC_DIR.joinpath(*parts) if parts else PUBLIC_DIR
                download_folder(dbx_path, local_base)
            else:
                with Session(engine) as session:
                    prev = session.get(SyncRecord, dbx_path)
                if not prev or prev.server_modified < meta.server_modified:
                    parent_parts = [sanitize(seg) for seg in Path(p).parent.parts]
                    local_parent = PUBLIC_DIR.joinpath(*parent_parts)
                    download_file(dbx_path, local_parent)

        return {"synced_files": synced_files}

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/sync/history", response_model=List[SyncRecord])
def get_sync_history():
    with Session(engine) as session:
        return session.exec(select(SyncRecord)).all()


@app.on_event("startup")
def init_db():
    SQLModel.metadata.create_all(engine)

@app.get("/data/{task_id}")
async def get_data_status(task_id: str) -> TaskStatus:
    """Get task status"""
    if task_id not in api_service.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return api_service.tasks[task_id]

@app.get("/data/{task_id}/result")
async def get_data_result(task_id: str):
    """Get task result"""
    if task_id not in api_service.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = api_service.tasks[task_id]
    if task.status == "completed":
        return task.result
    elif task.status == "failed":
        raise HTTPException(status_code=500, detail=task.error)
    else:
        raise HTTPException(status_code=202, detail="Processing not complete")


@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return await api_service.get_task_metrics()

async def run_service():
    """Run the API service"""
    logger.info("Starting API service...")

    try:
        # Initialize service
        logger.info("Initializing API service...")
        await api_service.initialize()
        
        # Start server
        logger.info(f"Starting server on port {config.port}")
        loop = asyncio.get_running_loop()
        uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=config.port, loop=loop)
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
        
    except Exception as e:
        logger.error(f"Service error: {str(e)}", exc_info=True)
        raise
    finally:
        # Shutdown service
        logger.info("Shutting down API service...")
        await api_service.shutdown()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generic API Service')
    parser.add_argument('--port', type=int, help='Port to run service on')
    parser.add_argument('--force-restart', action='store_true', help='Force restart if running')
    args = parser.parse_args()

    # Override config with command line arguments
    if args.port:
        config.port = args.port
    if args.force_restart:
        config.force_restart = True

    # Run service
    asyncio.run(run_service())










   # async def get_or_create_data(
    #     self,
    #     category: str,
    #     resource_id: str,
    #     force_refresh: bool = False
    # ) -> Any:
    #     try:
    #         data_key = f"{category}:{resource_id}"
            
    #         if not force_refresh:
    #             # Check memory cache
    #             cached_data = self._get_cached_data(data_key)
    #             if cached_data:
    #                 self.logger.debug(f"Cache hit for {data_key}")
    #                 return cached_data
                
    #             # Check resource cache
    #             if data_key in self.resource_cache:
    #                 return self.resource_cache[data_key]
            
    #         self.logger.info(f"Creating new data for {data_key}")
            
    #         # Initialize or get resource provider
    #         if category not in self.resource_providers or force_refresh:
    #             provider = await self.initialize_provider(category)
    #             self.resource_providers[category] = provider
    #         else:
    #             provider = self.resource_providers[category]
            
    #         # Fetch data
    #         data = await self._fetch_data(provider, resource_id)
            
    #         # Store in both caches
    #         self.resource_cache[data_key] = data
    #         self._cache_data(data_key, data)
            
    #         return data
            
    #     except Exception as e:
    #         self.logger.error(f"Error in get_or_create_data: {str(e)}", exc_info=True)
    #         raise

    # async def _fetch_data(self, provider, resource_id):
    #     """Fetch data from the provider"""
    #     # This is a placeholder for your data fetching logic
    #     # In a real implementation, you would call your data provider
    #     await asyncio.sleep(0.5)  # Simulate async work
        
    #     # Return dummy data
    #     return {
    #         "id": resource_id,
    #         "provider": str(provider),
    #         "timestamp": datetime.now().isoformat(),
    #         "data": {
    #             "sample": "This is sample data",
    #             "values": [1, 2, 3, 4, 5]
    #         }
    #     }

    # async def process_data_request(
    #     self,
    #     task_id: str,
    #     category: str,
    #     resource_id: str,
    #     force_refresh: bool,
    #     filters: Optional[FilterRequest] = None
    # ):
    #     """Process a data request"""
    #     try:
    #         self.logger.info(f"Processing task {task_id} for {category}/{resource_id}")
    #         self.tasks[task_id].status = "processing"
    #         self.tasks[task_id].start_time = asyncio.get_event_loop().time()
            
    #         # Initialize provider
    #         if category not in self.resource_providers or force_refresh:
    #             self.tasks[task_id].progress = 0.2
    #             provider = await self.initialize_provider(category)
    #             self.resource_providers[category] = provider
            
    #         self.tasks[task_id].progress = 0.4
    #         data = await self.get_or_create_data(
    #             category, resource_id, force_refresh
    #         )
            
    #         self.tasks[task_id].progress = 0.7

    #         # Apply filters if needed
    #         filtered_data = await self._apply_filters(data, filters)
            
    #         result = {
    #             "resource_meta": {
    #                 "category": category,
    #                 "resource_id": resource_id,
    #                 "timestamp": datetime.now().isoformat()
    #             },
    #             "version": filters.version if filters else 1,
    #             "data": filtered_data
    #         }
            
    #         self.tasks[task_id].status = "completed"
    #         self.tasks[task_id].progress = 1.0
    #         self.tasks[task_id].result = result
            
    #     except Exception as e:
    #         self.logger.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
    #         self.tasks[task_id].status = "failed"
    #         self.tasks[task_id].error = str(e)
    #         raise

    # async def _apply_filters(self, data: Any, filters: Optional[FilterRequest] = None) -> Any:
    #     """Apply filters to the data"""
    #     if not filters:
    #         return data
            
        # This is a placeholder for your filtering logic
        # In a real implementation, you would apply the filters to the data
        
        # For now, just return the original data with a filter note
        # return {
        #     "original_data": data,
        #     "filter_applied": {
        #         "version": filters.version if filters else 1,
        #         "included_categories": filters.include_categories if filters and filters.include_categories else [],
        #         "excluded_categories": filters.exclude_categories if filters and filters.exclude_categories else [],
        #     }
        # }