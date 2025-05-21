import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.requests import Request
import logging
import uvicorn
import asyncio
import uuid
from sqlmodel import SQLModel
import traceback
import argparse

from fastapi import (
    FastAPI, 
    HTTPException, 
    BackgroundTasks,
    Request, 
    Depends, 
    Body,
    BackgroundTasks
)

### Application imports ###
from api.service.dependencies import get_db, ensure_initialized, engine
from api.integrations import dropbox
from api.models.requests import FilterRequest, DataRequest
from api.models.service import ServiceConfig, Settings
from api.models.tasks import TaskStatus, RateLimiter
from api.constants import ALLOWED_ORIGINS
from api.tasks.on_start import PreHooks

# from api.app.hooks import PostHooks
from api.tableai import extract
from api.tableai import frontend
from api.database import run_queries
from api.routes.inference import vision

### ------------------- ###

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_service")

app = FastAPI(
    title="Generic API Service",
    description="Template for a general-purpose API service",
    default_response_class=ORJSONResponse,
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

service_config = ServiceConfig()

app.include_router(run_queries.router, prefix="/query")
app.include_router(extract.router, prefix="/tableai/extract")
app.include_router(frontend.router, prefix="/ui")
app.include_router(dropbox.router, prefix="/dropbox")
app.include_router(vision.router, prefix="/tableai/vision")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("UNCAUGHT EXCEPTION:", str(exc))
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


@app.on_event("startup")
def init_db():
    SQLModel.metadata.create_all(engine)
    PreHooks.mount_sync_folders(app)

@app.middleware("http")
async def ensure_service_initialized(request: Request, call_next):
    if not request.url.path.startswith("/health"):
        await ensure_initialized()
    return await call_next(request)

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return await ensure_initialized().get_task_metrics()

@app.get("/health")
async def health_check():
    """Service health check"""
    try:
        api_service = await ensure_initialized()
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
    

@app.post("/tasks/requests")
async def create_data_request(
    request: DataRequest,
    background_tasks: BackgroundTasks
) -> TaskStatus:
    """Create a new data processing request"""
    api_service = await ensure_initialized()

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

@app.get("/tasks/metrics")
async def get_data_status() -> dict:
    """Get task status metrics"""
    api_service = await ensure_initialized()
    return await api_service.get_task_metrics()

@app.get("/tasks/{task_id}")
async def get_data_status(task_id: str) -> TaskStatus:
    """Get task status"""
    api_service = await ensure_initialized()
    if task_id not in api_service.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return api_service.tasks[task_id]

@app.get("/tasks/{task_id}/result")
async def get_data_result(task_id: str):
    """Get task result"""
    api_service = await ensure_initialized()
    if task_id not in api_service.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = api_service.tasks[task_id]
    if task.status == "completed":
        return task.result
    elif task.status == "failed":
        raise HTTPException(status_code=500, detail=task.error)
    else:
        raise HTTPException(status_code=202, detail="Processing not complete")


async def run_service():
    """Run the API service"""
    logger.info("Starting API service...")

    try:
        # Initialize service
        logger.info("Initializing API service...")
        api_service = await ensure_initialized()        
        # Start server
        logger.info(f"Starting server on port {service_config.port}")
        loop = asyncio.get_running_loop()
        uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=service_config.port, loop=loop)
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

    parser = argparse.ArgumentParser(description='Generic API Service')
    parser.add_argument('--port', type=int, help='Port to run service on')
    parser.add_argument('--force-restart', action='store_true', help='Force restart if running')
    args = parser.parse_args()

    # Override config with command line arguments
    if args.port:
        service_config.port = args.port
    if args.force_restart:
        service_config.force_restart = True

    # Run service
    asyncio.run(run_service())
