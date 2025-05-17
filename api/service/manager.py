import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Literal

### Appliation imports ###
from api.models.service import Settings, ServiceConfig
from api.models.tasks import TaskStatus
from api.tasks.wrappers import task_runner
### ------------------ ###

class APIService:
    """Main API service implementation"""

    def __init__(self, DB: 'DBManager'):
        self.db = DB
        self.logger = logging.getLogger("api_service.main")
        self.resource_providers = {}
        self.resource_cache = {}
        self.settings = Settings()
        self.tasks: Dict[str, TaskStatus] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cache = {}
        self.service_config = ServiceConfig()
        self.cache_ttl = timedelta(hours=self.service_config.cache_ttl_hours)
        self._initialized = False
        self._initializing = False
        self._shutdown = False

        # print(self.service_config)
        # self.dbx_client = DropboxAuth().get_client()

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
                await asyncio.sleep(self.service_config.cleanup_interval)
                if self._shutdown:
                    break
                    
                current_time = asyncio.get_event_loop().time()
                to_remove = [
                    task_id for task_id, task in self.tasks.items()
                    if task.status in ["completed", "failed", "cancelled"]
                    and hasattr(task, 'start_time')
                    and current_time - task.start_time > self.service_config.cleanup_interval
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