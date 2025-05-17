from backend.database.sqllite_manager import DBManager, engine
from sqlalchemy import create_engine
from api.service.manager import APIService
import asyncio

db = DBManager(engine)
api_service = APIService(db)

def get_db():
    return db

async def ensure_initialized():
    await api_service.ensure_initialized()
    return api_service