# backend/database/sqllite_manager.py

'''
This now using the config.db_connection_url from the .env file, which may be a postgres url.
'''

from dotenv import load_dotenv
import os
from typing import Type, Optional, Any, List

from sqlalchemy import text
from sqlmodel import SQLModel, Session, create_engine

from logger import logger

load_dotenv()
DATABASE_URL = os.environ["DB_CONNECTION_URL"]

logger.info(f"DATABASE_URL: {DATABASE_URL}")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True
)

# if you happen to still use SQLite, enable WAL; otherwise no-op
if engine.url.drivername.startswith("sqlite"):
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))

class DBManager:
    def __init__(self, engine=engine):
        self.engine = engine

    def run_raw(self, query: str) -> List[dict]:
        with Session(self.engine) as session:
            result = session.exec(text(query))
            return [dict(row._mapping) for row in result.fetchall()]

    def run_op(
        self,
        model_cls: Type[SQLModel],
        operation: str,
        data: Optional[Any] = None,
        filter_by: Optional[dict] = None,
        columns: Optional[List[str]] = None
    ) -> Any:
        with Session(self.engine) as session:
            if operation == "insert":
                instance = model_cls(**data)
                session.add(instance)
                session.commit()
                return {"status": "inserted", "uuid": getattr(instance, "uuid", None)}

            elif operation == "merge":
                instance = data if isinstance(data, SQLModel) else model_cls(**data)
                session.merge(instance)
                session.commit()
                return {"status": "merged", "uuid": getattr(instance, "uuid", None)}

            elif operation == "merge_many":
                if not isinstance(data, list):
                    raise ValueError("merge_many expects a list of model instances")
                for record in data:
                    session.merge(record)
                session.commit()
                return {"status": "merged_many", "count": len(data)}

            elif operation == "delete":
                query = session.query(model_cls)
                if filter_by:
                    query = query.filter_by(**filter_by)
                deleted = query.delete()
                session.commit()
                return {"status": "deleted", "count": deleted}

            elif operation == "get":
                if columns:
                    selected = [getattr(model_cls, col) for col in columns]
                    query = session.query(*selected)
                else:
                    query = session.query(model_cls)

                if filter_by:
                    query = query.filter_by(**filter_by)

                return query.all()

            else:
                raise ValueError(f"Unsupported DB operation: {operation}")

def init_db():
    """
    Create all tables defined on SQLModel classes.
    Call this at your app startup.
    """
    SQLModel.metadata.create_all(engine)
