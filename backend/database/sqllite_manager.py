from sqlmodel import SQLModel, Session, create_engine
from typing import Type, Optional, Any, List
from sqlalchemy import text

DATABASE_URL = "sqlite:///./dbx_sync_history.db"
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 30},
    future=True
)

# Enable WAL
with engine.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL"))

class DBManager:
    def __init__(self, engine):
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
                    # Dynamically access columns from the model class
                    selected = [getattr(model_cls, col) for col in columns]
                    query = session.query(*selected)
                else:
                    query = session.query(model_cls)

                if filter_by:
                    query = query.filter_by(**filter_by)

                return query.all()

            else:
                raise ValueError(f"Unsupported DB operation: {operation}")
