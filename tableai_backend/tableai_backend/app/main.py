from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
import uvicorn

from .database import Base, engine, SessionLocal
from .core.config import settings

from .routers import users
from .routers import auth
from .routers import metadata
# from .routers import classifier
from .routers import oauth_dropbox
from .routers import integrations_dropbox

from .admin import init_admin
from .security import get_password_hash

from .models import User
from .models import OAuthToken

from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(
    title="Minimal FastAPI + Postgres + JWT",
    openapi_url=None
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "change-me"),
    same_site="lax",  # optional hardening
    https_only=False  # set True in prod with HTTPS
)

# CORS (frontend dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.all_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(oauth_dropbox.router)
app.include_router(integrations_dropbox.router)
# app.include_router(classifier.router)
app.include_router(metadata.router)

# Admin
init_admin(app)

# Create tables and optional superuser
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    if settings.FIRST_SUPERUSER_EMAIL and settings.FIRST_SUPERUSER_PASSWORD:
        db: Session = SessionLocal()
        try:
            existing = db.query(User).filter(User.email == settings.FIRST_SUPERUSER_EMAIL).first()
            if not existing:
                user = User(
                    email=settings.FIRST_SUPERUSER_EMAIL,
                    full_name="Admin",
                    hashed_password=get_password_hash(settings.FIRST_SUPERUSER_PASSWORD),
                    is_superuser=True,
                )
                db.add(user)
                db.commit()
        finally:
            db.close()

@app.get("/", tags=["health"])
def read_root():
    return {"status": "ok"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))