from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import Base, engine, SessionLocal
from .core.config import settings
from .routers import users, auth
from .admin import init_admin
from .models import User
from .security import get_password_hash

from .routers import oauth_dropbox, integrations_dropbox
from .models import OAuthToken
# from .routers import classifier

from starlette.middleware.sessions import SessionMiddleware
import os

app = FastAPI(title="Minimal FastAPI + Postgres + JWT")

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
