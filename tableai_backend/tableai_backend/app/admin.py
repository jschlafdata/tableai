from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
import wtforms
import os

from .database import engine, SessionLocal
from .models import User
from .security import get_password_hash
from .auth import authenticate  # reuses your existing verify logic

class AdminAuth(AuthenticationBackend):
    """Session-based guard for the /admin UI (superusers only)."""

    async def login(self, request: Request) -> bool:
        form = await request.form()
        email = form.get("username")  # SQLAdmin's login form uses 'username'
        password = form.get("password")
        if not email or not password:
            return False

        db = SessionLocal()
        try:
            user = authenticate(db, email, password)
            if not user or not user.is_superuser:
                return False
            # mark session as admin
            request.session.update({"user_id": user.id, "is_superuser": True})
            return True
        finally:
            db.close()

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        # Block everything under /admin unless session says superuser
        return bool(request.session.get("is_superuser"))

def init_admin(app):
    admin = Admin(
        app,
        engine,
        authentication_backend=AdminAuth(secret_key=os.getenv("SECRET_KEY", "change-me")),
    )

    # Optional: base class to enforce per-view access as well
    class SecureModelView(ModelView):
        def is_accessible(self, request: Request) -> bool:
            return bool(request.session.get("is_superuser"))

        def is_visible(self, request: Request) -> bool:
            return bool(request.session.get("is_superuser"))

    class UserAdmin(SecureModelView, model=User):
        column_list = [User.id, User.email, User.is_active, User.is_superuser]
        column_details_exclude_list = [User.hashed_password]

        # Admin form: show "Password" and hash on CREATE only
        column_labels = {"hashed_password": "Password"}
        form_overrides = dict(hashed_password=wtforms.PasswordField)
        form_create_rules = ["email", "full_name", "hashed_password", "is_active", "is_superuser"]
        form_edit_rules = ["email", "full_name", "is_active", "is_superuser"]

        async def on_model_change(self, data, model, is_created, request):
            pwd = data.get("hashed_password")
            if is_created and pwd:
                data["hashed_password"] = get_password_hash(pwd)

    admin.add_view(UserAdmin)
    return admin
