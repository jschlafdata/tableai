from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from .core.config import settings
from .database import get_db
from .models import User
from .security import verify_password
from .schemas import Token, TokenPayload

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

from passlib.exc import UnknownHashError
from .security import verify_password, get_password_hash
from .models import User

def authenticate(db, email: str, password: str) -> User | None:
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    try:
        if verify_password(password, user.hashed_password):
            return user
    except UnknownHashError:
        # If the stored value is plaintext and matches what user typed, upgrade it
        if user.hashed_password and "$" not in user.hashed_password and user.hashed_password == password:
            user.hashed_password = get_password_hash(password)
            db.add(user)
            db.commit()
            db.refresh(user)
            return user
        return None
    return None

def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {"exp": expire, "sub": str(subject)}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        token_data = TokenPayload(**payload)
    except JWTError:
        raise credentials_exception
    if token_data.sub is None:
        raise credentials_exception
    user = db.query(User).get(int(token_data.sub))
    if user is None:
        raise credentials_exception
    return user

def get_current_superuser(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough privileges")
    return current_user
