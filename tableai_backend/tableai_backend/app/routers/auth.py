from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

from ..database import get_db
from ..models import User
from ..security import get_password_hash
from ..auth import authenticate, create_access_token
from ..schemas import RegisterUser  # <- new

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):
    user = authenticate(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    token = create_access_token(str(user.id))
    return {"access_token": token, "token_type": "bearer"}

@router.post("/register", status_code=201)
def register(payload: RegisterUser, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.email == payload.email).first()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=payload.email,
        full_name=payload.full_name or "",
        hashed_password=get_password_hash(payload.password),
        is_active=True,
        is_superuser=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Return a token so the new user is logged in right away
    token = create_access_token(str(user.id)) 
    return {"access_token": token, "token_type": "bearer"}