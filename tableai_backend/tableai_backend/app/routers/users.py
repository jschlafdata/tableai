from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..schemas import UserCreate, UserUpdate, UserRead
from ..security import get_password_hash
from ..auth import get_current_user, get_current_superuser

router = APIRouter(prefix="/users", tags=["users"])

@router.post("", response_model=UserRead)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
        is_superuser=user_in.is_superuser or False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.get("/me", response_model=UserRead)
def read_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.patch("/{user_id}", response_model=UserRead)
def update_user(
    user_id: int,
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not (current_user.is_superuser or current_user.id == user_id):
        raise HTTPException(status_code=403, detail="Not enough privileges")

    if user_in.email is not None:
        # Check for conflicts
        existing = db.query(User).filter(User.email == user_in.email, User.id != user_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already in use")
        user.email = user_in.email
    if user_in.full_name is not None:
        user.full_name = user_in.full_name
    if user_in.password is not None:
        user.hashed_password = get_password_hash(user_in.password)
    if user_in.is_active is not None and current_user.is_superuser:
        user.is_active = user_in.is_active
    if user_in.is_superuser is not None and current_user.is_superuser:
        user.is_superuser = user_in.is_superuser

    db.add(user)
    db.commit()
    db.refresh(user)
    return user
