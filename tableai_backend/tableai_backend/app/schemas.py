from pydantic import BaseModel, EmailStr, Field, constr

class UserBase(BaseModel):
    email: EmailStr
    full_name: str | None = None
    is_active: bool = True
    is_superuser: bool = False

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str | None = None
    is_superuser: bool = False

class UserUpdate(BaseModel):
    email: EmailStr | None = None
    password: str | None = Field(default=None, min_length=6)
    full_name: str | None = None
    is_active: bool | None = None
    is_superuser: bool | None = None

class UserRead(UserBase):
    id: int
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: str | None = None


class RegisterUser(BaseModel):
    email: EmailStr
    password: constr(min_length=6)
    full_name: str | None = None