from pydantic import BaseModel
from models import UserRole  # Import the role enum


class UserSignup(BaseModel):
    username: str
    email: str
    timezone: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: UserRole
    timezone: str
    storage_used: int

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str

class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    user: UserResponse