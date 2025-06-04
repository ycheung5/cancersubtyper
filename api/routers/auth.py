from fastapi import APIRouter, Depends, HTTPException
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from helpers.security import get_current_user, create_access_token
from repository.user_repository import UserRepository
from schemas.auth import UserSignup, UserLogin, TokenResponse, AuthResponse, UserResponse
from models import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup", response_model=AuthResponse)
async def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """
    Create a new user account and return access & refresh tokens.
    """
    user_repo = UserRepository(db)
    response = user_repo.signup(user_data)

    return response

@router.post("/login", response_model=AuthResponse)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user and return access & refresh tokens.
    """
    user_repo = UserRepository(db)
    response = user_repo.authenticate_user(user_data)

    return response

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    """
    Refresh an expired access token using a valid refresh token.
    """
    try:
        payload = jwt.decode(refresh_token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id: str = payload.get("sub")

        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        user_repo = UserRepository(db)
        user = user_repo.get_user_by_id(int(user_id))

        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        # Generate new access & refresh tokens
        new_access_token = create_access_token(user.id)
        new_refresh_token = refresh_token  # Optionally regenerate a new refresh token

        return TokenResponse(access_token=new_access_token, refresh_token=new_refresh_token)

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user info.
    """
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        timezone=current_user.timezone,
        storage_used=current_user.storage_used
    )
