import datetime
from http import HTTPStatus

import bcrypt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from models import User

oauth2_scheme = HTTPBearer()

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_password.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a hashed password against a plain password."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def create_access_token(user_id: int) -> str:
    """Create an access token for authentication."""
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        minutes=settings.jwt_access_token_expire_minutes
    )
    payload = {"sub": str(user_id), "exp": expire.timestamp()}
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token

def create_refresh_token(user_id: int) -> str:
    """Create a refresh token for re-authentication."""
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        days=settings.jwt_refresh_token_expire_days
    )
    payload = {"sub": str(user_id), "exp": expire.timestamp()}
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Extract the user from the JWT token and validate authentication.

    Args:
        credentials (HTTPAuthorizationCredentials): Extracted token from request header.
        db (Session): Database session dependency.

    Returns:
        User: The authenticated user object.

    Raises:
        HTTPException: If the token is invalid, expired, or the user is not found.
    """
    from repository.user_repository import UserRepository

    token = credentials.credentials  # Extract token from Authorization header

    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id: str = payload.get("sub")
        exp: float = payload.get("exp")

        if user_id is None:
            raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Invalid token payload")

        # Check if token has expired
        if datetime.datetime.now(datetime.timezone.utc).timestamp() > exp:
            raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Token has expired")

        # Fetch user from the database
        user_repo = UserRepository(db)
        user = user_repo.get_user_by_id(int(user_id))

        if user is None:
            raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="User not found")

        return user

    except JWTError:
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Invalid token")
