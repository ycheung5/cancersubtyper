from http import HTTPStatus
from fastapi import HTTPException
from sqlalchemy.orm import Session

from helpers.security import hash_password, verify_password, create_access_token, create_refresh_token
from models import User, UserRole
from repository.base_repository import BaseRepository
from schemas.auth import UserSignup, UserLogin, TokenResponse, UserResponse, AuthResponse


class UserRepository(BaseRepository):
    def signup(self, user: UserSignup) -> AuthResponse:
        """
        Register a new user (Signup) and return tokens.

        Args:
            user (UserSignup): User registration data.

        Returns:
            TokenResponse: Access token & refresh token.
        """
        # Check if username or email already exists
        existing_user = self.db.query(User).filter(
            (User.username == user.username) | (User.email == user.email)
        ).first()

        if existing_user:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Username or email already exists")

        hashed_pw = hash_password(user.password)

        new_user = User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_pw,
            role=UserRole.USER,  # Default role
            timezone="America/New_York",  # Default if not provided
            storage_used=0  # Default to 0
        )

        self.db.add(new_user)
        self.db.commit()
        self.db.refresh(new_user)

        # Generate tokens
        access_token = create_access_token(new_user.id)
        refresh_token = create_refresh_token(new_user.id)

        user = UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            role=new_user.role,
            timezone=new_user.timezone,
            storage_used=new_user.storage_used
        )

        return AuthResponse(access_token=access_token, refresh_token=refresh_token, user=user)

    def authenticate_user(self, user_data: UserLogin) -> AuthResponse:
        """
        Authenticate user during login and return tokens.

        Args:
            user_data (UserLogin): User login credentials.

        Returns:
            TokenResponse: Access token & refresh token.
        """
        user = self.db.query(User).filter(User.username == user_data.username).first()

        if not user or not verify_password(user_data.password, user.hashed_password):
            raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Invalid email or password")

        # Generate tokens
        access_token = create_access_token(user.id)
        refresh_token = create_refresh_token(user.id)

        user = UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            timezone=user.timezone,
            storage_used=user.storage_used
        )

        return AuthResponse(access_token=access_token, refresh_token=refresh_token, user=user)

    def get_user_by_id(self, user_id: int) -> User | None:
        """
        Retrieve a user by their ID.

        Args:
            user_id (int): The user's ID.

        Returns:
            User or None: The user object if found, otherwise None.
        """
        return self.db.query(User).filter(User.id == user_id).first()

    def update_storage_usage(self, user_id: int, delta_bytes: int) -> None:
        """
        Update the user's storage usage by adding or subtracting bytes.

        Args:
            user_id (int): ID of the user.
            delta_bytes (int): Number of bytes to add (positive) or subtract (negative).

        Raises:
            HTTPException: If the resulting storage usage would go negative.
        """
        user = self.db.query(User).filter(User.id == user_id).first()

        if not user:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="User not found")

        new_usage = user.storage_used + delta_bytes
        if new_usage < 0:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Storage usage cannot be negative")

        user.storage_used = new_usage
        self.db.commit()

