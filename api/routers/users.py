from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import User
from database import get_db
from helpers.security import get_current_user
from schemas.users import UserTimezoneUpdate, UserTimezoneResponse, StorageUsageResponse
from config import settings
import pytz

router = APIRouter()

VALID_TIMEZONES = set(pytz.all_timezones)

@router.put("/timezone", response_model=UserTimezoneResponse)
def update_user_timezone(
    timezone_data: UserTimezoneUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Update the user's preferred time zone."""
    if timezone_data.timezone not in VALID_TIMEZONES:
        raise HTTPException(status_code=400, detail="Invalid time zone")

    user.timezone = timezone_data.timezone
    db.commit()

    return UserTimezoneResponse(message="Time zone updated successfully", timezone=user.timezone)

@router.get("/storage", response_model=StorageUsageResponse)
def get_storage_usage(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Retrieve the current storage usage of the user."""
    return StorageUsageResponse(
        current_storage_usage=f"{user.storage_used / (1024**3):.2f} GB",
        max_storage_limit=f"{settings.max_storage_bytes / (1024**3):.0f} GB"
    )
