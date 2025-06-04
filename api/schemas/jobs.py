from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from models import JobStatusEnum


class JobCreate(BaseModel):
    """Schema for creating a new job."""
    project_id: int
    model_id: int
    model_parameters: list


class JobResponse(BaseModel):
    """Schema for returning job details."""
    id: int
    project_id: int
    model_name: str
    user_id: int
    status: JobStatusEnum
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    class Config:
        orm_mode = True
