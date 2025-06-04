from datetime import datetime

from pydantic import BaseModel
from typing import Optional

class ProjectCreate(BaseModel):
    name: str
    tumor_type: str
    description: Optional[str] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    tumor_type: Optional[str] = None
    description: Optional[str] = None
        
class ProjectResponse(BaseModel):
    id: int
    name: str
    tumor_type: str = None
    description: Optional[str] = None
    user_id: int
    target_file: Optional[str] = None
    source_file: Optional[str] = None
    metadata_file: Optional[str] = None
    active: bool
    created_at: datetime
    edited_at: datetime

    class Config:
        from_attributes = True