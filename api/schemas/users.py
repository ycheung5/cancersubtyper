from pydantic import BaseModel

# Schema for updating user time zone
class UserTimezoneUpdate(BaseModel):
    timezone: str

# Response schema for successful time zone update
class UserTimezoneResponse(BaseModel):
    message: str
    timezone: str

# Schema for retrieving storage usage
class StorageUsageResponse(BaseModel):
    current_storage_usage: str
    max_storage_limit: str
