import os
from dotenv import load_dotenv
from pydantic.v1 import BaseSettings

# Load environment variables from .env
load_dotenv()

# Define Settings Class
class Settings(BaseSettings):
    # Database Configuration
    sqlalchemy_database_url: str = os.getenv("SQLALCHEMY_DATABASE_URL", "postgresql://postgres:postgres@db/cancersubtyper")

    # JWT Authentication
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "your-secret-key")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_access_token_expire_minutes: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    jwt_refresh_token_expire_days: int = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", 7))

    # Storage Limits
    max_storage_bytes: int = int(os.getenv("MAX_STORAGE_BYTES", 20 * 1024**3))  # Default 20GB
    data_dir: str = os.getenv("DATA_DIR", "/app/data")
    file_writer_chunk_size: int = int(os.getenv("FILE_WRITER_CHUNK_SIZE", 10))  # Default 1MB chunks

    # Global File
    cpg_info_file: str = os.getenv("CPG_INFO_FILE", "/app/data/cpg_info.csv")
    nemo_script_file: str = os.getenv("NEMO_SCRIPT_FILE", "/app/tasks/helper_scripts/run_nemo.R")

    # Celery & Redis Configuration
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")


# Initialize settings instance
settings = Settings()
