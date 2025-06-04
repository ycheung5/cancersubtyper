from celery import Celery
from config import settings

# Initialize Celery
celery = Celery(
    "tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["tasks.task"]  # Ensure Celery discovers tasks
)

# Celery Configuration
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
