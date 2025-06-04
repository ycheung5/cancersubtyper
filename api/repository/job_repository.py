from http import HTTPStatus
from fastapi import HTTPException
from celery.result import AsyncResult

from models import Job, JobStatusEnum, Model
from repository.base_repository import BaseRepository
from repository.project_repository import ProjectRepository
from schemas.jobs import JobCreate
from tasks.task import run_model
from celery_config import celery
from helpers.timezone import get_utc_time


class JobRepository(BaseRepository):
    def create_job(self, job_data: JobCreate, project_repo: ProjectRepository):
        """Creates a new job, validates project existence, and queues the task for Celery."""

        # Step 1: Validate Project Exists
        project = project_repo.get_project_by_id(job_data.project_id)
        if not project:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Project not found")

        # Step 2: Validate Model Exists
        model = self.db.query(Model).filter(Model.id == job_data.model_id).first()
        if not model:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Model not found")

        # Step 3: Check Project Files Exist
        if not project.source_file or not project.target_file:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Source or Target file missing in Project")

        # Step 4: Create a New Job
        new_job = Job(
            project_id=job_data.project_id,
            model_id=job_data.model_id,
            user_id=project.user_id,
            status=JobStatusEnum.PENDING,
            created_at=get_utc_time(),
        )
        self.db.add(new_job)
        self.db.commit()
        self.db.refresh(new_job)

        # Step 5: Send Job to Celery Worker
        run_model.delay(new_job.id, job_data.model_parameters)

        return new_job

    def get_jobs_by_project(self, project_id: int):
        """Retrieve all jobs for a given project."""
        return self.db.query(Job).filter(Job.project_id == project_id).all()

    def get_job_by_id(self, job_id: int):
        """Retrieve a specific job by its ID."""
        return self.db.query(Job).filter(Job.id == job_id).first()

    def cancel_job(self, job_id: int):
        """Cancel a job if it is still pending or running."""
        job = self.get_job_by_id(job_id)

        if not job:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Job not found")

        if job.status not in [JobStatusEnum.PENDING, JobStatusEnum.PREPROCESSING, JobStatusEnum.RUNNING]:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Only running or pending jobs can be canceled")

        # Revoke the Celery task (Force Termination)
        task = AsyncResult(str(job_id), app=celery)
        if task:
            task.revoke(terminate=True, signal="SIGKILL")  # Kill the task

        # Update job status in DB
        job.status = JobStatusEnum.FAILED
        job.finished_at = get_utc_time()
        self.db.commit()

        return {"message": f"Job {job_id} has been canceled successfully."}
