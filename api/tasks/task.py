from celery import shared_task
from sqlalchemy.orm import Session
from database import SessionLocal
from helpers.timezone import get_utc_time
from models import JobStatusEnum, Project, Job
from tasks.registry import get_model_pipeline
import traceback


@shared_task(bind=True)
def run_model(self, job_id: int, model_parameters: list):
    """Background task to execute a job (Pending → Preprocessing → Running → Completed/Failed)."""
    db: Session = SessionLocal()

    try:
        # Step 1: Fetch Job & Project Info
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": f"Job {job_id} not found"}

        job.task_id = self.request.id  # Store Celery Task ID
        db.commit()

        project = db.query(Project).filter(Project.id == job.project_id).first()
        if not project:
            return {"error": f"Project {job.project_id} not found"}

        # Step 2: Update Job Status to "Preprocessing"
        job.status = JobStatusEnum.PREPROCESSING
        job.started_at = get_utc_time()
        db.commit()
        print(f"[Job {job_id}] Preprocessing started.")

        # Step 3: Fetch Preprocessing & Execution Functions
        preprocess_func, execute_func = get_model_pipeline(job.model.name)

        # Step 4: Run Preprocessing
        preprocess_func(job.user_id, project.id, job_id, project.source_file, project.target_file)
        print(f"[Job {job_id}] Preprocessing completed.")

        # Step 5: Update Job Status to "Running"
        job.status = JobStatusEnum.RUNNING
        db.commit()
        print(f"[Job {job_id}] Model execution started.")

        # Step 6: Run Model Execution
        metadata_file = project.metadata_file if project.metadata_file else None
        execute_func(job.user_id, project.id, job_id, model_parameters, metadata_file)

        print(f"[Job {job_id}] Model execution completed.")

        # Step 7: Mark Job as Completed
        job.status = JobStatusEnum.COMPLETED
        job.finished_at = get_utc_time()
        db.commit()
        print(f"[Job {job_id}] Completed successfully.")

        return {"job_id": job_id, "status": job.status.value}

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"[Job {job_id}] Failed! Error: {error_msg}")
        print(traceback_str)

        job.status = JobStatusEnum.FAILED
        job.finished_at = get_utc_time()
        db.commit()

        return {"job_id": job_id, "status": job.status.value, "error": error_msg}

    finally:
        db.close()
