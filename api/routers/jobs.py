import os
from http import HTTPStatus
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import get_db
from helpers.security import get_current_user
from helpers.timezone import convert_to_user_tz
from models import User
from repository.job_repository import JobRepository
from repository.model_repository import ModelRepository
from repository.project_repository import ProjectRepository
from schemas.jobs import JobCreate, JobResponse
from util.path_untils import job_result_path, example_result_path, example_base_dir

router = APIRouter(prefix="/job", tags=["job"])


@router.post("", status_code=HTTPStatus.CREATED, response_model=JobResponse)
def create_job(
        job_data: JobCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Create a new job and enqueue it for execution.
    """
    job_repo = JobRepository(db)
    project_repo = ProjectRepository(db)
    model_repo = ModelRepository(db)

    job = job_repo.create_job(job_data, project_repo)

    return JobResponse(
        id=job.id,
        project_id=job.project_id,
        model_name=model_repo.get_model_by_id(job.model_id).name,
        user_id=job.user_id,
        status=job.status,
        created_at=convert_to_user_tz(job.created_at, current_user.timezone),
        started_at=convert_to_user_tz(job.started_at, current_user.timezone) if job.started_at else None,
        finished_at=convert_to_user_tz(job.finished_at, current_user.timezone) if job.finished_at else None
    )


@router.get("/project/{project_id}", status_code=HTTPStatus.OK, response_model=list[JobResponse])
def get_jobs(
        project_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Retrieve all jobs for a given project.
    """
    job_repo = JobRepository(db)
    model_repo = ModelRepository(db)

    jobs = job_repo.get_jobs_by_project(project_id)

    return [
        JobResponse(
            id=job.id,
            project_id=job.project_id,
            model_name=model_repo.get_model_by_id(job.model_id).name,
            user_id=job.user_id,
            status=job.status,
            created_at=convert_to_user_tz(job.created_at, current_user.timezone),
            started_at=convert_to_user_tz(job.started_at, current_user.timezone) if job.started_at else None,
            finished_at=convert_to_user_tz(job.finished_at, current_user.timezone) if job.finished_at else None
        )
        for job in jobs
    ]


@router.get("/{job_id}", status_code=HTTPStatus.OK, response_model=JobResponse)
def get_job(
        job_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Retrieve details of a specific job.
    """
    job_repo = JobRepository(db)
    model_repo = ModelRepository(db)
    job = job_repo.get_job_by_id(job_id)

    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="You do not have permission to view this job")

    return JobResponse(
        id=job.id,
        project_id=job.project_id,
        model_name=model_repo.get_model_by_id(job.model_id).name,  # Ensuring consistency
        user_id=job.user_id,
        status=job.status,
        created_at=convert_to_user_tz(job.created_at, current_user.timezone),
        started_at=convert_to_user_tz(job.started_at, current_user.timezone) if job.started_at else None,
        finished_at=convert_to_user_tz(job.finished_at, current_user.timezone) if job.finished_at else None
    )


@router.delete("/{job_id}/cancel", status_code=HTTPStatus.OK)
def cancel_job(
        job_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Cancel a job if it is still pending or running.
    """
    job_repo = JobRepository(db)
    job = job_repo.get_job_by_id(job_id)

    if not job:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Job not found")

    if job.user_id != current_user.id:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="You do not have permission to cancel this job")

    return job_repo.cancel_job(job_id)  # Now using repository for cancellation logic


@router.get("/model/all-models", status_code=HTTPStatus.OK)
def get_models(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    model_repo = ModelRepository(db)
    return model_repo.get_all_models()

@router.get("/examples/{model}/results", status_code=HTTPStatus.OK)
def get_example_results(model: str):
    """
    Endpoint to return the pre-zipped result file for example data.
    Model should be 'bctypefinder' or 'cancersubminer'.
    No authentication required for demo/example downloads.
    """
    
    # Validate model parameter
    if model not in ["bctypefinder", "cancersubminer"]:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, 
            detail="Model must be 'bctypefinder' or 'cancersubminer'"
        )
    
    # Get the example result path
    example_dir = example_result_path(model)
    zip_path = os.path.join(example_dir, "results.zip")
    
    # Check if the zip file exists
    if not os.path.exists(zip_path):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, 
            detail=f"Example result ZIP file not found for {model}"
        )
    
    # Return the ZIP file for download
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{model}_example_results.zip"
    )


@router.get("/examples/datasets/{dataset_type}", status_code=HTTPStatus.OK)
def download_example_dataset(dataset_type: str):
    """
    Endpoint to download example datasets (source, target, metadata).
    Dataset type should be 'source', 'target', or 'metadata'.
    No authentication required for demo/example downloads.
    """
    
    # Validate dataset type parameter
    if dataset_type not in ["source", "target", "metadata"]:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, 
            detail="Dataset type must be 'source', 'target', or 'metadata'"
        )
    
    # Get the example data path
    example_data_dir = os.path.join(example_base_dir(), "data")
    
    # Map dataset types to file names
    file_mapping = {
        "source": "example_source.csv",
        "target": "example_target.csv", 
        "metadata": "example_metadata.csv"
    }
    
    file_name = file_mapping[dataset_type]
    file_path = os.path.join(example_data_dir, file_name)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, 
            detail=f"Example {dataset_type} dataset not found"
        )
    
    # All files are CSV format
    media_type = "text/csv"
    download_name = file_name
    
    # Return the file for download
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=download_name
    )


@router.get("/{job_id}/download/results", status_code=HTTPStatus.OK)
async def download_results(
        job_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Endpoint to return the pre-zipped result file for download.
    """
    job_repo = JobRepository(db)
    job = job_repo.get_job_by_id(job_id)

    zip_path = os.path.join(job_result_path(current_user.id, job.project_id, job.id), "results.zip")

    # Check if the zip file exists
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Result ZIP file not found")

    # Return the ZIP file for download
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="results.zip"
    )