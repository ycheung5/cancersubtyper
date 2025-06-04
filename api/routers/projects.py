import gzip
import os
import shutil
from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Response
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from helpers.security import get_current_user
from models import User
from repository.project_repository import ProjectRepository
from repository.user_repository import UserRepository
from schemas.projects import ProjectCreate, ProjectResponse, ProjectUpdate
from util.file_utils import delete_file_async, write_uploadfile_chunked_async, decompress_gzip_file
from util.path_untils import target_sample_file_path, source_sample_file_path, metadata_sample_path

router = APIRouter(prefix="/project", tags=["project"])

@router.get("", status_code=HTTPStatus.OK, response_model=List[ProjectResponse])
async def get_projects(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):
    project_repo = ProjectRepository(db)
    return project_repo.get_project_by_user_id(current_user.id, current_user.timezone)

@router.post("/create", status_code=HTTPStatus.CREATED, response_model=ProjectResponse)
async def create_project(
        project_data: ProjectCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    project_repo = ProjectRepository(db)
    return project_repo.create_project(project_data, current_user.id)

@router.put("/{project_id}/upload", status_code=HTTPStatus.OK, response_model=ProjectResponse)
async def upload_project_samples(
    project_id: int,
    target: UploadFile = File(...),
    source: UploadFile = File(...),
    target_checksum: str = Form(...),
    source_checksum: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    project_repo = ProjectRepository(db)
    user_repo = UserRepository(db)
    project = project_repo.get_project_by_id(project_id)

    if not project:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Project not found")
    if project.user_id != current_user.id:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="You do not have permission to modify this project")
    if not target.filename.endswith(".gz") or not source.filename.endswith(".gz"):
        raise HTTPException(status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE, detail="Only .gz files are supported")

    # File paths
    target_gz_path = target_sample_file_path(current_user.id, project_id, target.filename)
    source_gz_path = source_sample_file_path(current_user.id, project_id, source.filename)
    target_csv_path = target_gz_path[:-3]
    source_csv_path = source_gz_path[:-3]

    # Track old sizes for rollback
    old_target_csv_path = (
        target_sample_file_path(current_user.id, project_id, project.target_file)
        if project.target_file else None
    )
    old_source_csv_path = (
        source_sample_file_path(current_user.id, project_id, project.source_file)
        if project.source_file else None
    )

    old_target_size = os.path.getsize(old_target_csv_path) if old_target_csv_path and os.path.exists(old_target_csv_path) else 0
    old_source_size = os.path.getsize(old_source_csv_path) if old_source_csv_path and os.path.exists(old_source_csv_path) else 0
    old_total = old_target_size + old_source_size

    # Delete old files
    if old_target_csv_path:
        await delete_file_async(old_target_csv_path)
    if old_source_csv_path:
        await delete_file_async(old_source_csv_path)

    # Write new gz files
    computed_target_checksum = await write_uploadfile_chunked_async(target, target_gz_path)
    computed_source_checksum = await write_uploadfile_chunked_async(source, source_gz_path)

    if computed_target_checksum != target_checksum or computed_source_checksum != source_checksum:
        await delete_file_async(target_gz_path)
        await delete_file_async(source_gz_path)
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Checksum mismatch")

    # Decompress
    try:
        with gzip.open(target_gz_path, 'rb') as f_in, open(target_csv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        with gzip.open(source_gz_path, 'rb') as f_in, open(source_csv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        # Delete original .gz files
        await delete_file_async(target_gz_path)
        await delete_file_async(source_gz_path)
    except Exception:
        await delete_file_async(target_gz_path)
        await delete_file_async(source_gz_path)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to decompress gzip files")

    # Check storage quota (adjusted for replacement)
    new_total = os.path.getsize(target_csv_path) + os.path.getsize(source_csv_path)
    net_storage_change = new_total - old_total

    if current_user.storage_used + net_storage_change > settings.max_storage_bytes:
        await delete_file_async(target_csv_path)
        await delete_file_async(source_csv_path)
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Upload exceeds your storage quota")

    # Update user storage usage
    user_repo.update_storage_usage(current_user.id, net_storage_change)

    # Update DB with new filenames
    return project_repo.upload_project_samples(
        project_id, os.path.basename(target_csv_path), os.path.basename(source_csv_path)
    )

@router.put("/{project_id}/upload_metadata", status_code=HTTPStatus.OK, response_model=ProjectResponse)
async def upload_project_metadata(
        project_id: int,
        metadata: UploadFile = File(...),
        metadata_checksum: str = Form(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    project_repo = ProjectRepository(db)
    project = project_repo.get_project_by_id(project_id)

    if not project:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Project not found")

    if project.user_id != current_user.id:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="You do not have permission to modify this project")

    if not metadata.filename.endswith(".csv"):
        raise HTTPException(status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE, detail="Only CSV files are supported")

    if project.metadata_file:
        await delete_file_async(metadata_sample_path(current_user.id, project_id, project.metadata_file))

    metadata_path = metadata_sample_path(current_user.id, project_id, metadata.filename)

    computed_metadata_checksum = await write_uploadfile_chunked_async(metadata, metadata_path)

    if computed_metadata_checksum != metadata_checksum:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Metadata checksum mismatch")

    return project_repo.upload_project_metadata(project_id, metadata.filename)

@router.put("/{project_id}", status_code=HTTPStatus.OK, response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    project_repo = ProjectRepository(db)
    project = project_repo.get_project_by_id(project_id)

    if not project:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Project not found")

    if project.user_id != current_user.id:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="You do not have permission to modify this project")

    return project_repo.update_project(project_id, project_data)

@router.delete("/{project_id}", status_code=HTTPStatus.NO_CONTENT)
async def delete_project(
        project_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    project_repo = ProjectRepository(db)
    project = project_repo.get_project_by_id(project_id)

    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="You do not have permission to delete this project")

    project_repo.delete_project_by_id(project_id)
    return Response(status_code=HTTPStatus.NO_CONTENT)
