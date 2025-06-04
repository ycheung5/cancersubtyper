import os
from http import HTTPStatus
from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import joinedload

from helpers.timezone import convert_to_user_tz, get_utc_time
from models import Project
from repository.base_repository import BaseRepository
from schemas.projects import ProjectCreate, ProjectResponse, ProjectUpdate
from util.path_untils import delete_directory, project_root_path, source_sample_file_path, target_sample_file_path


class ProjectRepository(BaseRepository):
    def create_project(self, project_data: ProjectCreate, user_id: int) -> ProjectResponse:
        existing_project = self.get_project_by_name(project_data.name)
        if existing_project:
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT,
                detail="Project name already taken"
            )

        new_project = Project(
            user_id=user_id,
            name=project_data.name,
            tumor_type=project_data.tumor_type,
            description=project_data.description,
            created_at=get_utc_time(),
            edited_at=get_utc_time(),
        )

        self.db.add(new_project)
        self.db.commit()
        self.db.refresh(new_project)

        return self.to_project_response(new_project)

    def update_project(self, project_id: int, project_data: ProjectUpdate) -> ProjectResponse:
        project = self.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Project not found")

        if project_data.name:
            existing_project = self.get_project_by_name(project_data.name)
            if existing_project and existing_project.id != project.id:
                raise HTTPException(
                    status_code=HTTPStatus.CONFLICT, detail="Project name already taken"
                )
            project.name = project_data.name

        if project_data.description:
            project.description = project_data.description

        if project_data.tumor_type:
            project.tumor_type = project_data.tumor_type

        project.edited_at = get_utc_time()

        self.db.commit()
        self.db.refresh(project)

        return self.to_project_response(project)

    def upload_project_samples(
        self,
        project_id: int,
        target_filename: str,
        source_filename: str,
    ) -> ProjectResponse:
        project = self.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Project not found")

        project.target_file = target_filename
        project.source_file = source_filename
        project.edited_at = get_utc_time()

        self.db.commit()
        self.db.refresh(project)

        return self.to_project_response(project)

    def upload_project_metadata(
        self, project_id: int, metadata_filename: str
    ) -> ProjectResponse:
        project = self.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Project not found")

        project.metadata_file = metadata_filename
        project.edited_at = get_utc_time()

        self.db.commit()
        self.db.refresh(project)

        return self.to_project_response(project)

    def get_project_by_user_id(self, user_id: int, user_timezone: str) -> List[ProjectResponse]:
        projects = (
            self.db.query(Project)
            .filter(Project.user_id == user_id)
            .options(joinedload(Project.jobs))
            .order_by(Project.created_at.desc())
            .all()
        )

        return [self.to_project_response(project, user_timezone) for project in projects]

    def get_project_by_id(self, project_id: int) -> Optional[Project]:
        return (
            self.db.query(Project)
            .options(joinedload(Project.jobs))
            .filter(Project.id == project_id)
            .first()
        )

    def get_project_by_name(self, project_name: str) -> Optional[Project]:
        return self.db.query(Project).filter(Project.name == project_name).first()

    def delete_project_by_id(self, project_id: int):
        project = self.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Project not found")

        user = project.owner
        bytes_to_free = 0

        # Only subtract sizes for source and target files
        if project.source_file:
            source_path = source_sample_file_path(user.id, project.id, project.source_file)
            if os.path.exists(source_path):
                bytes_to_free += os.path.getsize(source_path)

        if project.target_file:
            target_path = target_sample_file_path(user.id, project.id, project.target_file)
            if os.path.exists(target_path):
                bytes_to_free += os.path.getsize(target_path)

        # Delete all project-related files
        try:
            root_path = project_root_path(user.id, project.id)
            delete_directory(root_path)
        except Exception:
            pass  # Silently continue if files cannot be deleted

        # Update DB
        try:
            if user and bytes_to_free > 0:
                user.storage_used = max(0, user.storage_used - bytes_to_free)

            self.db.delete(project)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Failed to delete project",
            )

    def to_project_response(self, project: Project, user_timezone: Optional[str] = None) -> ProjectResponse:
        return ProjectResponse(
            id=project.id,
            name=project.name,
            tumor_type=project.tumor_type,
            user_id=project.user_id,
            description=project.description,
            target_file=project.target_file,
            source_file=project.source_file,
            metadata_file=project.metadata_file,
            active=any(
                job.status not in ["Completed", "Failed"]
                for job in (getattr(project, "jobs", []) or [])
            ),
            created_at=convert_to_user_tz(project.created_at, user_timezone)
            if user_timezone else project.created_at,
            edited_at=convert_to_user_tz(project.edited_at, user_timezone)
            if user_timezone else project.edited_at,
        )
