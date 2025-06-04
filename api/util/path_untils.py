import os
import shutil

from config import settings

def target_sample_file_path(user_id: int, project_id: int, filename: str) -> str:
    """
    Generate a file path for the target sample file.

    Args:
        user_id (int): ID of the user.
        project_id (int): ID of the project.
        filename (str): Name of the file being uploaded.

    Returns:
        str: Absolute file path.
    """
    return os.path.join(settings.data_dir, f"user_{user_id}", f"project_{project_id}", "target", filename)


def source_sample_file_path(user_id: int, project_id: int, filename: str) -> str:
    """
    Generate a file path for the source sample file.

    Args:
        user_id (int): ID of the user.
        project_id (int): ID of the project.
        filename (str): Name of the file being uploaded.

    Returns:
        str: Absolute file path.
    """
    return os.path.join(settings.data_dir, f"user_{user_id}", f"project_{project_id}", "source", filename)


def metadata_sample_path(user_id: int, project_id: int, filename: str) -> str:
    """
    Generate a file path for the metadata file.

    Args:
        user_id (int): ID of the user.
        project_id (int): ID of the project.
        filename (str): Name of the file being uploaded.

    Returns:
        str: Absolute file path for the metadata file.
    """
    return os.path.join(settings.data_dir, f"user_{user_id}", f"project_{project_id}", "metadata", filename)

def job_preprocessing_path(user_id: int, project_id: int, job_id: int) -> str:
    """
    Generate a file path for the job's preprocessing output.

    Args:
        user_id (int): ID of the user.
        project_id (int): ID of the project.
        job_id (int): ID of the job.

    Returns:
        str: Absolute directory path for the job's preprocessing output.
    """
    return os.path.join(settings.data_dir, f"user_{user_id}", f"project_{project_id}", f"job_{job_id}", "preprocessing")

def job_result_path(user_id: int, project_id: int, job_id: int) -> str:
    """
    Generate a file path for the job's final results.

    Args:
        user_id (int): ID of the user.
        project_id (int): ID of the project.
        job_id (int): ID of the job.

    Returns:
        str: Absolute directory path for the job's final results.
    """
    return os.path.join(settings.data_dir, f"user_{user_id}", f"project_{project_id}", f"job_{job_id}", "results")

def project_root_path(user_id: int, project_id: int) -> str:
    """
    Returns the root directory for a project.
    """
    return os.path.join(settings.data_dir, f"user_{user_id}", f"project_{project_id}")

def delete_directory(path: str):
    """
    Recursively delete a directory and its contents.

    Args:
        path (str): Absolute path to the directory.
    """
    if os.path.exists(path):
        shutil.rmtree(path)