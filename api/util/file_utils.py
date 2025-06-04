import asyncio
import hashlib
import os
from pathlib import Path
from typing import Union
import gzip
import shutil

import aiofiles
from fastapi import UploadFile

from config import settings


async def write_uploadfile_async(file: UploadFile, filepath: str) -> str:
    """Writes an UploadFile asynchronously in chunks and returns the SHA1 checksum."""
    hasher = hashlib.sha1()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists

    async with aiofiles.open(filepath, "wb") as out_file:
        file.file.seek(0)
        while chunk := await file.read(8192):  # Read in 8KB chunks
            await out_file.write(chunk)
            hasher.update(chunk)

    return hasher.hexdigest()


async def write_uploadfile_chunked_async(file: UploadFile, filepath: str) -> str:
    """Writes an UploadFile asynchronously in user-defined chunks and returns the SHA256 checksum."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists

    hasher = hashlib.sha256()
    async with aiofiles.open(filepath, "wb") as output_file:
        file.file.seek(0)
        chunk_size = settings.file_writer_chunk_size * 1024 * 1024
        while chunk := await file.read(chunk_size):
            await output_file.write(chunk)
            hasher.update(chunk)

    return hasher.hexdigest()


async def delete_file_async(filepath: Union[str, Path]):
    """Deletes a file asynchronously. Ignores missing files and handles OS file locks."""
    filepath = str(filepath)
    try:
        await asyncio.to_thread(os.remove, filepath)
    except FileNotFoundError:
        pass  # Ignore if file doesn't exist
    except PermissionError:
        await asyncio.sleep(0.1)  # Retry in case of OS lock
        try:
            await asyncio.to_thread(os.remove, filepath)
        except PermissionError:
            pass  # Still locked, ignore


def file_iterator(file_path: str, chunk_size: int = 1024 * 1024):
    """Yields file content in chunks for streaming large files."""
    with open(file_path, "rb") as file:
        while chunk := file.read(chunk_size):
            yield chunk

async def compute_sha256(file: UploadFile) -> str:
    """Computes SHA-256 checksum for the uploaded file."""
    hasher = hashlib.sha256()
    while chunk := await file.read(4096):  # Read file in chunks
        hasher.update(chunk)
    await file.seek(0)  # Reset file pointer
    return hasher.hexdigest()

async def decompress_gzip_file(gz_path: Union[str, Path], output_csv_path: Union[str, Path]):
    """Decompresses a .gz file into a .csv file."""
    gz_path = str(gz_path)
    output_csv_path = str(output_csv_path)
    await asyncio.to_thread(os.makedirs, os.path.dirname(output_csv_path), exist_ok=True)
    await asyncio.to_thread(_decompress_gzip_file_blocking, gz_path, output_csv_path)

def _decompress_gzip_file_blocking(gz_path: str, output_csv_path: str):
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_csv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
