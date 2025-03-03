import os
import magic
import filetype
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException
from typing import Dict, Any
import aiofiles
from fastapi import UploadFile

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_FILE_TYPES = ["pdf", "csv", "txt"]

def validate_file(file: UploadFile):
    """Validate file size and type."""
    file_size = 0
    for chunk in file.file:
        file_size += len(chunk)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds the maximum allowed size of {MAX_FILE_SIZE / 1024 / 1024} MB"
            )
    
    # Reset file pointer after reading
    file.file.seek(0)
    
    # Detect file type using python-magic
    file_type = magic.from_buffer(file.file.read(1024), mime=True)
    file.file.seek(0)  # Reset file pointer
    
    # Validate file type
    file_extension = filetype.guess_extension(file.file.read(1024))
    file.file.seek(0)  # Reset file pointer
    
    if file_extension not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Supported types are {ALLOWED_FILE_TYPES}"
        )
    
    return file_extension

def generate_unique_filename(filename: str) -> str:
    """Generate a unique filename to avoid conflicts."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{timestamp}_{filename}"

async def save_uploaded_file(file: UploadFile, upload_dir: str) -> str:
    """Save the uploaded file asynchronously."""
    unique_filename = generate_unique_filename(file.filename)
    file_path = os.path.join(upload_dir, unique_filename)
    
    async with aiofiles.open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # Read in 1 MB chunks
            await buffer.write(chunk)
    
    return file_path