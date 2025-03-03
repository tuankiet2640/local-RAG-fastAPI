from PyPDF2 import PdfReader
import csv
from datetime import datetime
import os
from typing import Dict, Any

def extract_metadata(file_path: str, file_type: str) -> Dict[str, Any]:
    """Extract metadata from uploaded files."""
    metadata = {
        "file_path": file_path,
        "file_type": file_type,
        "size": os.path.getsize(file_path),
        "created_at": datetime.fromtimestamp(os.path.getctime(file_path)),
        "modified_at": datetime.fromtimestamp(os.path.getmtime(file_path)),
    }
    
    if file_type == "pdf":
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                metadata.update({
                    "num_pages": len(reader.pages),
                    "author": reader.metadata.get("/Author", ""),
                    "title": reader.metadata.get("/Title", ""),
                })
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
    
    elif file_type == "csv":
        try:
            with open(file_path, "r") as f:
                reader = csv.reader(f)
                metadata["num_rows"] = sum(1 for row in reader)
        except Exception as e:
            logger.error(f"Error extracting CSV metadata: {str(e)}")
    
    elif file_type == "txt":
        try:
            with open(file_path, "r") as f:
                metadata["num_lines"] = sum(1 for line in f)
        except Exception as e:
            logger.error(f"Error extracting TXT metadata: {str(e)}")
    
    return metadata