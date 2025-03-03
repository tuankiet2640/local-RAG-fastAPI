from pydantic import BaseModel, Field
from typing import List

class Document(BaseModel):
    file_path: str
    file_type: str = Field(..., description="Supported types: pdf, csv, txt")

class DocumentsUploadRequest(BaseModel):
    documents: List[Document]

