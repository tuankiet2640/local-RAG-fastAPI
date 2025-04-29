from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Document(BaseModel):
    file_path: str
    file_type: str = Field(..., description="Supported types: pdf, csv, txt")
    kb_ids: List[str] = Field(default_factory=list, description="Associated KnowledgeBase IDs")
    metadata: Optional[Dict[str, Any]] = None
    version: Optional[str] = None
    updated_at: Optional[str] = None
    status: Optional[str] = Field(default="active", description="Document status, e.g., active, archived")
    tags: Optional[List[str]] = None
    chunk_size: Optional[int] = None  # Per-document override
    chunk_overlap: Optional[int] = None  # Per-document override
    embedding_model: Optional[str] = None  # Per-document override

class DocumentsUploadRequest(BaseModel):
    documents: List[Document]

