from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class KnowledgeBase(BaseModel):
    id: str = Field(..., description="Unique KB identifier")
    name: str
    description: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: Optional[str] = None
    retriever_params: Optional[Dict[str, Any]] = None  # e.g., {"k": 5, "search_type": "similarity"}
    prompt_template: Optional[str] = None  # Custom prompt for this KB
