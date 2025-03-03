from pydantic import BaseModel, Field

class Document(BaseModel):
    file_path: str
    file_type: str = Field(..., description="Supported types: pdf, csv, txt")
