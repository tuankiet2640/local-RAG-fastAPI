from fastapi import APIRouter, HTTPException, status, UploadFile, File, Body, Query
from typing import List, Optional
import shutil
import os
import logging
from ....models.document import Document, DocumentsUploadRequest
from ....models.knowledge_base import KnowledgeBase
from ....services.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)
rag_service = RAGService()  # Uses provider factory by default

# In-memory store for demonstration (replace with DB in production)
documents_store: List[Document] = []

@router.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_documents(documents: DocumentsUploadRequest, kb: KnowledgeBase = Body(None)):
    """Upload documents by path to build or update the RAG knowledge base, supporting advanced fields."""
    try:
        # Pass KB object for per-KB parameterization
        result = rag_service.initialize_rag(documents.documents, kb=kb)
        return result
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading documents: {str(e)}"
        )

@router.post("/upload-files", status_code=status.HTTP_201_CREATED)
async def upload_files(files: List[UploadFile] = File(...), kb: KnowledgeBase = Body(None)):
    """Upload files directly to the server and process them for RAG, supporting advanced document fields and per-KB parameterization."""
    try:
        UPLOAD_DIR = "uploaded_files"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        documents = []
        for file in files:
            file_extension = os.path.splitext(file.filename)[1].lower()
            if file_extension == ".pdf":
                file_type = "pdf"
            elif file_extension == ".csv":
                file_type = "csv"
            elif file_extension in [".txt", ".text"]:
                file_type = "txt"
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_extension}. Supported types are .pdf, .csv, and .txt"
                )
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            try:
                base, ext = os.path.splitext(file_path)
                counter = 1
                while os.path.exists(file_path):
                    file_path = f"{base}_{counter}{ext}"
                    counter += 1
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            finally:
                file.file.close()
            # Extract metadata and populate advanced fields
            from ....utils.metadata_utils import extract_metadata
            metadata = extract_metadata(file_path, file_type)
            document = Document(
                file_path=file_path,
                file_type=file_type,
                metadata=metadata,
                kb_ids=[kb.id] if kb and kb.id else [],
                # You can extend here to accept tags, status, etc. from request if needed
            )
            documents.append(document)
        # Process the documents with per-KB parameterization
        result = rag_service.initialize_rag(documents, kb=kb)
        return {
            "status": "success",
            "message": f"Successfully uploaded and processed {len(documents)} files",
            "files": [os.path.basename(doc.file_path) for doc in documents]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing uploaded files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing uploaded files: {str(e)}"
        )

@router.get("/documents", response_model=List[Document])
def list_documents(
    kb_id: Optional[str] = Query(None, description="Filter by KnowledgeBase ID"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    results = documents_store
    if kb_id:
        results = [doc for doc in results if kb_id in doc.kb_ids]
    if tag:
        results = [doc for doc in results if doc.tags and tag in doc.tags]
    if status:
        results = [doc for doc in results if doc.status == status]
    return results

@router.get("/documents/{file_path}", response_model=Document)
def get_document(file_path: str):
    for doc in documents_store:
        if doc.file_path == file_path:
            return doc
    raise HTTPException(status_code=404, detail="Document not found")

@router.post("/documents", response_model=Document, status_code=status.HTTP_201_CREATED)
def create_document(doc: Document):
    documents_store.append(doc)
    return doc

@router.put("/documents/{file_path}", response_model=Document)
def update_document(file_path: str, doc_update: Document):
    for i, doc in enumerate(documents_store):
        if doc.file_path == file_path:
            documents_store[i] = doc_update
            return doc_update
    raise HTTPException(status_code=404, detail="Document not found")

@router.delete("/documents/{file_path}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(file_path: str):
    for i, doc in enumerate(documents_store):
        if doc.file_path == file_path:
            del documents_store[i]
            return
    raise HTTPException(status_code=404, detail="Document not found")