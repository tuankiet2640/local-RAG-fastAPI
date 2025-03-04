from fastapi import APIRouter, HTTPException, status, UploadFile, File
from typing import List
import shutil
import os
import logging
from ....models.document import Document, DocumentsUploadRequest
from ....services.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)
rag_service = RAGService()

@router.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_documents(documents: DocumentsUploadRequest):
    """Upload documents by path to build or update the RAG knowledge base"""
    try:
        result = rag_service.initialize_rag(documents.documents)
        return result
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading documents: {str(e)}"
        )

@router.post("/upload-files", status_code=status.HTTP_201_CREATED)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files directly to the server and process them for RAG"""
    
    try:
        # Create upload directory if it doesn't exist
        UPLOAD_DIR = "uploaded_files"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        documents = []
        
        for file in files:
            # Determine file type from extension
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
            
            # Save the uploaded file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            
            try:
                # Create a unique filename to avoid overwriting
                base, ext = os.path.splitext(file_path)
                counter = 1
                while os.path.exists(file_path):
                    file_path = f"{base}_{counter}{ext}"
                    counter += 1
                    
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            finally:
                file.file.close()
            
            # Add to documents list
            documents.append(Document(file_path=file_path, file_type=file_type))
        
        # Process the documents
        result = rag_service.initialize_rag(documents)
        
        return {
            "status": "success", 
            "message": f"Successfully uploaded and processed {len(documents)} files",
            "files": [os.path.basename(doc.file_path) for doc in documents]
        }
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error processing uploaded files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing uploaded files: {str(e)}"
        )