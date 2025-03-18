import os
import logging
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from ..models.document import Document
from ..models.chat_message import Message
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, status
from ..utils.metadata_utils import extract_metadata
from ..utils.file_utils import validate_file, save_uploaded_file
from fastapi import UploadFile
from app.config import get_settings, ProviderType
import importlib

logger = logging.getLogger(__name__)

class RAGServiceWrapper:
    def __init__(self):
        self.vector_store = None
        self.settings = get_settings()
        self._rag_service = None
        self._initialize_rag_service()
        
    def _initialize_rag_service(self):
        """Initialize the RAG service with the configured provider"""
        try:
            # Import here to avoid circular imports
            from app.services.rag import create_rag_service
            self._rag_service = create_rag_service()
            logger.info(f"Successfully initialized RAG service with provider: {self.settings.provider_type}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise

    @property
    def embeddings(self):
        """Get the embeddings from the RAG service"""
        if not self._rag_service:
            self._initialize_rag_service()
        return self._rag_service.embedding_provider.get_embeddings()

    @property
    def llm(self):
        """Get the LLM from the RAG service"""
        if not self._rag_service:
            self._initialize_rag_service()
        return self._rag_service.llm_provider.get_llm()

    def initialize_rag(self, documents: List[Document]):
        """Initialize or update the RAG system with documents"""
        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for doc in documents:
            try:
                if doc.file_type == "pdf":
                    loader = PyPDFLoader(doc.file_path)
                elif doc.file_type == "csv":
                    loader = CSVLoader(doc.file_path)
                elif doc.file_type == "txt":
                    loader = TextLoader(doc.file_path)
                else:
                    raise ValueError(f"Unsupported file type: {doc.file_type}")

                documents = loader.load()
                chunks = text_splitter.split_documents(documents)
                all_docs.extend(chunks)

            except Exception as e:
                logger.error(f"Error loading document {doc.file_path}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")

        embeddings = self.embeddings

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(all_docs, embeddings)
        else:
            self.vector_store.add_documents(all_docs)

        # Save the index for future use
        self.vector_store.save_local(self.settings.faiss_index_path)

        return {"status": "success", "message": f"Processed {len(documents)} documents"}

    def load_existing_index(self):
        """Load existing FAISS index if available"""
        if os.path.exists(self.settings.faiss_index_path) and self.vector_store is None:
            try:
                logger.info("Attempting to load FAISS index...")
                embeddings = self.embeddings
                self.vector_store = FAISS.load_local(
                    self.settings.faiss_index_path, embeddings, allow_dangerous_deserialization=True
                )
                logger.info("Successfully loaded existing FAISS index")
                return True
            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")
                return False
        elif self.vector_store is not None:
            logger.info("FAISS index already loaded")
            return True
        else:
            logger.error("FAISS index not found or not initialized")
            return False

    def get_rag_chain(self, conversation_id: str, messages: List[Message]):
        """Get RAG chain with conversation memory"""
        logger.info("Checking if vector store is initialized...")
        if not self.load_existing_index():
            logger.error("Vector store not initialized. Please upload documents first.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store not initialized. Please upload documents first."
            )

        logger.info("Vector store initialized. Creating RAG chain...")

        # Create a memory object for this conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Restore conversation history if it exists
        for msg in messages:
            if msg.role == "user":
                memory.chat_memory.add_user_message(msg.content)
            elif msg.role == "assistant":
                memory.chat_memory.add_ai_message(msg.content)

        # Custom prompt template for better RAG responses
        CUSTOM_PROMPT = """
        You are a helpful assistant that answers questions based on the provided context.

        Context: {context}

        Chat History: {chat_history}

        Human: {question}

        Assistant:"""

        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=CUSTOM_PROMPT
        )

        # Create the RAG chain with the configured LLM
        logger.info(f"Initializing LLM with provider: {self.settings.provider_type}")
        llm = self.llm

        logger.info("Creating retriever from vector store...")
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        logger.info("Creating ConversationalRetrievalChain...")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        logger.info("RAG chain created successfully")
        return chain

    async def process_uploaded_files(self, files: List[UploadFile], upload_dir: str):
        """Process uploaded files and return documents for RAG."""
        documents = []
        uploaded_files = []

        for file in files:
            try:
                # Validate file
                file_extension = validate_file(file)

                # Save file
                file_path = await save_uploaded_file(file, upload_dir)

                # Extract metadata
                metadata = extract_metadata(file_path, file_extension)

                # Add to documents list
                documents.append(Document(file_path=file_path, file_type=file_extension))
                uploaded_files.append({
                    "filename": file.filename,
                    "file_path": file_path,
                    "metadata": metadata
                })

            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing file {file.filename}: {str(e)}"
                )
            finally:
                await file.close()

        return documents, uploaded_files

# Backwards compatibility class name
RAGService = RAGServiceWrapper
