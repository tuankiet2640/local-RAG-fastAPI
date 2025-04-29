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
from app.config import get_settings
import logging
from .rag.factory import create_providers
from ..models.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, index_path: str):
        self.vector_store = None
        self.index_path = index_path

    def load_existing_index(self, embeddings):
        if os.path.exists(self.index_path) and self.vector_store is None:
            try:
                logger.info("Attempting to load FAISS index...")
                self.vector_store = FAISS.load_local(
                    self.index_path, embeddings, allow_dangerous_deserialization=True
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

    def save_index(self):
        if self.vector_store:
            self.vector_store.save_local(self.index_path)

    def add_documents(self, docs, embeddings):
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, embeddings)
        else:
            self.vector_store.add_documents(docs)
        self.save_index()

    def as_retriever(self):
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

class RAGServiceWrapper:
    def __init__(self, embedding_provider=None, llm_provider=None, index_path=None):
        self.settings = get_settings()
        if not embedding_provider or not llm_provider or not index_path:
            embedding_provider, llm_provider, index_path = create_providers()
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.vector_store_manager = VectorStoreManager(index_path)

    def initialize_rag(self, documents: List[Document], kb: KnowledgeBase = None):
        all_docs = []
        for doc in documents:
            # Determine chunking parameters (document > KB > default)
            chunk_size = doc.chunk_size or (kb.chunk_size if kb and kb.chunk_size else 1000)
            chunk_overlap = doc.chunk_overlap or (kb.chunk_overlap if kb and kb.chunk_overlap else 200)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            try:
                if doc.file_type == "pdf":
                    loader = PyPDFLoader(doc.file_path)
                elif doc.file_type == "csv":
                    loader = CSVLoader(doc.file_path)
                elif doc.file_type == "txt":
                    loader = TextLoader(doc.file_path)
                else:
                    raise ValueError(f"Unsupported file type: {doc.file_type}")
                loaded_docs = loader.load()
                chunks = text_splitter.split_documents(loaded_docs)
                all_docs.extend(chunks)
            except Exception as e:
                logger.error(f"Error loading document {doc.file_path}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")
        # Determine embedding model (document > KB > default)
        embedding_model = None
        for doc in documents:
            if doc.embedding_model:
                embedding_model = doc.embedding_model
                break
        if not embedding_model and kb and kb.embedding_model:
            embedding_model = kb.embedding_model
        embeddings = self.embedding_provider.get_embeddings() if not embedding_model else self.embedding_provider.__class__(model_name=embedding_model).get_embeddings()
        self.vector_store_manager.add_documents(all_docs, embeddings)
        return {"status": "success", "message": f"Processed {len(documents)} documents"}

    def get_rag_chain(self, conversation_id: str, messages: List[Message], kb: KnowledgeBase = None):
        embeddings = self.embedding_provider.get_embeddings() if not kb or not kb.embedding_model else self.embedding_provider.__class__(model_name=kb.embedding_model).get_embeddings()
        if not self.vector_store_manager.load_existing_index(embeddings):
            logger.error("Vector store not initialized. Please upload documents first.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store not initialized. Please upload documents first."
            )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        for msg in messages:
            if msg.role == "user":
                memory.chat_memory.add_user_message(msg.content)
            elif msg.role == "assistant":
                memory.chat_memory.add_ai_message(msg.content)
        prompt_template = kb.prompt_template if kb and kb.prompt_template else """
        You are a helpful assistant that answers questions based on the provided context.
        Context: {context}
        Chat History: {chat_history}
        Human: {question}
        Assistant:""
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=prompt_template
        )
        llm = self.llm_provider.get_llm()
        retriever_params = kb.retriever_params if kb and kb.retriever_params else {"k": 5, "search_type": "similarity"}
        retriever = self.vector_store_manager.vector_store.as_retriever(**retriever_params)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        return chain

    async def process_uploaded_files(self, files: List[UploadFile], upload_dir: str):
        documents = []
        uploaded_files = []
        for file in files:
            try:
                file_extension = validate_file(file)
                file_path = await save_uploaded_file(file, upload_dir)
                metadata = extract_metadata(file_path, file_extension)
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
