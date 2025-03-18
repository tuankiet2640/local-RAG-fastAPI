from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.llms import Ollama, OpenAI, AzureOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from ..models.document import Document
from ..models.chat_message import Message
from fastapi import HTTPException, status
import os
import logging

logger = logging.getLogger(__name__)

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def get_embeddings(self):
        pass

class BaseLLMProvider(ABC):
    @abstractmethod
    def get_llm(self):
        pass

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name

    def get_embeddings(self):
        return OllamaEmbeddings(model=self.model_name)

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_embeddings(self):
        return OpenAIEmbeddings(openai_api_key=self.api_key)

class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, deployment_name: str, api_base: str):
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_base = api_base

    def get_embeddings(self):
        return AzureOpenAIEmbeddings(
            openai_api_key=self.api_key,
            deployment=self.deployment_name,
            azure_endpoint=self.api_base
        )

class OllamaLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "mistral", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self):
        return Ollama(model=self.model_name, temperature=self.temperature)

class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self):
        return OpenAI(
            openai_api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )

class AzureOpenAILLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, deployment_name: str, api_base: str, temperature: float = 0.7):
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_base = api_base
        self.temperature = temperature

    def get_llm(self):
        return AzureOpenAI(
            openai_api_key=self.api_key,
            deployment_name=self.deployment_name,
            azure_endpoint=self.api_base,
            temperature=self.temperature
        )

class RAGService:
    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        llm_provider: BaseLLMProvider,
        index_path: str = "faiss_index"
    ):
        self.vector_store = None
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.index_path = index_path

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

        embeddings = self.embedding_provider.get_embeddings()

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(all_docs, embeddings)
        else:
            self.vector_store.add_documents(all_docs)

        # Save the index for future use
        self.vector_store.save_local(self.index_path)

        return {"status": "success", "message": f"Processed {len(documents)} documents"}

    def load_existing_index(self):
        """Load existing FAISS index if available"""
        if os.path.exists(self.index_path) and self.vector_store is None:
            try:
                logger.info("Attempting to load FAISS index...")
                embeddings = self.embedding_provider.get_embeddings()
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

        # Get LLM from provider
        llm = self.llm_provider.get_llm()

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