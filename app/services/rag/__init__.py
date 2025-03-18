from .service import RAGService
from .base import BaseEmbeddingProvider, BaseLLMProvider
from .providers.embeddings import (
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    AzureOpenAIEmbeddingProvider
)
from .providers.llm import (
    OllamaLLMProvider,
    OpenAILLMProvider,
    AzureOpenAILLMProvider
)
from .factory import create_providers
from app.config import get_settings, ProviderType
import logging
import importlib.util

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed based on the selected provider"""
    settings = get_settings()
    
    if settings.provider_type == ProviderType.OLLAMA:
        if not importlib.util.find_spec("langchain_ollama"):
            logger.error("langchain-ollama package is required for Ollama provider")
            raise ImportError(
                "langchain-ollama package is required. Please install it with: pip install langchain-ollama"
            )
    
    elif settings.provider_type == ProviderType.OPENAI:
        if not importlib.util.find_spec("openai"):
            logger.error("openai package is required for OpenAI provider")
            raise ImportError(
                "openai package is required. Please install it with: pip install openai"
            )
    
    elif settings.provider_type == ProviderType.AZURE_OPENAI:
        if not importlib.util.find_spec("openai"):
            logger.error("openai package is required for Azure OpenAI provider")
            raise ImportError(
                "openai package is required. Please install it with: pip install openai"
            )

def create_rag_service() -> RAGService:
    """Create a RAGService instance with configuration from environment variables"""
    try:
        # Validate dependencies
        check_dependencies()
        
        # Create providers
        embedding_provider, llm_provider, index_path = create_providers()
        
        logger.info("Successfully created RAG service providers")
        
        return RAGService(
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            index_path=index_path
        )
    except Exception as e:
        logger.error(f"Failed to create RAG service: {str(e)}")
        raise

__all__ = [
    'RAGService',
    'BaseEmbeddingProvider',
    'BaseLLMProvider',
    'OllamaEmbeddingProvider',
    'OpenAIEmbeddingProvider',
    'AzureOpenAIEmbeddingProvider',
    'OllamaLLMProvider',
    'OpenAILLMProvider',
    'AzureOpenAILLMProvider',
    'create_rag_service'
] 