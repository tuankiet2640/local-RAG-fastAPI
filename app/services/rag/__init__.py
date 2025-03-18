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

def create_rag_service() -> RAGService:
    """Create a RAGService instance with configuration from environment variables"""
    
    embedding_provider, llm_provider, index_path = create_providers()
    
    return RAGService(
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        index_path=index_path
    )

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