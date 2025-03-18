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
from .config import RAGConfig, ProviderType
from .factory import create_providers

def create_rag_service(config: RAGConfig = None) -> RAGService:
    """Create a RAGService instance with the specified configuration"""
    if config is None:
        config = RAGConfig.from_env()
    
    embedding_provider, llm_provider = create_providers(config)
    return RAGService(
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        index_path=config.index_path
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
    'RAGConfig',
    'ProviderType',
    'create_rag_service'
] 