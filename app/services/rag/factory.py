from typing import Tuple
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
from app.config import get_settings, ProviderType

def create_providers() -> Tuple[BaseEmbeddingProvider, BaseLLMProvider, str]:
    """Create embedding and LLM providers based on centralized configuration"""
    settings = get_settings()
    
    if settings.provider_type == ProviderType.OLLAMA:
        ollama_settings = settings.get_ollama_settings()
        return (
            OllamaEmbeddingProvider(model_name=ollama_settings.embedding_model),
            OllamaLLMProvider(
                model_name=ollama_settings.llm_model,
                temperature=ollama_settings.temperature
            ),
            settings.faiss_index_path
        )

    elif settings.provider_type == ProviderType.OPENAI:
        openai_settings = settings.get_openai_settings()
        return (
            OpenAIEmbeddingProvider(api_key=openai_settings.api_key),
            OpenAILLMProvider(
                api_key=openai_settings.api_key,
                model_name=openai_settings.llm_model,
                temperature=openai_settings.temperature
            ),
            settings.faiss_index_path
        )

    elif settings.provider_type == ProviderType.AZURE_OPENAI:
        azure_settings = settings.get_azure_openai_settings()
        return (
            AzureOpenAIEmbeddingProvider(
                api_key=azure_settings.api_key,
                deployment_name=azure_settings.deployment_name,
                api_base=azure_settings.api_base
            ),
            AzureOpenAILLMProvider(
                api_key=azure_settings.api_key,
                deployment_name=azure_settings.deployment_name,
                api_base=azure_settings.api_base,
                temperature=azure_settings.temperature
            ),
            settings.faiss_index_path
        )

    raise ValueError(f"Unsupported provider type: {settings.provider_type}") 