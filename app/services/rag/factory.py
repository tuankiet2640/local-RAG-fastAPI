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
import logging

logger = logging.getLogger(__name__)

def create_providers() -> Tuple[BaseEmbeddingProvider, BaseLLMProvider, str]:
    """Create embedding and LLM providers based on centralized configuration"""
    settings = get_settings()
    logger.info(f"Creating providers for {settings.provider_type}")
    
    try:
        if settings.provider_type == ProviderType.OLLAMA:
            ollama_settings = settings.get_ollama_settings()
            logger.info(f"Using Ollama with embedding model: {ollama_settings.embedding_model}, LLM model: {ollama_settings.llm_model}")
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
            if not openai_settings.api_key:
                logger.error("OpenAI API key is missing")
                raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your environment variables.")
            
            logger.info(f"Using OpenAI with embedding model: {openai_settings.embedding_model}, LLM model: {openai_settings.llm_model}")
            return (
                OpenAIEmbeddingProvider(
                    api_key=openai_settings.api_key,
                    model_name=openai_settings.embedding_model
                ),
                OpenAILLMProvider(
                    api_key=openai_settings.api_key,
                    model_name=openai_settings.llm_model,
                    temperature=openai_settings.temperature
                ),
                settings.faiss_index_path
            )

        elif settings.provider_type == ProviderType.AZURE_OPENAI:
            azure_settings = settings.get_azure_openai_settings()
            if not all([azure_settings.api_key, azure_settings.deployment_name, azure_settings.api_base]):
                logger.error("Azure OpenAI configuration is incomplete")
                raise ValueError("Azure OpenAI requires API key, deployment name, and API base URL.")
            
            logger.info(f"Using Azure OpenAI with deployment: {azure_settings.deployment_name}")
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
    
    except Exception as e:
        logger.error(f"Error creating providers: {str(e)}")
        raise 