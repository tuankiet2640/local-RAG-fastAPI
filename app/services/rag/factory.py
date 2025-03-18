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
from .config import RAGConfig, ProviderType

def create_providers(config: RAGConfig) -> Tuple[BaseEmbeddingProvider, BaseLLMProvider]:
    """Create embedding and LLM providers based on configuration"""
    
    if config.provider_type == ProviderType.OLLAMA:
        if not config.ollama_config:
            raise ValueError("Ollama configuration is required")
        
        return (
            OllamaEmbeddingProvider(model_name=config.ollama_config.embedding_model),
            OllamaLLMProvider(
                model_name=config.ollama_config.llm_model,
                temperature=config.ollama_config.temperature
            )
        )

    elif config.provider_type == ProviderType.OPENAI:
        if not config.openai_config:
            raise ValueError("OpenAI configuration is required")
        
        return (
            OpenAIEmbeddingProvider(api_key=config.openai_config.api_key),
            OpenAILLMProvider(
                api_key=config.openai_config.api_key,
                model_name=config.openai_config.llm_model,
                temperature=config.openai_config.temperature
            )
        )

    elif config.provider_type == ProviderType.AZURE_OPENAI:
        if not config.azure_openai_config:
            raise ValueError("Azure OpenAI configuration is required")
        
        return (
            AzureOpenAIEmbeddingProvider(
                api_key=config.azure_openai_config.api_key,
                deployment_name=config.azure_openai_config.deployment_name,
                api_base=config.azure_openai_config.api_base
            ),
            AzureOpenAILLMProvider(
                api_key=config.azure_openai_config.api_key,
                deployment_name=config.azure_openai_config.deployment_name,
                api_base=config.azure_openai_config.api_base,
                temperature=config.azure_openai_config.temperature
            )
        )

    raise ValueError(f"Unsupported provider type: {config.provider_type}") 