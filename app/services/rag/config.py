from typing import Optional
from pydantic import BaseModel
from enum import Enum

class ProviderType(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"

class OllamaConfig(BaseModel):
    embedding_model: str = "llama2"
    llm_model: str = "mistral"
    temperature: float = 0.7

class OpenAIConfig(BaseModel):
    api_key: str
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7

class AzureOpenAIConfig(BaseModel):
    api_key: str
    deployment_name: str
    api_base: str
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-35-turbo"
    temperature: float = 0.7

class RAGConfig(BaseModel):
    provider_type: ProviderType
    ollama_config: Optional[OllamaConfig] = None
    openai_config: Optional[OpenAIConfig] = None
    azure_openai_config: Optional[AzureOpenAIConfig] = None
    index_path: str = "faiss_index"

    @classmethod
    def from_env(cls):
        """Create RAGConfig from environment variables"""
        import os
        from dotenv import load_dotenv

        load_dotenv()

        provider_type = ProviderType(os.getenv("RAG_PROVIDER_TYPE", "ollama"))

        if provider_type == ProviderType.OLLAMA:
            return cls(
                provider_type=provider_type,
                ollama_config=OllamaConfig(
                    embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "llama2"),
                    llm_model=os.getenv("OLLAMA_LLM_MODEL", "mistral"),
                    temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
                )
            )

        elif provider_type == ProviderType.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            return cls(
                provider_type=provider_type,
                openai_config=OpenAIConfig(
                    api_key=api_key,
                    embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
                    llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo"),
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
                )
            )

        elif provider_type == ProviderType.AZURE_OPENAI:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            api_base = os.getenv("AZURE_OPENAI_API_BASE")

            if not all([api_key, deployment_name, api_base]):
                raise ValueError(
                    "AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME, and "
                    "AZURE_OPENAI_API_BASE environment variables are required"
                )

            return cls(
                provider_type=provider_type,
                azure_openai_config=AzureOpenAIConfig(
                    api_key=api_key,
                    deployment_name=deployment_name,
                    api_base=api_base,
                    embedding_model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
                    llm_model=os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-35-turbo"),
                    temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.7"))
                )
            )

        raise ValueError(f"Unsupported provider type: {provider_type}") 