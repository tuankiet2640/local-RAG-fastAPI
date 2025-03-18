import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional
from enum import Enum
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class ProviderType(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"

class OllamaSettings(BaseSettings):
    embedding_model: str = "llama2"
    llm_model: str = "mistral"
    temperature: float = 0.7

class OpenAISettings(BaseSettings):
    api_key: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7

class AzureOpenAISettings(BaseSettings):
    api_key: Optional[str] = None
    deployment_name: Optional[str] = None
    api_base: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-35-turbo"
    temperature: float = 0.7

class Settings(BaseSettings):
    # App settings
    app_name: str = "Local RAG FastAPI"
    debug: bool = True
    
    # Provider settings
    provider_type: ProviderType = ProviderType.OLLAMA
    
    # FAISS settings
    faiss_index_path: str = "faiss_index"
    
    # Ollama settings
    ollama_embedding_model: str = "llama2"
    ollama_llm_model: str = "mistral"
    ollama_temperature: float = 0.7
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_llm_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.7
    
    # Azure OpenAI settings
    azure_openai_api_key: Optional[str] = None
    azure_openai_deployment_name: Optional[str] = None
    azure_openai_api_base: Optional[str] = None
    azure_openai_embedding_model: str = "text-embedding-ada-002"
    azure_openai_llm_model: str = "gpt-35-turbo"
    azure_openai_temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_ollama_settings(self) -> OllamaSettings:
        return OllamaSettings(
            embedding_model=self.ollama_embedding_model,
            llm_model=self.ollama_llm_model,
            temperature=self.ollama_temperature
        )
    
    def get_openai_settings(self) -> OpenAISettings:
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        return OpenAISettings(
            api_key=self.openai_api_key,
            embedding_model=self.openai_embedding_model,
            llm_model=self.openai_llm_model,
            temperature=self.openai_temperature
        )
    
    def get_azure_openai_settings(self) -> AzureOpenAISettings:
        if not all([self.azure_openai_api_key, self.azure_openai_deployment_name, self.azure_openai_api_base]):
            raise ValueError("Azure OpenAI API key, deployment name, and API base are required")
            
        return AzureOpenAISettings(
            api_key=self.azure_openai_api_key,
            deployment_name=self.azure_openai_deployment_name,
            api_base=self.azure_openai_api_base,
            embedding_model=self.azure_openai_embedding_model,
            llm_model=self.azure_openai_llm_model,
            temperature=self.azure_openai_temperature
        )

@lru_cache()
def get_settings() -> Settings:
    """Get application settings, cached for performance"""
    return Settings() 