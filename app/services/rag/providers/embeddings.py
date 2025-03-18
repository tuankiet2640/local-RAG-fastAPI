from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from ..base import BaseEmbeddingProvider
import logging

logger = logging.getLogger(__name__)

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name

    def get_embeddings(self):
        logger.info(f"Initializing Ollama embeddings with model: {self.model_name}")
        try:
            return OllamaEmbeddings(model=self.model_name)
        except Exception as e:
            logger.error(f"Error initializing Ollama embeddings: {str(e)}")
            raise

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.model_name = model_name

    def get_embeddings(self):
        try:
            logger.info(f"Initializing OpenAI embeddings with model: {self.model_name}")
            return OpenAIEmbeddings(
                api_key=self.api_key, 
                model=self.model_name
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
            raise

class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, deployment_name: str, api_base: str):
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_base = api_base

    def get_embeddings(self):
        try:
            logger.info(f"Initializing Azure OpenAI embeddings (deployment: {self.deployment_name})")
            return AzureOpenAIEmbeddings(
                api_key=self.api_key,
                deployment=self.deployment_name,
                azure_endpoint=self.api_base
            )
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI embeddings: {str(e)}")
            raise 