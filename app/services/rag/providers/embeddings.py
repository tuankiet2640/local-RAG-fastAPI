from langchain_community.embeddings import (
    OllamaEmbeddings,
    OpenAIEmbeddings,
    AzureOpenAIEmbeddings
)
from ..base import BaseEmbeddingProvider

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