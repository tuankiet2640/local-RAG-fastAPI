from abc import ABC, abstractmethod
from typing import Any

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def get_embeddings(self) -> Any:
        """Get the embedding model instance"""
        pass

class BaseLLMProvider(ABC):
    @abstractmethod
    def get_llm(self) -> Any:
        """Get the LLM model instance"""
        pass