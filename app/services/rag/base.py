from abc import ABC, abstractmethod

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def get_embeddings(self):
        """Get the embedding model instance"""
        pass

class BaseLLMProvider(ABC):
    @abstractmethod
    def get_llm(self):
        """Get the LLM model instance"""
        pass 