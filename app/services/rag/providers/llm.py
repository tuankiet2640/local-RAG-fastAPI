from langchain_openai import AzureOpenAI, OpenAI
from langchain_ollama import OllamaLLM
from ..base import BaseLLMProvider
import logging

logger = logging.getLogger(__name__)

class OllamaLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "mistral", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self):
        try:
            logger.info(f"Initializing Ollama LLM with model: {self.model_name}")
            return OllamaLLM(model=self.model_name, temperature=self.temperature)
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM: {str(e)}")
            raise

class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self):
        try:
            logger.info(f"Initializing OpenAI LLM with model: {self.model_name}")
            return OpenAI(
                api_key=self.api_key,
                model=self.model_name,
                temperature=self.temperature
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAI LLM: {str(e)}")
            raise

class AzureOpenAILLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, deployment_name: str, api_base: str, temperature: float = 0.7):
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_base = api_base
        self.temperature = temperature

    def get_llm(self):
        try:
            logger.info(f"Initializing Azure OpenAI LLM with deployment: {self.deployment_name}")
            return AzureOpenAI(
                api_key=self.api_key,
                deployment_name=self.deployment_name,
                azure_endpoint=self.api_base,
                temperature=self.temperature
            )
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI LLM: {str(e)}")
            raise 