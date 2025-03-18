from langchain_community.llms import Ollama, OpenAI, AzureOpenAI
from ..base import BaseLLMProvider

class OllamaLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "mistral", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self):
        return Ollama(model=self.model_name, temperature=self.temperature)

class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self):
        return OpenAI(
            openai_api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )

class AzureOpenAILLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, deployment_name: str, api_base: str, temperature: float = 0.7):
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_base = api_base
        self.temperature = temperature

    def get_llm(self):
        return AzureOpenAI(
            openai_api_key=self.api_key,
            deployment_name=self.deployment_name,
            azure_endpoint=self.api_base,
            temperature=self.temperature
        ) 