from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from mine_ai_agent_service.config import settings
from mine_ai_agent_service.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """Provider para OpenAI."""

    def __init__(self, model: str = 'gpt-4o', temperature: float = 0.0):
        self._model = model
        self._temperature = temperature

    @property
    def provider_name(self) -> str:
        return 'openai'

    def get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=SecretStr(settings.OPENAI_KEY),
            model=self._model,
            temperature=self._temperature,
        )
