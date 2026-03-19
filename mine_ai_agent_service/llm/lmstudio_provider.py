from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from mine_ai_agent_service.config import settings
from mine_ai_agent_service.llm.base import BaseLLMProvider


class LMStudioProvider(BaseLLMProvider):
    """Provider para LMStudio via API compatível com OpenAI."""

    def __init__(self, model: str = 'local-model', temperature: float = 0.0):
        self._model = model
        self._temperature = temperature

    @property
    def provider_name(self) -> str:
        return 'lmstudio'

    def get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=settings.LMSTUDIO_URL,
            api_key=SecretStr('lm-studio'),
            model=self._model,
            temperature=self._temperature,
        )
