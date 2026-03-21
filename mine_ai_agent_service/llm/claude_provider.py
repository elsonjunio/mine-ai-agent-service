from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from mine_ai_agent_service.config import settings
from mine_ai_agent_service.llm.base import BaseLLMProvider


class ClaudeProvider(BaseLLMProvider):
    """Provider para Anthropic Claude."""

    def __init__(self, model: str = 'claude-opus-4-6', temperature: float = 0.0):
        self._model = model
        self._temperature = temperature

    @property
    def provider_name(self) -> str:
        return 'anthropic'

    def get_llm(self) -> ChatAnthropic:
        return ChatAnthropic(
            api_key=SecretStr(settings.ANTHROPIC_KEY),
            model=self._model,
            temperature=self._temperature,
        )
