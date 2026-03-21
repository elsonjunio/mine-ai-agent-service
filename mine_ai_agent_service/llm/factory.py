from langchain_core.language_models import BaseChatModel

from mine_ai_agent_service.config import settings
from mine_ai_agent_service.llm.claude_provider import ClaudeProvider
from mine_ai_agent_service.llm.lmstudio_provider import LMStudioProvider
from mine_ai_agent_service.llm.openai_provider import OpenAIProvider


def get_llm() -> BaseChatModel:
    provider = settings.MODEL_PROVIDER
    if provider == 'lmstudio':
        return LMStudioProvider(model=settings.LMSTUDIO_MODEL).get_llm()
    if provider == 'openai':
        return OpenAIProvider(model=settings.OPENAI_MODEL).get_llm()
    if provider == 'anthropic':
        return ClaudeProvider(model=settings.ANTHROPIC_MODEL).get_llm()
    raise ValueError(f'MODEL_PROVIDER não suportado: {provider}')
