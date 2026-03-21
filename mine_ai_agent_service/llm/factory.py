from langchain_core.language_models import BaseChatModel

from mine_ai_agent_service.config import settings
from mine_ai_agent_service.llm.lmstudio_provider import LMStudioProvider


def get_llm() -> BaseChatModel:
    provider = settings.MODEL_PROVIDER
    if provider == 'lmstudio':
        return LMStudioProvider(model=settings.LMSTUDIO_MODEL).get_llm()
    raise ValueError(f'MODEL_PROVIDER não suportado: {provider}')
