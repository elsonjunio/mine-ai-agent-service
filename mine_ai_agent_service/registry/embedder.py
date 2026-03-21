from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import OpenAIEmbeddings

from mine_ai_agent_service.config import settings


class Embedder:
    """Solicita embeddings de acordo com o EMBEDDING_PROVIDER configurado em config.py.

    Providers suportados:
      - lmstudio  : OpenAI-compatible via LMSTUDIO_URL com LMSTUDIO_EMBEDDING_MODEL
      - openai    : OpenAI Embeddings via OPENAI_KEY
      - fastembed : FastEmbed offline via FASTEMBED_EMBEDDING_MODEL

    EMBEDDING_PROVIDER é independente de MODEL_PROVIDER — qualquer combinação é válida.
    """

    def __init__(self) -> None:
        provider = settings.EMBEDDING_PROVIDER

        if provider == 'lmstudio':
            self._backend = OpenAIEmbeddings(
                base_url=settings.LMSTUDIO_URL,
                api_key='lm-studio',
                model=settings.LMSTUDIO_EMBEDDING_MODEL,
                check_embedding_ctx_length=False,
            )
        elif provider == 'openai':
            self._backend = OpenAIEmbeddings(
                api_key=settings.OPENAI_KEY,
            )
        elif provider == 'fastembed':
            self._backend = FastEmbedEmbeddings(
                model_name=settings.FASTEMBED_EMBEDDING_MODEL,
            )
        else:
            raise ValueError(
                f'Embedding não suportado para EMBEDDING_PROVIDER="{provider}". '
                'Use "lmstudio", "openai" ou "fastembed".'
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Retorna uma lista de vetores, um por texto."""
        return self._backend.embed_documents(texts)
