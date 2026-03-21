from pathlib import Path

from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings, extra='allow'):
    # ------------------------------------------------------------------
    # LLM provider
    # Opções: 'lmstudio' | 'openai' | 'anthropic'
    # ------------------------------------------------------------------
    MODEL_PROVIDER: str = 'lmstudio'

    # ------------------------------------------------------------------
    # Embedding provider (independente do MODEL_PROVIDER)
    # Opções:
    #   'lmstudio'  — API compatível com OpenAI rodando localmente via LMStudio
    #   'openai'    — OpenAI Embeddings (requer OPENAI_KEY)
    #   'fastembed' — offline, sem API key; modelo baixado em cache local
    # ------------------------------------------------------------------
    EMBEDDING_PROVIDER: str = 'lmstudio'

    # ------------------------------------------------------------------
    # LMStudio (MODEL_PROVIDER=lmstudio ou EMBEDDING_PROVIDER=lmstudio)
    # ------------------------------------------------------------------
    LMSTUDIO_URL: str = 'http://localhost:1234/v1'
    LMSTUDIO_MODEL: str = 'google/gemma-3n-e4b'
    LMSTUDIO_EMBEDDING_MODEL: str = 'text-embedding-nomic-embed-text-v1.5'

    # ------------------------------------------------------------------
    # FastEmbed (EMBEDDING_PROVIDER=fastembed)
    # Modelos recomendados:
    #   'BAAI/bge-small-en-v1.5'  — leve, rápido (~33 MB)
    #   'BAAI/bge-large-en-v1.5'  — mais preciso (~1.2 GB)
    #   'nomic-ai/nomic-embed-text-v1.5' — multilíngue, equilibrado
    # ------------------------------------------------------------------
    FASTEMBED_EMBEDDING_MODEL: str = 'BAAI/bge-small-en-v1.5'

    # ------------------------------------------------------------------
    # OpenAI (MODEL_PROVIDER=openai ou EMBEDDING_PROVIDER=openai)
    # ------------------------------------------------------------------
    OPENAI_KEY: str = ''
    OPENAI_MODEL: str = 'gpt-4o'

    # ------------------------------------------------------------------
    # Anthropic / Claude (MODEL_PROVIDER=anthropic)
    # Modelos disponíveis: 'claude-opus-4-6' | 'claude-sonnet-4-6' | 'claude-haiku-4-5'
    # ------------------------------------------------------------------
    ANTHROPIC_KEY: str = ''
    ANTHROPIC_MODEL: str = 'claude-opus-4-6'

    # ------------------------------------------------------------------
    # MCP — URLs dos serviços que expõem tools via protocolo MCP
    # ------------------------------------------------------------------
    MCP_URLS: list[str] = ['http://localhost:8000/mcp']

    # ------------------------------------------------------------------
    # Armazenamento local (summaries, índice FAISS)
    # ------------------------------------------------------------------
    DATA_DIR: str = str(_PROJECT_ROOT / 'data')

    # ------------------------------------------------------------------
    # Segurança — segredo para assinar tokens JWT internos
    # ------------------------------------------------------------------
    INTERNAL_TOKEN_SECRET: str = 'super-secret-change-this'

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()
