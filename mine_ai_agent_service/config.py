from pathlib import Path

from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings, extra='allow'):
    MODEL_PROVIDER: str = 'lmstudio'
    OPENAI_KEY: str = ''
    ANTHROPIC_KEY: str = ''
    LMSTUDIO_URL: str = 'http://localhost:1234/v1'

    MCP_URLS: list[str] = ['http://localhost:8000/mcp']

    LMSTUDIO_EMBEDDING_MODEL: str = 'text-embedding-nomic-embed-text-v1.5'
    DATA_DIR: str = str(_PROJECT_ROOT / 'data')

    REDIS_HOST: str = ''
    REDIS_PORT: int = 0
    REDIS_DB: int = 0

    OPENID_ROLE_CLAIM: str = 'policy'
    ADMIN_ROLE: str = 'consoleAdmin'

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()
