from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class BaseLLMProvider(ABC):
    """Classe base para providers de LLM.

    Cada provider deve retornar um BaseChatModel compatível com LangGraph,
    permitindo uso de .bind_tools() para injeção dinâmica de tools MCP.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identificador do provider (ex: 'openai', 'anthropic', 'lmstudio')."""
        ...

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """Retorna a instância do modelo configurada e pronta para uso."""
        ...
