from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph

from mine_ai_agent_service.agents.events import AgentCallback, AgentEvent


class BaseAgent(ABC):
    """Classe base para todos os agentes do sistema.

    Arquitetura planner-executor com graph dinâmico:

    - Planner   : analisa a requisição e decompõe em subtarefas.
    - GraphBuilder: monta o StateGraph combinando agentes especializados
                    conforme o plano recebido.
    - Executor  : executa o graph compilado e monitora o progresso.
    - Specialized: agentes de domínio invocados como nós dentro do graph.

    Callbacks:
        Cada callback recebe (event, state, context) e pode retornar
        (novo_state, novo_context) para modificar o fluxo, ou None para
        manter os originais. São executados em ordem de registro.

        Eventos suportados por todo agente:
            PRE_LLM  — antes de invocar a LLM.
            POST_LLM — após receber a resposta da LLM.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        callbacks: list[AgentCallback] | None = None,
    ) -> None:
        self.llm = llm
        self._callbacks: list[AgentCallback] = callbacks or []

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Identificador único do agente."""
        ...

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Constrói e retorna o StateGraph deste agente."""
        ...

    @abstractmethod
    def invoke(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        """Executa o agente com um prompt e contexto opcional.

        Args:
            prompt: instrução principal para o agente.
            context: parâmetros adicionais (ex: {'token': 'eyJ...', 'user': 'x'}).
                     Cada agente decide como utilizá-los.
        """
        ...

    def compile(self) -> Any:
        """Compila o StateGraph e retorna o grafo executável."""
        return self.build_graph().compile()

    async def _dispatch(
        self,
        event: AgentEvent,
        state: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Executa todos os callbacks registrados para o evento."""
        for callback in self._callbacks:
            result = await callback(event, state, context)
            if result is not None:
                state, context = result
        return state, context
