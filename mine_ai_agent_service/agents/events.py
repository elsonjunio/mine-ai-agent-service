from enum import Enum
from typing import Any, Awaitable, Callable


class AgentEvent(str, Enum):
    """Momentos do ciclo de execução de qualquer agente.

    PRE_LLM  — antes de enviar mensagens à LLM.
                Útil para enriquecer/reescrever o prompt, injetar contexto.

    POST_LLM — após receber a resposta da LLM.
                Útil para logging, transformação de saída, auditoria.

    PRE_MCP  — antes de chamar execute_tool (exclusivo de MCPToolAgent).
                Ideal para injetar valores sensíveis nos argumentos.

    POST_MCP — após retorno do servidor MCP (exclusivo de MCPToolAgent).
                Útil para normalizar/armazenar o resultado bruto.
    """

    PRE_LLM  = 'pre_llm'
    POST_LLM = 'post_llm'
    PRE_MCP  = 'pre_mcp'
    POST_MCP = 'post_mcp'


# Contrato do callback:
#   - Recebe: evento, state atual, context atual.
#   - Retorna: (novo_state, novo_context) para substituir os originais,
#              ou None para mantê-los inalterados.
#   - Se lançar exceção, a execução é abortada.
AgentCallback = Callable[
    [AgentEvent, dict[str, Any], dict[str, Any]],
    Awaitable[tuple[dict[str, Any], dict[str, Any]] | None],
]
