"""Callbacks reutilizáveis para o ciclo de execução dos agentes."""

import json
import logging
import re
from typing import Any, Literal

from mine_ai_agent_service.agents.events import AgentEvent

logger = logging.getLogger(__name__)

# Captura blocos ```python ... ``` ou ``` ... ```
_FENCED_CODE_RE = re.compile(r'```(?:python|Python)?\s*\n?(.*?)\n?\s*```', re.DOTALL)

ContentType = Literal['json', 'python', 'text']


def _detect_type(text: str) -> ContentType:
    """Infere o tipo do conteúdo: 'json', 'python' ou 'text'."""
    try:
        json.loads(text)
        return 'json'
    except (json.JSONDecodeError, ValueError):
        pass
    return 'text'


def _make_entry(content: str, content_type: ContentType) -> dict[str, str]:
    return {'content': content, 'type': content_type}


# ------------------------------------------------------------------ #
# inject_token
# ------------------------------------------------------------------ #

async def inject_token(
    event: AgentEvent,
    state: dict[str, Any],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Substitui o placeholder {{token}} pelo JWT real nos argumentos (PRE_MCP).

    A LLM nunca vê o token — recebe apenas o placeholder {{token}}.
    O valor real é injetado aqui, imediatamente antes da chamada ao servidor MCP.
    """
    if event is not AgentEvent.PRE_MCP:
        return None

    token_value = context.get('token', '')
    state['arguments'] = {
        k: token_value if v == '{{token}}' else v
        for k, v in state['arguments'].items()
    }
    return state, context


# ------------------------------------------------------------------ #
# store_result_in_context
# ------------------------------------------------------------------ #

async def store_result_in_context(
    event: AgentEvent,
    state: dict[str, Any],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Armazena o retorno de uma MCP tool no context e substitui por placeholder (POST_MCP).

    Salva em ``context["{tool_name}.result"]`` um dicionário com:
        - ``content``: conteúdo bruto retornado pela tool.
        - ``type``:    tipo inferido do conteúdo ('json' | 'text').

    Substitui ``state["final_output"]`` pelo placeholder ``{{tool_name.result}}``.

    Não faz nada se:
    - O evento não for POST_MCP.
    - A execução tiver falhado (mcp_result.isError == True).
    - O final_output estiver vazio.
    """
    if event is not AgentEvent.POST_MCP:
        return None

    mcp_result = state.get('mcp_result')
    if mcp_result is None:
        return None

    if getattr(mcp_result, 'isError', False):
        logger.debug(
            '[%s] POST_MCP: execução falhou, store_result_in_context ignorado',
            state.get('tool_name', '?'),
        )
        return None

    final_output = state.get('final_output', '')
    if not final_output:
        return None

    tool_name = state.get('tool_name', 'result')
    context_key = f'{tool_name}.result'
    placeholder = f'{{{{{context_key}}}}}'

    context[context_key] = _make_entry(final_output, _detect_type(final_output))
    state['final_output'] = placeholder

    logger.debug(
        '[%s] POST_MCP: resultado armazenado em context["%s"] (type=%s)',
        tool_name,
        context_key,
        context[context_key]['type'],
    )

    return state, context


# ------------------------------------------------------------------ #
# store_code_in_context
# ------------------------------------------------------------------ #

def _extract_code(text: str) -> str:
    """Extrai o conteúdo de um bloco de código markdown.

    Retorna o conteúdo dentro de ``` ... ``` ou ```python ... ```.
    Se não houver bloco fenced, retorna o texto original sem alteração.
    """
    match = _FENCED_CODE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


async def store_code_in_context(
    event: AgentEvent,
    state: dict[str, Any],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Extrai o código gerado pela LLM, armazena no context e substitui por placeholder (POST_LLM).

    Salva em ``context["{agent_name}.result"]`` um dicionário com:
        - ``content``: código limpo (sem delimitadores markdown).
        - ``type``:    sempre 'python'.

    Substitui ``state["final_output"]`` pelo placeholder ``{{agent_name.result}}``.

    Não faz nada se:
    - O evento não for POST_LLM.
    - O final_output estiver vazio.
    """
    if event is not AgentEvent.POST_LLM:
        return None

    final_output = state.get('final_output', '')
    if not final_output:
        return None

    agent_name = state.get('agent_name', 'agent')
    context_key = f'{agent_name}.result'
    placeholder = f'{{{{{context_key}}}}}'

    code = _extract_code(final_output)
    context[context_key] = _make_entry(code, 'python')
    state['final_output'] = placeholder

    logger.debug(
        '[%s] POST_LLM: código extraído e armazenado em context["%s"] (type=python)',
        agent_name,
        context_key,
    )

    return state, context


# ------------------------------------------------------------------ #
# resolve_placeholders
# ------------------------------------------------------------------ #

def resolve_placeholders(text: str, context: dict[str, Any]) -> str:
    """Substitui todos os placeholders {{chave}} no texto pelos valores do context.

    Se o valor no context for um dict com chave ``content`` (formato gerado pelos
    callbacks store_*_in_context), usa o campo ``content`` como valor resolvido.
    Placeholders sem correspondência no context são mantidos intactos.

    Exemplos:
        context = {"python_coder.result": {"content": "def fib(n): ...", "type": "python"}}
        resolve_placeholders("{{python_coder.result}}", context)
        → "def fib(n): ..."

        context = {"list_buckets.result": {"content": '["a","b"]', "type": "json"}}
        resolve_placeholders("{{list_buckets.result}}", context)
        → '["a","b"]'
    """
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in context:
            return match.group(0)
        value = context[key]
        if isinstance(value, dict) and 'content' in value:
            return f"\n```{value['type']}\n{value['content']} \n```"
        return str(value)

    return re.sub(r'\{\{([^}]+)\}\}', _replace, text)
