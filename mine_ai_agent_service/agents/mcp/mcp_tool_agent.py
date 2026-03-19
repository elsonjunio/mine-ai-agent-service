import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from mcp.types import Tool

from mine_ai_agent_service.agents.base import BaseAgent
from mine_ai_agent_service.agents.mcp.events import MCPCallback, MCPEvent
from mine_ai_agent_service.mcp.mcp_client import MCPClientFactory

logger = logging.getLogger(__name__)

_ARG_EXTRACTION_SYSTEM = """\
Você é um extrator de argumentos para chamadas de ferramentas MCP.

Ferramenta  : {tool_name}
Descrição   : {tool_description}

Schema de entrada (JSON Schema):
{input_schema}

Chaves disponíveis no contexto: {context_keys}

Regras:
- Extraia os argumentos necessários a partir do prompt e das chaves de contexto.
- Para valores que existem no contexto, use o placeholder {{{{chave}}}} (ex: {{{{token}}}}).
- Para valores que devem vir do prompt, extraia diretamente.
- Retorne APENAS um objeto JSON válido com os argumentos. Sem explicações.\
"""

# Captura blocos ```json ... ``` ou ``` ... ``` (com ou sem linguagem declarada)
_FENCED_BLOCK_RE = re.compile(
    r'```(?:json|JSON)?\s*\n?(.*?)\n?\s*```',
    re.DOTALL,
)


def _parse_json_from_llm(raw: str) -> dict[str, Any]:
    """Extrai e valida um objeto JSON de uma resposta de LLM.

    Cobre as variações mais comuns de saída:
      - JSON puro                        → {"key": "value"}
      - Bloco markdown com linguagem     → ```json\\n{...}\\n```
      - Bloco markdown sem linguagem     → ```\\n{...}\\n```
      - Texto com JSON embutido          → Aqui está: {"key": "value"}
      - Espaços/newlines extras          → qualquer combinação

    Returns:
        Dict com os argumentos extraídos, ou {} se nenhum JSON válido for encontrado.
    """
    if not raw or not raw.strip():
        return {}

    candidates: list[str] = []

    # 1. Conteúdo de blocos markdown fenced (prioridade máxima)
    for match in _FENCED_BLOCK_RE.finditer(raw):
        candidates.append(match.group(1).strip())

    # 2. String completa sem os blocos (fallback)
    candidates.append(raw.strip())

    for candidate in candidates:
        if not candidate:
            continue

        # Tenta parse direto
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Extrai o primeiro objeto JSON {...} encontrado na string
        brace_start = candidate.find('{')
        brace_end   = candidate.rfind('}')
        if brace_start != -1 and brace_end > brace_start:
            try:
                result = json.loads(candidate[brace_start : brace_end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    return {}


class MCPToolAgent(BaseAgent):
    """Agente dinâmico que representa uma única tool MCP.

    Fluxo de execução:
        PRE_LLM  → LLM extrai argumentos via inputSchema (se necessário)
        PRE_MCP  → execute_tool (chamada ao servidor MCP)
        POST_MCP → formatação do resultado
        POST_LLM → encerramento

    Callbacks:
        Cada callback recebe (event, state, context) e pode retornar
        (novo_state, novo_context) para modificar o fluxo, ou None para
        manter os originais. São executados em ordem de registro.

    Exemplo de callback para injeção de token:

        async def inject_token(event, state, context):
            if event is not MCPEvent.PRE_MCP:
                return None
            token = context.get('token', '')
            state['arguments'] = {
                k: token if v == '{{token}}' else v
                for k, v in state['arguments'].items()
            }
            return state, context

        agent = MCPToolAgent(..., callbacks=[inject_token])
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tool: Tool,
        server_url: str,
        transport: str = 'streamable_http',
        server_headers: dict[str, str] | None = None,
        callbacks: list[MCPCallback] | None = None,
    ) -> None:
        """
        Args:
            llm:            Modelo de linguagem para extração de argumentos.
            tool:           Objeto Tool retornado pelo servidor MCP (contém
                            name, description, inputSchema).
            server_url:     URL do endpoint MCP.
            transport:      'streamable_http' (padrão) ou 'sse'.
            server_headers: Headers fixos enviados ao servidor (ex: API keys).
            callbacks:      Lista de callbacks para interceptar o fluxo.
        """
        super().__init__(llm, callbacks)
        self._tool = tool
        self._server_url = server_url
        self._transport = transport
        self._server_headers = server_headers or {}

    # ------------------------------------------------------------------ #
    # BaseAgent interface
    # ------------------------------------------------------------------ #

    @property
    def agent_name(self) -> str:
        return self._tool.name

    def describe(self) -> str:
        return self._tool.description or ''

    def build_graph(self):
        raise NotImplementedError(
            'MCPToolAgent não usa build_graph(). Use invoke() diretamente.'
        )

    def invoke(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        return asyncio.run(self._async_invoke(prompt, context or {}))

    async def ainvoke(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        """Versão assíncrona de invoke — use dentro de contextos async."""
        return await self._async_invoke(prompt, context or {})

    # ------------------------------------------------------------------ #
    # Fluxo principal
    # ------------------------------------------------------------------ #

    async def _async_invoke(self, prompt: str, context: dict[str, Any]) -> str:
        state: dict[str, Any] = {
            'tool_name': self._tool.name,
            'tool_description': self._tool.description or '',
            'input_schema': self._tool.inputSchema,
            'prompt': prompt,
            'arguments': {},
            'mcp_result': None,
            'final_output': '',
        }

        # PRE_LLM — enriquecer state/context antes da extração de argumentos
        state, context = await self._dispatch(MCPEvent.PRE_LLM, state, context)

        # Extração de argumentos: context direto (sem LLM) ou via LLM
        state['arguments'] = await self._extract_arguments(state, context)

        # PRE_MCP — injetar valores sensíveis, validar argumentos, etc.
        state, context = await self._dispatch(MCPEvent.PRE_MCP, state, context)

        # Chamada ao servidor MCP
        async with MCPClientFactory.create(
            name=self._tool.name,
            server_url=self._server_url,
            transport=self._transport,
            headers=self._server_headers,
        ) as client:
            state['mcp_result'] = await client.execute_tool(
                self._tool.name,
                state['arguments'],
            )

        state['final_output'] = self._format_result(state['mcp_result'])

        # POST_MCP — transformar/normalizar resultado antes de qualquer síntese
        state, context = await self._dispatch(
            MCPEvent.POST_MCP, state, context
        )

        # POST_LLM — encerramento; aqui entraria síntese por LLM se necessário
        state, context = await self._dispatch(
            MCPEvent.POST_LLM, state, context
        )

        return state['final_output']

    # ------------------------------------------------------------------ #
    # Extração de argumentos
    # ------------------------------------------------------------------ #

    async def _extract_arguments(
        self,
        state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Monta o dict de argumentos a partir do inputSchema.

        Estratégia em duas etapas:
        1. Mapeamento direto: campos requeridos que já existem no context
           recebem um placeholder {{chave}} — sem custo de LLM.
        2. LLM: campos restantes que precisam ser extraídos do prompt.
        """
        schema = state['input_schema'] or {}
        required: list[str] = schema.get('required', [])

        direct: dict[str, Any] = {}
        needs_llm: list[str] = []

        for field in required:
            if field in context:
                direct[field] = f'{{{{{field}}}}}'  # ex: {{token}}
            else:
                needs_llm.append(field)

        if not needs_llm:
            logger.debug(
                '[%s] all args resolved from context, skipping LLM',
                self._tool.name,
            )
            return direct

        logger.debug(
            '[%s] LLM needed for args: %s', self._tool.name, needs_llm
        )

        system = _ARG_EXTRACTION_SYSTEM.format(
            tool_name=state['tool_name'],
            tool_description=state['tool_description'],
            input_schema=json.dumps(schema, indent=2),
            context_keys=list(context.keys()),
        )
        response = await self.llm.ainvoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=state['prompt']),
            ]
        )

        extracted: dict[str, Any] = _parse_json_from_llm(
            getattr(response, 'content', '') or ''
        )
        if not extracted:
            logger.warning(
                '[%s] could not extract valid JSON from LLM response, using empty dict',
                self._tool.name,
            )

        # direct tem prioridade: context placeholders não são sobrescritos pela LLM
        return {**extracted, **direct}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_result(result: Any) -> str:
        """Converte o CallToolResult do MCP SDK para string."""
        if result is None:
            return ''
        if hasattr(result, 'content'):
            return '\n'.join(
                item.text for item in result.content if hasattr(item, 'text')
            )
        return str(result)
