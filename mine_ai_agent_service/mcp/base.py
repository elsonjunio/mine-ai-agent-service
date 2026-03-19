import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.types import Prompt, Resource, ResourceTemplate, Tool

logger = logging.getLogger(__name__)


class MCPBaseClient(ABC):
    """Interface comum para clientes MCP.

    Toda a lógica compartilhada (list_tools, execute_tool, cleanup, context
    manager) vive aqui. As subclasses só precisam implementar `connect`,
    que difere apenas no transporte utilizado (SSE ou Streamable HTTP).

    Self-contained: sem dependências do restante do projeto.
    Parâmetros recebidos via construtor.
    """

    def __init__(
        self,
        name: str,
        server_url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            name:       Identificador da conexão (usado em logs).
            server_url: URL completa do endpoint MCP.
            headers:    Headers HTTP opcionais (ex: Authorization Bearer).
        """
        self.name = name
        self.server_url = server_url
        self.headers = headers or {}
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._exit_stack: AsyncExitStack = AsyncExitStack()

    # ------------------------------------------------------------------ #
    # Interface obrigatória
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def connect(self) -> None:
        """Estabelece a conexão com o servidor MCP e inicializa a sessão."""

    # ------------------------------------------------------------------ #
    # Operações de listagem
    # ------------------------------------------------------------------ #

    async def list_tools(self) -> list[Tool]:
        """Retorna todas as tools disponíveis no servidor."""
        self._require_session()
        result = await self.session.list_tools()  # type: ignore[union-attr]
        return result.tools

    async def list_resources(self) -> list[Resource]:
        """Retorna todos os resources disponíveis no servidor."""
        self._require_session()
        result = await self.session.list_resources()  # type: ignore[union-attr]
        return result.resources

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """Retorna todos os resource templates disponíveis no servidor."""
        self._require_session()
        result = await self.session.list_resource_templates()  # type: ignore[union-attr]
        return result.resourceTemplates

    async def list_prompts(self) -> list[Prompt]:
        """Retorna todos os prompts disponíveis no servidor."""
        self._require_session()
        result = await self.session.list_prompts()  # type: ignore[union-attr]
        return result.prompts

    # ------------------------------------------------------------------ #
    # Execução de tool
    # ------------------------------------------------------------------ #

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Executa uma tool no servidor com retry automático.

        Args:
            tool_name:  Nome da tool a executar.
            arguments:  Argumentos para a tool.
            retries:    Número máximo de tentativas (padrão 2).
            delay:      Segundos entre tentativas (padrão 1.0).

        Returns:
            Resultado bruto retornado pelo servidor MCP.

        Raises:
            RuntimeError: Se o cliente não estiver conectado.
            Exception:    Se todas as tentativas falharem.
        """
        self._require_session()

        for attempt in range(1, retries + 1):
            try:
                logger.info(
                    '[%s] executing %s (attempt %d/%d)',
                    self.name,
                    tool_name,
                    attempt,
                    retries,
                )
                return await self.session.call_tool(tool_name, arguments)  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning('[%s] %s failed: %s', self.name, tool_name, exc)
                if attempt < retries:
                    logger.info('[%s] retrying in %.1fs...', self.name, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        '[%s] max retries reached for %s.',
                        self.name,
                        tool_name,
                    )
                    raise

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #

    async def cleanup(self) -> None:
        """Fecha a conexão e libera todos os recursos."""
        async with self._cleanup_lock:
            try:
                await self._exit_stack.aclose()
                self.session = None
            except Exception as exc:
                logger.error('[%s] cleanup error: %s', self.name, exc)

    # ------------------------------------------------------------------ #
    # Async context manager
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> 'MCPBaseClient':
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.cleanup()

    # ------------------------------------------------------------------ #
    # Helpers internos
    # ------------------------------------------------------------------ #

    def _require_session(self) -> None:
        if self.session is None:
            raise RuntimeError(
                f'[{self.name}] not connected — call connect() first.'
            )
