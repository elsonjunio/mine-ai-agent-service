import logging

from mcp import ClientSession
from mcp.client.sse import sse_client

from mine_ai_agent_service.mcp.base import MCPBaseClient

logger = logging.getLogger(__name__)


class MCPSSEClient(MCPBaseClient):
    """Cliente MCP via transporte SSE (Server-Sent Events).

    Use quando o servidor expõe um endpoint SSE — geralmente montado com
    `mcp.sse_app()` no FastMCP, tipicamente em `/sse`.

    Exemplo:
        async with MCPSSEClient('srv', 'http://localhost:8000/sse') as client:
            tools = await client.list_tools()
    """

    async def connect(self) -> None:
        try:
            streams = await self._exit_stack.enter_async_context(
                sse_client(self.server_url, headers=self.headers)
            )
            session = await self._exit_stack.enter_async_context(
                ClientSession(streams[0], streams[1])
            )
            await session.initialize()
            self.session = session
            logger.info(
                '[%s] connected via SSE to %s', self.name, self.server_url
            )
        except Exception as exc:
            logger.error('[%s] SSE connection error: %s', self.name, exc)
            await self.cleanup()
            raise
