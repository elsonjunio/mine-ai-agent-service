import logging

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from mine_ai_agent_service.mcp.base import MCPBaseClient

logger = logging.getLogger(__name__)


class MCPStreamableHTTPClient(MCPBaseClient):
    """Cliente MCP via transporte Streamable HTTP.

    Use quando o servidor expõe um endpoint Streamable HTTP — o padrão atual
    do FastMCP, montado com `app.add_route("/mcp", mcp_app)`.

    Exemplo:
        async with MCPStreamableHTTPClient('srv', 'http://localhost:8000/mcp') as client:
            tools = await client.list_tools()
    """

    async def connect(self) -> None:
        try:
            # streamablehttp_client retorna (read_stream, write_stream, get_session_id)
            streams = await self._exit_stack.enter_async_context(
                streamablehttp_client(self.server_url, headers=self.headers)
            )
            session = await self._exit_stack.enter_async_context(
                ClientSession(streams[0], streams[1])
            )
            await session.initialize()
            self.session = session
            logger.info(
                '[%s] connected via Streamable HTTP to %s',
                self.name,
                self.server_url,
            )
        except Exception as exc:
            logger.error(
                '[%s] Streamable HTTP connection error: %s', self.name, exc
            )
            await self.cleanup()
            raise
