import logging

from langchain_core.language_models import BaseChatModel

from mine_ai_agent_service.agents.mcp.events import MCPCallback
from mine_ai_agent_service.agents.mcp.mcp_tool_agent import MCPToolAgent
from mine_ai_agent_service.mcp.mcp_client import MCPClientFactory

logger = logging.getLogger(__name__)


async def load_mcp_agents(
    server_urls: list[str],
    llm: BaseChatModel,
    transport: str = 'streamable_http',
    server_headers: dict[str, str] | None = None,
    callbacks: list[MCPCallback] | None = None,
) -> dict[str, MCPToolAgent]:
    """Descobre tools em cada servidor MCP e instancia um MCPToolAgent por tool.

    Args:
        server_urls:    Lista de URLs dos servidores MCP.
        llm:            Modelo de linguagem repassado a cada agente.
        transport:      Transporte MCP: 'streamable_http' (padrão) ou 'sse'.
        server_headers: Headers HTTP fixos enviados ao servidor (ex: API key).
        callbacks:      Callbacks de ciclo de vida repassados a cada agente.

    Returns:
        Dicionário {tool_name: MCPToolAgent} pronto para uso no planner/executor.
    """
    agents: dict[str, MCPToolAgent] = {}

    for url in server_urls:
        logger.info('conectando ao servidor MCP: %s', url)
        try:
            async with MCPClientFactory.create(
                name=url,
                server_url=url,
                transport=transport,
                headers=server_headers,
            ) as client:
                tools = await client.list_tools()

            logger.info('  %d tool(s) encontrada(s) em %s', len(tools), url)

            for tool in tools:
                agents[tool.name] = MCPToolAgent(
                    llm=llm,
                    tool=tool,
                    server_url=url,
                    transport=transport,
                    server_headers=server_headers,
                    callbacks=callbacks,
                )
                logger.info('  [+] agente registrado: %s', tool.name)

        except Exception as exc:
            logger.error('falha ao carregar tools de %s: %s', url, exc)

    return agents


def describe_mcp_agents(agents: dict[str, MCPToolAgent]) -> dict[str, str]:
    """Retorna {tool_name: descrição} para alimentar o PlannerAgent."""
    return {name: agent.describe() for name, agent in agents.items()}
