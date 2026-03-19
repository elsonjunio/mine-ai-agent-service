from mine_ai_agent_service.mcp.base import MCPBaseClient
from mine_ai_agent_service.mcp.mcp_sse_client import MCPSSEClient
from mine_ai_agent_service.mcp.mcp_streamable_http_client import (
    MCPStreamableHTTPClient,
)

TRANSPORT_SSE = 'sse'
TRANSPORT_STREAMABLE_HTTP = 'streamable_http'


class MCPClientFactory:
    """Factory Method com registry para criação de clientes MCP.

    Associa nomes de transporte às classes concretas. Novos transportes
    podem ser registrados em runtime via `MCPClientFactory.register()`,
    sem alterar esta classe.

    Exemplos:
        # Streamable HTTP (padrão FastMCP)
        client = MCPClientFactory.create('srv', 'http://localhost:8000/mcp')

        # SSE
        client = MCPClientFactory.create('srv', 'http://localhost:8000/sse', transport='sse')

        # Com autenticação
        client = MCPClientFactory.create(
            name='srv',
            server_url='http://localhost:8000/mcp',
            transport='streamable_http',
            headers={'Authorization': 'Bearer eyJ...'},
        )

        # Uso como context manager (igual para qualquer transporte)
        async with MCPClientFactory.create('srv', 'http://localhost:8000/mcp') as client:
            tools = await client.list_tools()

        # Registrar um transporte customizado
        MCPClientFactory.register('my_transport', MyCustomClient)
        client = MCPClientFactory.create('srv', 'http://...', transport='my_transport')
    """

    _registry: dict[str, type[MCPBaseClient]] = {
        TRANSPORT_STREAMABLE_HTTP: MCPStreamableHTTPClient,
        TRANSPORT_SSE: MCPSSEClient,
    }

    @classmethod
    def create(
        cls,
        name: str,
        server_url: str,
        transport: str = TRANSPORT_STREAMABLE_HTTP,
        headers: dict[str, str] | None = None,
    ) -> MCPBaseClient:
        """Instancia o cliente MCP correspondente ao transporte solicitado.

        Args:
            name:       Identificador da conexão (usado em logs).
            server_url: URL completa do endpoint MCP.
            transport:  Chave de transporte registrada (padrão: 'streamable_http').
            headers:    Headers HTTP opcionais (ex: Authorization Bearer).

        Returns:
            Instância concreta de MCPBaseClient.

        Raises:
            ValueError: Se o transporte não estiver registrado.
        """
        client_class = cls._registry.get(transport)
        if client_class is None:
            available = ', '.join(f'{k!r}' for k in cls._registry)
            raise ValueError(
                f'Transporte {transport!r} não registrado. '
                f'Disponíveis: {available}.'
            )
        return client_class(name, server_url, headers)

    @classmethod
    def register(
        cls, transport: str, client_class: type[MCPBaseClient]
    ) -> None:
        """Registra um novo transporte na factory.

        Args:
            transport:    Chave de identificação do transporte.
            client_class: Classe concreta que herda de MCPBaseClient.
        """
        if not issubclass(client_class, MCPBaseClient):
            raise TypeError(
                f'{client_class.__name__} deve herdar de MCPBaseClient.'
            )
        cls._registry[transport] = client_class

    @classmethod
    def available_transports(cls) -> list[str]:
        """Retorna os transportes registrados."""
        return list(cls._registry.keys())
