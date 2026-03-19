from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """Contrato para stores de vetores de agentes.

    Projetado para permitir substituição futura por outras implementações
    (Qdrant, Weaviate, Chroma, etc.) sem alterar o AgentRegistry.

    IDs são strings que identificam cada agente pelo nome.
    A propriedade `ids` expõe a lista de IDs atualmente no índice.
    """

    @property
    @abstractmethod
    def ids(self) -> list[str]:
        """Retorna os IDs atualmente indexados."""
        ...

    @abstractmethod
    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        """Adiciona vetores ao índice associando-os aos IDs fornecidos."""
        ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 5) -> list[str]:
        """Retorna os IDs dos top_k vizinhos mais próximos do query_vector."""
        ...

    @abstractmethod
    def save(self) -> None:
        """Persiste o índice e os metadados no diretório configurado."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Carrega o índice e os metadados do diretório configurado."""
        ...

    @abstractmethod
    def is_persisted(self) -> bool:
        """Retorna True se o índice já existe no disco."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Limpa o índice em memória (não remove arquivos em disco)."""
        ...
