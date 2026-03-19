import json
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from mine_ai_agent_service.agents.base import BaseAgent
from mine_ai_agent_service.registry.embedder import Embedder
from mine_ai_agent_service.registry.store.base import BaseVectorStore
from mine_ai_agent_service.registry.store.faiss_store import FaissVectorStore

_SUMMARIES_DIR = 'summaries'


class AgentSummary(BaseModel):
    name: str
    summary: str
    tags: list[str]


class AgentRegistry:
    """Gerencia resumos, indexação vetorial e busca semântica de agentes.

    Fluxo típico:
      1. `generate_summary(agent)` — para cada agente, gera e persiste um resumo via LLM.
      2. `index_all()`             — lê todos os resumos do disco e indexa no FAISS.
      3. `search_agents(query)`    — otimiza a query via LLM e retorna nomes de agentes relevantes.

    O store pode ser substituído por outra implementação de BaseVectorStore
    (ex: QdrantVectorStore) passando-o via `store=`.
    """

    def __init__(
        self,
        data_dir: str,
        llm: BaseChatModel,
        store: BaseVectorStore | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._llm = llm
        self._embedder = Embedder()
        self._store = store or FaissVectorStore(data_dir=data_dir)
        self._summaries_dir = os.path.join(data_dir, _SUMMARIES_DIR)

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    def generate_summary(self, agent: BaseAgent) -> AgentSummary:
        """Retorna o resumo do agente.

        Se já existir em disco, carrega. Caso contrário, solicita à LLM
        um resumo estruturado (name, summary, tags) em inglês e persiste.
        """
        path = os.path.join(self._summaries_dir, f'{agent.agent_name}.json')

        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return AgentSummary(**json.load(f))

        description = getattr(agent, 'describe', lambda: agent.agent_name)()

        structured_llm = self._llm.with_structured_output(AgentSummary)
        summary: AgentSummary = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        'You are an assistant that generates structured summaries for AI agents. '
                        'Always respond in English with concise, search-optimized text.'
                    )
                ),
                HumanMessage(
                    content=(
                        f'Generate a structured summary for the following AI agent.\n\n'
                        f'Agent name: {agent.agent_name}\n'
                        f'Agent description: {description}\n\n'
                        'Return:\n'
                        '- name: snake_case identifier for the agent\n'
                        '- summary: 1-2 sentences describing what this agent does\n'
                        '- tags: list of relevant English keywords for semantic search'
                    )
                ),
            ]
        )

        os.makedirs(self._summaries_dir, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary.model_dump(), f, ensure_ascii=False, indent=2)

        return summary

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_all(self) -> None:
        """Gera embeddings de todos os resumos em disco e indexa no store.

        Cruza os nomes dos arquivos em `summaries/` com os IDs persistidos no store.
        Se forem idênticos, o store é carregado (pronto para buscas) e a re-indexação
        é pulada. Caso contrário, o store é limpo e reconstruído do zero.

        O texto indexado combina summary + tags para enriquecer a busca semântica.
        """
        if not os.path.exists(self._summaries_dir):
            print('[AgentRegistry] Nenhum resumo encontrado para indexar.')
            return

        summary_names = sorted(
            fname[:-5]  # remove .json
            for fname in os.listdir(self._summaries_dir)
            if fname.endswith('.json')
        )

        if not summary_names:
            return

        if self._store.is_persisted():
            self._store.load()
            if sorted(self._store.ids) == summary_names:
                print(
                    '[AgentRegistry] Índice já atualizado, nada a re-indexar.'
                )
                return
            # IDs divergem: descarta o que foi carregado antes de re-indexar
            self._store.clear()

        summaries: list[AgentSummary] = []
        for name in summary_names:
            with open(
                os.path.join(self._summaries_dir, f'{name}.json'),
                encoding='utf-8',
            ) as f:
                summaries.append(AgentSummary(**json.load(f)))

        texts = [f'{s.summary} {" ".join(s.tags)}' for s in summaries]
        ids = [s.name for s in summaries]

        print(f'[AgentRegistry] Indexando {len(summaries)} agentes...')
        vectors = self._embedder.embed(texts)
        self._store.add(ids=ids, vectors=vectors)
        self._store.save()
        print('[AgentRegistry] Índice salvo.')

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_agents(self, query: str, top_k: int = 5) -> list[str]:
        """Retorna nomes dos agentes mais relevantes para a query.

        1. Solicita à LLM uma versão em inglês otimizada para busca semântica.
        2. Gera o embedding da query otimizada.
        3. Busca no store e retorna a lista de nomes de agentes.
        """
        response = self._llm.invoke(
            [
                SystemMessage(
                    content=(
                        'You are a search query optimizer.'
                        'Rewrite the user request into a concise English search query for tool selection.'
                        'Rules:'
                        '- Preserve ALL user intents and actions (verbs like create, list, update, delete).'
                        '- Keep technical meaning (e.g., python, fibonacci, buckets).'
                        '- Prefer imperative form (e.g., "create python code", "list buckets").'
                        '- Keep multiple actions if present.'
                        '- Do NOT drop important steps.'
                        '- No explanations, no punctuation — just the query text.'
                        'Return only the query.'
                    )
                ),
                HumanMessage(content=query),
            ]
        )
        optimized_query = response.content.strip()
        print(f'[AgentRegistry] Query otimizada: "{optimized_query}"')

        query_vector = self._embedder.embed([optimized_query])[0]
        return self._store.search(query_vector=query_vector, top_k=top_k)
