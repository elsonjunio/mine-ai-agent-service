from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

if TYPE_CHECKING:
    from mine_ai_agent_service.registry.agent_registry import AgentRegistry


_SYSTEM_PROMPT_TEMPLATE = """\
Você é um agente planejador. Sua função é analisar uma requisição e decompô-la \
em subtarefas sequenciais, atribuindo cada uma ao agente especializado correto.

Agentes disponíveis:
{agent_list}

Regras:
- Use apenas agentes da lista acima. O campo `agent` deve ser exatamente o nome listado.
- Decomponha a requisição no menor número de passos necessários.
- Cada passo deve ser autocontido e ter uma instrução clara para o agente.
- O campo `reasoning` deve explicar brevemente a estratégia escolhida.\
"""


class PlanStep(BaseModel):
    agent: str
    task: str


class Plan(BaseModel):
    reasoning: str
    steps: list[PlanStep]


class PlannerAgent:
    """Analisa a requisição e produz um plano de execução estruturado.

    Args:
        llm: modelo de linguagem a usar.
        agent_descriptions: mapeamento {agent_name: descrição} dos agentes disponíveis.
    """

    def __init__(
        self, llm: BaseChatModel, agent_descriptions: dict[str, str]
    ) -> None:
        self._llm = llm.with_structured_output(Plan)
        self._agent_descriptions = agent_descriptions

    def plan(
        self, request: str, registry: AgentRegistry | None = None
    ) -> Plan:
        """Produz um plano de execução para a requisição.

        Args:
            request: instrução do usuário.
            registry: se fornecido, usa busca semântica para pré-filtrar os agentes
                      relevantes antes de enviar ao LLM, reduzindo o contexto.
        """
        if registry is not None:
            relevant_names = set(registry.search_agents(request))
            descriptions = {
                name: desc
                for name, desc in self._agent_descriptions.items()
                if name in relevant_names
            }
            if not descriptions:
                descriptions = self._agent_descriptions
        else:
            descriptions = self._agent_descriptions

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            agent_list='\n'.join(
                f'- {name}: {desc}' for name, desc in descriptions.items()
            )
        )

        return self._llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=request),
            ]
        )
