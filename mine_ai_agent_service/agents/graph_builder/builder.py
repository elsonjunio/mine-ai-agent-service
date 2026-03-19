import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from mine_ai_agent_service.agents.base import BaseAgent
from mine_ai_agent_service.agents.planner.agent import Plan


class PipelineState(TypedDict):
    request: str
    context: dict[
        str, str
    ]          # parâmetros que fluem por todos os nós (ex: token)
    results: Annotated[list[str], operator.add]


class GraphBuilder:
    """Monta dinamicamente um StateGraph a partir de um plano do Planner.

    Cada passo do plano vira um nó sequencial no graph.
    O contexto (token, credenciais, etc.) flui pelo estado e é repassado
    a cada agente via invoke(prompt, context).
    """

    def __init__(self, registry: dict[str, BaseAgent]) -> None:
        self._registry = registry

    def build(self, plan: Plan) -> CompiledStateGraph:
        graph: StateGraph = StateGraph(PipelineState)

        previous_node = START
        for i, step in enumerate(plan.steps):
            if step.agent not in self._registry:
                raise ValueError(
                    f"Agente '{step.agent}' não encontrado no registry. "
                    f'Disponíveis: {list(self._registry.keys())}'
                )

            node_name = f'step_{i}__{step.agent}'
            node_fn = self._make_node(self._registry[step.agent], step.task)

            graph.add_node(node_name, node_fn)
            graph.add_edge(previous_node, node_name)
            previous_node = node_name

        graph.add_edge(previous_node, END)
        return graph.compile()

    @staticmethod
    def _make_node(agent: BaseAgent, task: str):
        def node(state: PipelineState) -> dict[str, Any]:
            result = agent.invoke(task, context=state.get('context'))
            return {'results': [result]}

        node.__name__ = agent.agent_name
        return node
