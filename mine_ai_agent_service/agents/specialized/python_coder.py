import asyncio
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from mine_ai_agent_service.agents.base import BaseAgent
from mine_ai_agent_service.agents.events import AgentCallback, AgentEvent


SYSTEM_PROMPT = """Você é um especialista em Python. Sua única responsabilidade é gerar código Python correto, limpo e idiomático.

Regras:
- Responda APENAS com o bloco de código Python, sem explicações fora do código.
- Use docstrings e comentários somente quando necessário para clareza.
- Prefira soluções simples e diretas.
- Siga PEP 8.
- Se precisar de dependências externas, liste-as em um comentário no topo do arquivo."""


class CoderState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class PythonCoderAgent(BaseAgent):
    """Agente especializado em geração de código Python."""

    def __init__(
        self,
        llm: BaseChatModel,
        callbacks: list[AgentCallback] | None = None,
    ) -> None:
        super().__init__(llm, callbacks)
        self._graph = self.compile()

    @property
    def agent_name(self) -> str:
        return 'python_coder'

    def describe(self) -> str:
        return 'Gera código Python: funções, classes, scripts e módulos.'

    def build_graph(self) -> StateGraph:
        graph = StateGraph(CoderState)
        graph.add_node('generate', self._generate)
        graph.add_edge(START, 'generate')
        graph.add_edge('generate', END)
        return graph

    def _generate(self, state: CoderState) -> CoderState:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        response: AIMessage = self.llm.invoke(messages)
        return {'messages': [response]}

    def invoke(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        return asyncio.run(self._async_invoke(prompt, context or {}))

    async def _async_invoke(self, prompt: str, context: dict[str, Any]) -> str:
        state: dict[str, Any] = {
            'agent_name': self.agent_name,
            'prompt': prompt,
            'messages': [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)],
            'final_output': '',
        }

        # PRE_LLM — permite enriquecer/reescrever o prompt antes da LLM
        state, context = await self._dispatch(AgentEvent.PRE_LLM, state, context)

        response: AIMessage = await self.llm.ainvoke(state['messages'])
        state['final_output'] = response.content

        # POST_LLM — permite transformar/inspecionar a saída da LLM
        state, context = await self._dispatch(AgentEvent.POST_LLM, state, context)

        return state['final_output']
