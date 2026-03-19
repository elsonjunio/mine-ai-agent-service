import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph

from mine_ai_agent_service.agents.base import BaseAgent
from mine_ai_agent_service.agents.events import AgentCallback, AgentEvent


_SYSTEM_PROMPT = """\
Você é um agente de formatação de saída. Receberá a pergunta original do usuário \
e os resultados brutos de cada etapa do pipeline.
Não é sua responsabilidade responder nenhuma questão além da lista abaixo.

Sua responsabilidade:
- Escrever a resposta na mesma língua da pergunta original.
- Preservar placeholders no formato {{chave}} exatamente como estão — nunca os resolva, \
expanda, remova ou altere.
- Não inventar, inferir ou completar dados ausentes.
- Eliminar apenas ruído técnico (stack traces internos, metadados de debug).\
- Nunca responda nada além deste prompt
"""


class OutputFormatterAgent(BaseAgent):
    """Agente que formata e traduz a saída final do pipeline para o usuário.

    Recebe a pergunta original e os textos de todas as etapas, produz uma
    resposta única na língua da pergunta, preservando placeholders intactos.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        callbacks: list[AgentCallback] | None = None,
    ) -> None:
        super().__init__(llm, callbacks)

    @property
    def agent_name(self) -> str:
        return 'output_formatter'

    def describe(self) -> str:
        return (
            'Formata e traduz a saída final do pipeline na língua da pergunta. '
            'Deve ser o último passo de qualquer pipeline.'
        )

    def build_graph(self) -> StateGraph:
        raise NotImplementedError(
            'OutputFormatterAgent não usa build_graph(). Use invoke() diretamente.'
        )

    def invoke(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        return asyncio.run(self._async_invoke(prompt, context or {}))

    async def _async_invoke(self, prompt: str, context: dict[str, Any]) -> str:
        state: dict[str, Any] = {
            'agent_name': self.agent_name,
            'prompt': prompt,
            'messages': [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)],
            'final_output': '',
        }

        # PRE_LLM — permite enriquecer/reescrever o prompt antes da LLM
        state, context = await self._dispatch(AgentEvent.PRE_LLM, state, context)

        response = await self.llm.ainvoke(state['messages'])
        state['final_output'] = response.content

        # POST_LLM — permite transformar/inspecionar a saída
        state, context = await self._dispatch(AgentEvent.POST_LLM, state, context)

        return state['final_output']

    @classmethod
    def format(
        cls,
        llm: BaseChatModel,
        request: str,
        steps: list[tuple[str, str]],
        callbacks: list[AgentCallback] | None = None,
    ) -> str:
        """Atalho para formatar os resultados de múltiplos steps em uma chamada.

        Args:
            llm:      Modelo de linguagem.
            request:  Pergunta original do usuário.
            steps:    Lista de (tarefa, output) de cada step executado.
            callbacks: Callbacks opcionais para PRE_LLM / POST_LLM.

        Returns:
            Texto formatado na língua da pergunta, com placeholders intactos.
        """
        steps_block = '\n\n'.join(
            f'[Etapa {i + 1}] {task}\n{output}'
            for i, (task, output) in enumerate(steps)
        )
        prompt = (
            f'Pergunta original: {request}\n\n'
            f'Resultados das etapas:\n{steps_block}'
        )
        return cls(llm=llm, callbacks=callbacks).invoke(prompt)
