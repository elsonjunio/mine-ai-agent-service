import asyncio
import logging

from mine_ai_agent_service.agents.executor.agent import ExecutorAgent
from mine_ai_agent_service.agents.graph_builder.builder import GraphBuilder
from mine_ai_agent_service.agents.mcp.callbacks import (
    inject_token,
    resolve_placeholders,
    store_code_in_context,
    store_result_in_context,
)
from mine_ai_agent_service.agents.mcp.loader import describe_mcp_agents, load_mcp_agents
from mine_ai_agent_service.agents.planner.agent import PlannerAgent
from mine_ai_agent_service.agents.specialized.output_formatter import OutputFormatterAgent
from mine_ai_agent_service.agents.specialized.python_coder import PythonCoderAgent
from mine_ai_agent_service.config import settings
from mine_ai_agent_service.llm.factory import get_llm
from mine_ai_agent_service.registry.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


async def run_agent(request: str, token: str) -> str:
    """Executa o pipeline completo do agente para uma requisição.

    Args:
        request: Mensagem/instrução do usuário.
        token:   JWT raw do usuário, usado para autenticar chamadas MCP.

    Returns:
        Resposta final formatada na língua da requisição.
    """
    llm = get_llm()

    # --- Agentes MCP: descoberta dinâmica (async) ---
    mcp_agents = await load_mcp_agents(
        server_urls=settings.MCP_URLS,
        llm=llm,
        transport='streamable_http',
        server_headers={'Authorization': f'Bearer {token}'},
        callbacks=[inject_token, store_result_in_context],
    )
    mcp_descriptions = describe_mcp_agents(mcp_agents)

    static_agents = {
        'python_coder': PythonCoderAgent(llm=llm, callbacks=[store_code_in_context]),
    }
    static_descriptions = {
        'python_coder': 'Gera código Python: funções, classes, scripts e módulos.',
    }

    registry = {**static_agents, **mcp_agents}
    descriptions = {**static_descriptions, **mcp_descriptions}

    # context acumula resultados intermediários via callbacks (store_result_in_context)
    context: dict = {'token': token}

    # --- Pipeline síncrono: executado em thread para não bloquear o event loop ---
    # OutputFormatterAgent.format() usa asyncio.run() internamente; ao rodar em
    # thread separada, isso cria um novo event loop no thread sem conflito.
    def _pipeline() -> str:
        agent_registry = AgentRegistry(data_dir=settings.DATA_DIR, llm=llm)
        for agent in registry.values():
            agent_registry.generate_summary(agent)
        agent_registry.index_all()

        planner = PlannerAgent(llm=llm, agent_descriptions=descriptions)
        plan = planner.plan(request, agent_registry)

        logger.info('Plano (%d passo(s)): %s', len(plan.steps), plan.reasoning)
        for i, step in enumerate(plan.steps):
            logger.info('  [%d] %s → %s', i + 1, step.agent, step.task)

        graph = GraphBuilder(registry=registry).build(plan)
        result = ExecutorAgent().run(
            graph=graph,
            request=request,
            plan=plan,
            context=context,
        )

        formatted = OutputFormatterAgent.format(
            llm=llm,
            request=request,
            steps=[(s.task, s.output) for s in result.steps],
        )

        return resolve_placeholders(formatted, context)

    return await asyncio.to_thread(_pipeline)
