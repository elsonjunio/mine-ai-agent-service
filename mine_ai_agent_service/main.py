import asyncio
import logging

from mine_ai_agent_service.agents.base import BaseAgent
from mine_ai_agent_service.agents.executor.agent import ExecutorAgent
from mine_ai_agent_service.agents.graph_builder.builder import GraphBuilder
from mine_ai_agent_service.agents.mcp.callbacks import inject_token, resolve_placeholders, store_code_in_context, store_result_in_context
from mine_ai_agent_service.agents.mcp.loader import describe_mcp_agents, load_mcp_agents
from mine_ai_agent_service.agents.planner.agent import PlannerAgent
from mine_ai_agent_service.agents.specialized.output_formatter import OutputFormatterAgent
from mine_ai_agent_service.agents.specialized.python_coder import PythonCoderAgent
from mine_ai_agent_service.config import settings
from mine_ai_agent_service.llm.lmstudio_provider import LMStudioProvider
from mine_ai_agent_service.registry.agent_registry import AgentRegistry

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s — %(message)s')

# ------------------------------------------------------------------
# Configuração MCP — ajuste antes de rodar
# ------------------------------------------------------------------
MCP_URL   = 'http://localhost:8000/mcp'
TRANSPORT = 'streamable_http'
TOKEN     = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InVzZXIxQHZhbG9yZXMuY29tIiwiZm5hbWUiOiJVc3VhcmlvIFZhbG9yZXMiLCJzdWIiOiI4NmUzNzFhZC1kMTIwLTRhZGUtODlmMC03MTFkYzMxYWJhYTYiLCJ1c2VybmFtZSI6InVzZXIxIiwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtaW5mcmEiXSwic3RzIjp7ImFjY2Vzc19rZXkiOiJTWkpCQ0s3UlRGUUZKMzkxUDNNMiIsInNlY3JldF9rZXkiOiI1UUxBM2IxRGZTYnZVbm5IVlcwQ3NydnVhcDBBa3B3K1lMSis0WjFuIiwic2Vzc2lvbl90b2tlbiI6ImV5SmhiR2NpT2lKSVV6VXhNaUlzSW5SNWNDSTZJa3BYVkNKOS5leUpoWTJObGMzTkxaWGtpT2lKVFdrcENRMHMzVWxSR1VVWktNemt4VUROTk1pSXNJbUZqY2lJNklqRWlMQ0poYkd4dmQyVmtMVzl5YVdkcGJuTWlPbHNpYUhSMGNEb3ZMMnh2WTJGc2FHOXpkRG80TURnd0lsMHNJbUYxWkNJNld5SnRhVzVwYnlJc0ltRmpZMjkxYm5RaVhTd2lZWHB3SWpvaWJXbHVhVzhpTENKbGJXRnBiQ0k2SW5WelpYSXhRSFpoYkc5eVpYTXVZMjl0SWl3aVpXMWhhV3hmZG1WeWFXWnBaV1FpT25SeWRXVXNJbVY0Y0NJNk1UYzNNemsyTmpneU1pd2labUZ0YVd4NVgyNWhiV1VpT2lKV1lXeHZjbVZ6SWl3aVoybDJaVzVmYm1GdFpTSTZJbFZ6ZFdGeWFXOGlMQ0pwWVhRaU9qRTNOek01TmpNeE5ERXNJbWx6Y3lJNkltaDBkSEE2THk4eE1qY3VNQzR3TGpFNk9EQTRNUzl5WldGc2JYTXZhVzVtY21FaUxDSnFkR2tpT2lJNU56QmpaVEptTkMxaU5tWTFMVFEzTWprdFlqVXdZUzAxTjJZME9HVTBPRGRtWmpnaUxDSnVZVzFsSWpvaVZYTjFZWEpwYnlCV1lXeHZjbVZ6SWl3aWNHOXNhV041SWpvaVkyOXVjMjlzWlVGa2JXbHVJaXdpY0hKbFptVnljbVZrWDNWelpYSnVZVzFsSWpvaWRYTmxjakVpTENKeVpXRnNiVjloWTJObGMzTWlPbnNpY205c1pYTWlPbHNpYjJabWJHbHVaVjloWTJObGMzTWlMQ0oxYldGZllYVjBhRzl5YVhwaGRHbHZiaUlzSW1SbFptRjFiSFF0Y205c1pYTXRhVzVtY21FaVhYMHNJbkpsYzI5MWNtTmxYMkZqWTJWemN5STZleUpoWTJOdmRXNTBJanA3SW5KdmJHVnpJanBiSW0xaGJtRm5aUzFoWTJOdmRXNTBJaXdpYldGdVlXZGxMV0ZqWTI5MWJuUXRiR2x1YTNNaUxDSjJhV1YzTFhCeWIyWnBiR1VpWFgwc0ltMXBibWx2SWpwN0luSnZiR1Z6SWpwYkltTnZibk52YkdWQlpHMXBiaUpkZlgwc0luTmpiM0JsSWpvaVpXMWhhV3dnY0hKdlptbHNaU0lzSW5ObGMzTnBiMjVmYzNSaGRHVWlPaUpoWkRJNU1EVmhOUzA0T1RSaExUUmhNVFV0T0daaE9TMHhZbUV3WVRobU5qaGxNR1FpTENKemFXUWlPaUpoWkRJNU1EVmhOUzA0T1RSaExUUmhNVFV0T0daaE9TMHhZbUV3WVRobU5qaGxNR1FpTENKemRXSWlPaUk0Tm1Vek56RmhaQzFrTVRJd0xUUmhaR1V0T0RsbU1DMDNNVEZrWXpNeFlXSmhZVFlpTENKMGVYQWlPaUpDWldGeVpYSWlmUS4zMUJKaUNHckJtNHRDbWtlRE9uUFJGcWREbFBaZGJTTlJiUVZlYjZVMUlTREZLeE5KMm51cUNpM2hCa2dVN3gydFNkZjNFXzdGZGxPMDdTWFY0TmVhQSIsImV4cGlyYXRpb24iOiIyMDI2LTAzLTIwVDAwOjMzOjQyWiJ9LCJ0eXBlIjoibWluZV9zZXNzaW9uIiwiZXhwIjoxNzczOTY1MDIyLCJwb2xpY3kiOlsiY29uc29sZUFkbWluIl19.gqWfmlCWk-MJh3xXp_MzstRufCl5bAnkVbR7UvXoQIY'


async def build_registry(
    llm,
) -> tuple[dict[str, BaseAgent], dict[str, str], AgentRegistry]:
    """Monta o registry completo: agentes estáticos + agentes MCP dinâmicos,
    gera resumos via LLM e indexa no FAISS para busca semântica.

    Returns:
        registry:       {agent_name: agent_instance}
        descriptions:   {agent_name: description}  → alimenta o PlannerAgent
        agent_registry: AgentRegistry pronto para search_agents()
    """
    # ------------------------------------------------------------------
    # 1. Agentes estáticos
    # ------------------------------------------------------------------
    static_agents: dict[str, BaseAgent] = {
        'python_coder': PythonCoderAgent(llm=llm, callbacks=[store_code_in_context]),
    }
    static_descriptions: dict[str, str] = {
        'python_coder': 'Gera código Python: funções, classes, scripts e módulos.',
    }

    # ------------------------------------------------------------------
    # 2. Agentes MCP dinâmicos — descobertos em runtime
    # ------------------------------------------------------------------
    mcp_agents = await load_mcp_agents(
        server_urls=[MCP_URL],
        llm=llm,
        transport=TRANSPORT,
        server_headers={'Authorization': f'Bearer {TOKEN}'},
        callbacks=[inject_token, store_result_in_context],
    )
    mcp_descriptions = describe_mcp_agents(mcp_agents)

    registry     = {**static_agents, **mcp_agents}
    descriptions = {**static_descriptions, **mcp_descriptions}

    # ------------------------------------------------------------------
    # 3. Indexação: gera resumos para agentes novos e re-indexa se necessário
    # ------------------------------------------------------------------
    agent_registry = AgentRegistry(data_dir=settings.DATA_DIR, llm=llm)

    for agent in registry.values():
        agent_registry.generate_summary(agent)

    agent_registry.index_all()

    return registry, descriptions, agent_registry


if __name__ == '__main__':
    llm = LMStudioProvider().get_llm()

    # ------------------------------------------------------------------
    # 1. Registry: estáticos + MCP + indexação semântica
    # ------------------------------------------------------------------
    registry, descriptions, agent_registry = asyncio.run(build_registry(llm))

    print('\nAgentes disponíveis:')
    for name, desc in descriptions.items():
        print(f'  [{name}] {desc}')

    # ------------------------------------------------------------------
    # 2. Requisição e contexto
    # ------------------------------------------------------------------
    request = 'crie uma url para eu fazer upload de um arquivo no bucket papinha. Depois list os buckets disponíveis para meu usuário. E crie um código python que me da o n-nesimo número da sequência de fibonatti'
    context = {'token': TOKEN}

    # ------------------------------------------------------------------
    # 3. Planner: decompõe a requisição em passos usando busca semântica
    # ------------------------------------------------------------------
    planner = PlannerAgent(llm=llm, agent_descriptions=descriptions)
    plan = planner.plan(request, agent_registry)

    print(f'\nPlano ({len(plan.steps)} passo(s)): {plan.reasoning}')
    for i, step in enumerate(plan.steps):
        print(f'  [{i + 1}] {step.agent} → {step.task}')

    # ------------------------------------------------------------------
    # 4. GraphBuilder: monta o graph a partir do plano
    # ------------------------------------------------------------------
    graph = GraphBuilder(registry=registry).build(plan)

    # ------------------------------------------------------------------
    # 5. Executor: roda o graph
    # ------------------------------------------------------------------
    result = ExecutorAgent().run(
        graph=graph,
        request=request,
        plan=plan,
        context=context,
    )

    # ------------------------------------------------------------------
    # 6. OutputFormatter: traduz e formata a saída na língua da pergunta
    # ------------------------------------------------------------------
    formatted = OutputFormatterAgent.format(
        llm=llm,
        request=request,
        steps=[(s.task, s.output) for s in result.steps],
    )

    final = resolve_placeholders(formatted, context)

    print(f'\nResultado final:\n{final}')
