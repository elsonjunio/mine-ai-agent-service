# Re-export para compatibilidade — a definição canônica está em agents/events.py
from mine_ai_agent_service.agents.events import AgentCallback, AgentEvent

MCPEvent    = AgentEvent
MCPCallback = AgentCallback

__all__ = ['MCPEvent', 'MCPCallback']
