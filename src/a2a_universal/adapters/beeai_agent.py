from __future__ import annotations
from beeai_framework.adapters.a2a.agents.agent import A2AAgent as BeeA2AAgent
from beeai_framework.memory import UnconstrainedMemory

def make_beeai_agent(base_url: str = "http://localhost:8000") -> BeeA2AAgent:
    card_url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
    return BeeA2AAgent(agent_card_url=card_url, memory=UnconstrainedMemory())
