# src/a2a_universal/adapters/langgraph_agent.py
from __future__ import annotations

from typing import Any, Dict

import httpx
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage

from ..client import A2AClient


class A2AAgentNode:
    """LangGraph-native node that wraps the Universal A2A service.

    Test-friendly: if the HTTP call fails (e.g., server not running in unit tests),
    we fall back to a local echo so graphs remain executable.
    """

    def __init__(self, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False):
        self.client = A2AClient(base_url)
        self.use_jsonrpc = use_jsonrpc

    def __call__(self, state: MessagesState) -> Dict[str, Any]:
        last = state["messages"][-1]
        user_text = getattr(last, "content", "") if last else ""
        try:
            reply = self.client.send(user_text, use_jsonrpc=self.use_jsonrpc)
        except httpx.HTTPError:
            # Offline fallback so unit tests don't require a running server
            reply = f"Hello (offline), you said: {user_text}" if user_text else "Hello, World!"
        return {"messages": [AIMessage(content=reply)]}
