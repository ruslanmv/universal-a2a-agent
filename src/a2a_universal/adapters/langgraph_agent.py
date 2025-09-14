from __future__ import annotations
from typing import Any
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage
from ..client import A2AClient

class A2AAgentNode:
    """LangGraph-native node that wraps the Universal A2A service."""
    def __init__(self, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False):
        self.client = A2AClient(base_url)
        self.use_jsonrpc = use_jsonrpc

    def __call__(self, state: MessagesState) -> dict[str, Any]:
        last = state["messages"][-1]
        user_text = getattr(last, "content", "") if last else ""
        reply = self.client.send(user_text, use_jsonrpc=self.use_jsonrpc)
        return {"messages": [AIMessage(content=reply)]}
