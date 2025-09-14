from __future__ import annotations
from typing import Dict, Any
from ..client import A2AClient

class A2ANode:
    def __init__(self, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False):
        self.client = A2AClient(base_url)
        self.use_jsonrpc = use_jsonrpc

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("input", "")
        reply = self.client.send(text, use_jsonrpc=self.use_jsonrpc)
        return {**state, "a2a_reply": reply}
