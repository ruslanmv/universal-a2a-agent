from __future__ import annotations
try:
    from crewai_tools import tool
except Exception:
    def tool(fn=None, **kwargs):
        def wrap(f):
            return f
        return wrap if fn is None else wrap(fn)

from ..client import A2AClient

@tool("a2a_hello")
def a2a_hello(text: str, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False) -> str:
    return A2AClient(base_url).send(text, use_jsonrpc=use_jsonrpc)
