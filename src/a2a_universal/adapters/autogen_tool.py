from __future__ import annotations
try:
    from autogen import register_function
except Exception:
    def register_function(*args, **kwargs):
        def deco(f):
            return f
        return deco

from ..client import A2AClient

@register_function("a2a_hello", description="Send text to A2A agent and return reply")
def a2a_hello(text: str, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False) -> str:
    return A2AClient(base_url).send(text, use_jsonrpc=use_jsonrpc)
