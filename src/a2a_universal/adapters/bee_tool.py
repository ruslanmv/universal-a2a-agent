from __future__ import annotations
from ..client import A2AClient

def a2a_call(text: str, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False) -> str:
    return A2AClient(base_url).send(text, use_jsonrpc=use_jsonrpc)
