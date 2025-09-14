from __future__ import annotations
from langchain.tools import tool
from ..client import A2AClient

@tool("a2a_hello", return_direct=False)
def a2a_hello(text: str, base_url: str = "http://localhost:8000", use_jsonrpc: bool = False) -> str:
    """Send text to the Universal A2A agent and return its reply."""
    return A2AClient(base_url).send(text, use_jsonrpc=use_jsonrpc)
