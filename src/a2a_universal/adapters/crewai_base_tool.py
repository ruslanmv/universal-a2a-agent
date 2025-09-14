from __future__ import annotations
from typing import Any

try:
    from crewai.tools import BaseTool
except Exception:
    class BaseTool:  # shim
        name: str = "a2a_hello"
        description: str = "A2A Hello tool"
        base_url: str = "http://localhost:8000"
        use_jsonrpc: bool = False
        def run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            return self._run(*args, **kwargs)

from ..client import A2AClient

class A2AHelloTool(BaseTool):
    name: str = "a2a_hello"
    description: str = "Send text to the Universal A2A agent and return the reply."
    base_url: str = "http://localhost:8000"
    use_jsonrpc: bool = False

    def _run(self, text: str) -> str:
        return A2AClient(self.base_url).send(text, use_jsonrpc=self.use_jsonrpc)
