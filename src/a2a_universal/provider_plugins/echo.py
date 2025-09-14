from __future__ import annotations
from typing import Optional
from ..providers import ProviderBase

class Provider(ProviderBase):
    id = "echo"
    name = "Echo"
    ready = True
    reason = "Echo provider is always ready."
    supports_messages = True

    def generate(self, prompt: str = "", messages: Optional[list] = None) -> str:
        p = (prompt or "").strip()
        return f"Hello, you said: {p}" if p else "Hello, World!"
