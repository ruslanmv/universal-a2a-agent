from __future__ import annotations
from typing import Optional
import os
from ..providers import ProviderBase

class Provider(ProviderBase):
    id = "anthropic"
    name = "Anthropic Claude"
    supports_messages = True

    def __init__(self) -> None:
        api = os.getenv("ANTHROPIC_API_KEY")
        if not api:
            self.ready = False
            self.reason = "ANTHROPIC_API_KEY not set"
            self._client = None
            return
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(api_key=api)
            self.ready = True
            self.reason = "Anthropic client ready"
        except Exception as e:
            self.ready = False
            self.reason = f"anthropic not installed/usable: {e}"
            self._client = None

    def generate(self, prompt: str = "", messages: Optional[list] = None) -> str:
        if not self.ready or self._client is None:
            return f"[anthropic not ready] {self.reason}"
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        msg = (prompt or "").strip() or "Say hello."
        try:
            res = self._client.messages.create(
                model=model, max_tokens=512, messages=[{"role": "user", "content": msg}]
            )
            # Claude returns rich content; pick first text
            for c in res.content:
                if getattr(c, "type", "") == "text":
                    return (getattr(c, "text", "") or "").strip()
            return "Empty response from Claude."
        except Exception as e:
            return f"[anthropic error] {e}"
