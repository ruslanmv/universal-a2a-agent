from __future__ import annotations
from typing import Optional
import os

from ..providers import ProviderBase

class Provider(ProviderBase):
    id = "openai"
    name = "OpenAI"
    supports_messages = True

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self._mode = None
        self._client = None
        if not api_key:
            self.ready = False
            self.reason = "OPENAI_API_KEY not set"
            return
        # Try new SDK (>=1.0)
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=api_key)
            self._mode = "new"
            self.ready = True
            self.reason = "OpenAI new SDK ready"
            return
        except Exception:
            pass
        # Try legacy SDK (<1.0)
        try:
            import openai  # type: ignore
            openai.api_key = api_key
            self._client = openai
            self._mode = "legacy"
            self.ready = True
            self.reason = "OpenAI legacy SDK ready"
        except Exception as e:
            self.ready = False
            self.reason = f"OpenAI SDK not available: {e}"

    def generate(self, prompt: str = "", messages: Optional[list] = None) -> str:
        if not self.ready or self._client is None:
            return f"[openai not ready] {self.reason}"
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        msg = (prompt or "").strip()
        if not msg and messages:
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str):
                        msg = c
                        break
        msg = msg or "Say hello."
        try:
            if self._mode == "new":
                res = self._client.chat.completions.create(  # type: ignore[union-attr]
                    model=model,
                    messages=[{"role": "user", "content": msg}],
                )
                return (res.choices[0].message.content or "").strip()
            else:
                res = self._client.ChatCompletion.create(  # type: ignore[union-attr]
                    model=model,
                    messages=[{"role": "user", "content": msg}],
                )
                return (res["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            return f"[openai error] {e}"
