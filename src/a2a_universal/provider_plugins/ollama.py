from __future__ import annotations
from typing import Optional
import os
import httpx

from ..providers import ProviderBase

class Provider(ProviderBase):
    id = "ollama"
    name = "Ollama"
    supports_messages = False  # using simple /generate endpoint

    def __init__(self) -> None:
        self._base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self._model = os.getenv("OLLAMA_MODEL", "llama3")
        # Lazy-ready: assume daemon reachable; errors handled at call time
        self.ready = True
        self.reason = f"Ollama ready (model={self._model})"

    def generate(self, prompt: str = "", messages: Optional[list] = None) -> str:
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
            r = httpx.post(
                f"{self._base}/api/generate",
                json={"model": self._model, "prompt": msg, "stream": False},
                timeout=30.0,
            )
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip() or "Empty response from Ollama."
        except Exception as e:
            return f"[ollama error] {e}"
