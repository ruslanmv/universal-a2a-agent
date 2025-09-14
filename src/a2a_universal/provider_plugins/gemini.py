from __future__ import annotations
from typing import Optional
import os
from ..providers import ProviderBase

class Provider(ProviderBase):
    id = "gemini"
    name = "Google Gemini"
    supports_messages = True

    def __init__(self) -> None:
        api = os.getenv("GOOGLE_API_KEY")
        if not api:
            self.ready = False
            self.reason = "GOOGLE_API_KEY not set"
            self._model = None
            return
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api)
            model_id = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            self._model = genai.GenerativeModel(model_id)
            self.ready = True
            self.reason = f"Gemini client ready (model={model_id})"
        except Exception as e:
            self.ready = False
            self.reason = f"google-generativeai not installed/usable: {e}"
            self._model = None

    def generate(self, prompt: str = "", messages: Optional[list] = None) -> str:
        if not self.ready or self._model is None:
            return f"[gemini not ready] {self.reason}"
        msg = (prompt or "").strip() or "Say hello."
        try:
            r = self._model.generate_content(msg)
            return (getattr(r, "text", "") or "").strip() or "Empty response from Gemini."
        except Exception as e:
            return f"[gemini error] {e}"
