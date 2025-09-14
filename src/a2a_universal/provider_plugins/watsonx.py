from __future__ import annotations
from typing import Optional
import os

from ..providers import ProviderBase

class Provider(ProviderBase):
    id = "watsonx"
    name = "IBM watsonx.ai"
    supports_messages = True

    def __init__(self) -> None:
        api_key = os.getenv("WATSONX_API_KEY")
        url = os.getenv("WATSONX_URL")
        proj = os.getenv("WATSONX_PROJECT_ID")
        self._model_id = os.getenv("MODEL_ID", "ibm/granite-3-3-8b-instruct")
        try:
            from ibm_watsonx_ai import Credentials  # type: ignore
            from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
        except Exception as e:
            self.ready = False
            self.reason = f"ibm-watsonx-ai not installed: {e}"
            self._model = None
            return
        if not (api_key and url and proj):
            self.ready = False
            self.reason = "Missing WATSONX_API_KEY, WATSONX_URL, or WATSONX_PROJECT_ID"
            self._model = None
            return
        try:
            creds = Credentials(url=url, api_key=api_key)
            self._model = ModelInference(model_id=self._model_id, credentials=creds, project_id=proj)
            # sanity ping
            self._model.get_details()
            self.ready = True
            self.reason = f"watsonx.ai ready (model={self._model_id})"
        except Exception as e:
            self.ready = False
            self.reason = f"watsonx.ai init failed: {e}"
            self._model = None

    def generate(self, prompt: str = "", messages: Optional[list] = None) -> str:
        if not self.ready or self._model is None:
            return f"[watsonx not ready] {self.reason}"
        try:
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # type: ignore
            msg = (prompt or "").strip()
            if not msg and messages:
                for m in reversed(messages):
                    if isinstance(m, dict) and m.get("role") == "user":
                        c = m.get("content")
                        if isinstance(c, str):
                            msg = c
                            break
            msg = msg or "Say hello."
            raw = self._model.generate_text(prompt=msg, params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 256,
            }, raw_response=True)
            if raw and raw.get("results"):
                return (raw["results"][0]["generated_text"] or "").strip()
            return "Sorry, empty response from watsonx.ai."
        except Exception as e:
            return f"[watsonx error] {e}"
