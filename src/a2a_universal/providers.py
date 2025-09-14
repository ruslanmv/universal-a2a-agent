from __future__ import annotations
import os
from typing import Optional

class ProviderBase:
    name: str = "base"
    ready: bool = False
    reason: str = ""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

# ---- Echo (no external deps) ----
class EchoProvider(ProviderBase):
    name = "echo"
    ready = True
    reason = "Echo provider is always ready."

    def generate(self, prompt: str) -> str:
        p = (prompt or "").strip()
        return f"Hello, you said: {p}" if p else "Hello, World!"

# ---- OpenAI (optional) ----
class OpenAIProvider(ProviderBase):
    name = "openai"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.ready = False
            self.reason = "OPENAI_API_KEY not set"
            self._client = None
            return
        # Try new SDK first
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=api_key)
            self._mode = "new"
            self.ready = True
            self.reason = "OpenAI client ready (new SDK)."
        except Exception:
            # Fallback to legacy openai
            try:
                import openai  # type: ignore
                openai.api_key = api_key
                self._client = openai
                self._mode = "legacy"
                self.ready = True
                self.reason = "OpenAI client ready (legacy SDK)."
            except Exception as e:
                self.ready = False
                self.reason = f"OpenAI SDK not available: {e}"
                self._client = None

    def generate(self, prompt: str) -> str:
        if not self.ready or self._client is None:
            return f"[openai not ready] {self.reason}"
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        msg = (prompt or "").strip() or "Say hello."
        try:
            if self._mode == "new":
                # openai>=1.0 style
                res = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": msg}],
                )
                return (res.choices[0].message.content or "").strip()
            else:
                # legacy style
                res = self._client.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": msg}],
                )
                return (res["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            return f"[openai error] {e}"

# ---- IBM watsonx.ai (optional) ----
class WatsonxProvider(ProviderBase):
    name = "watsonx"

    def __init__(self) -> None:
        api_key = os.getenv("WATSONX_API_KEY")
        url = os.getenv("WATSONX_URL")
        proj = os.getenv("WATSONX_PROJECT_ID")
        self._model_id = os.getenv("MODEL_ID", "ibm/granite-3-3-8b-instruct")
        try:
            from ibm_watsonx_ai import Credentials  # type: ignore
            from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
            from ibm_watsonx_ai.wml_client_error import WMLClientError  # type: ignore
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
            # force-check
            self._model.get_details()
            self.ready = True
            self.reason = f"watsonx.ai ready (model={self._model_id})."
        except Exception as e:
            self.ready = False
            self.reason = f"watsonx.ai init failed: {e}"
            self._model = None

    def generate(self, prompt: str) -> str:
        if not self.ready or self._model is None:
            return f"[watsonx not ready] {self.reason}"
        try:
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # type: ignore
            params = {GenParams.DECODING_METHOD: "greedy", GenParams.MAX_NEW_TOKENS: 256}
            raw = self._model.generate_text(prompt=prompt or "Say hello.", params=params, raw_response=True)
            if raw and raw.get("results"):
                return (raw["results"][0]["generated_text"] or "").strip()
            return "Sorry, empty response from watsonx.ai."
        except Exception as e:
            return f"[watsonx error] {e}"

# ---- Factory ----
def build_provider() -> ProviderBase:
    which = (os.getenv("LLM_PROVIDER", "echo") or "echo").lower()
    if which == "openai":
        return OpenAIProvider()
    if which == "watsonx":
        return WatsonxProvider()
    return EchoProvider()
