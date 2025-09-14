from __future__ import annotations
from typing import Optional
import os
from ..providers import ProviderBase

class Provider(ProviderBase):
    id = "azure_openai"
    name = "Azure OpenAI"
    supports_messages = True

    def __init__(self) -> None:
        key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        self._deployment = deployment
        if not (key and endpoint and deployment):
            self.ready = False
            self.reason = "Missing AZURE_OPENAI_API_KEY/ENDPOINT/DEPLOYMENT"
            self._client = None
            return
        try:
            from azure.ai.openai import OpenAIClient  # type: ignore
            from azure.core.credentials import AzureKeyCredential  # type: ignore
            self._client = OpenAIClient(endpoint=endpoint, credential=AzureKeyCredential(key), api_version=api_version)
            self.ready = True
            self.reason = "Azure OpenAI client ready"
        except Exception as e:
            self.ready = False
            self.reason = f"azure-ai-openai not installed/usable: {e}"
            self._client = None

    def generate(self, prompt: str = "", messages: Optional[list] = None) -> str:
        if not self.ready or self._client is None or not self._deployment:
            return f"[azure openai not ready] {self.reason}"
        msg = (prompt or "").strip() or "Say hello."
        try:
            res = self._client.get_chat_completions(
                deployment_id=self._deployment,
                messages=[{"role": "user", "content": msg}],
                temperature=0.2,
            )
            return (res.choices[0].message.content or "").strip()
        except Exception as e:
            return f"[azure openai error] {e}"
