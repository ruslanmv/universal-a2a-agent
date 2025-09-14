from __future__ import annotations
from typing import Optional
import os, json
from ..providers import ProviderBase

class Provider(ProviderBase):
    id = "bedrock"
    name = "AWS Bedrock"
    supports_messages = False

    def __init__(self) -> None:
        self._model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
        self._region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        try:
            import boto3  # type: ignore
            self._client = boto3.client("bedrock-runtime", region_name=self._region)  # creds via env/instance
            self.ready = True
            self.reason = f"Bedrock client ready (model={self._model_id})"
        except Exception as e:
            self.ready = False
            self.reason = f"boto3/bedrock runtime not available: {e}"
            self._client = None

    def generate(self, prompt: str = "", messages: Optional[list] = None) -> str:
        if not self.ready or self._client is None:
            return f"[bedrock not ready] {self.reason}"
        # Simple unified prompt -> Claude style request body (works for Anthropic models on Bedrock)
        msg = (prompt or "").strip() or "Say hello."
        body = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": msg}]}],
            "max_tokens": 512,
            "anthropic_version": "bedrock-2023-05-31",
        }
        try:
            res = self._client.invoke_model(modelId=self._model_id, body=json.dumps(body))
            payload = json.loads(res.get("body").read().decode("utf-8"))
            for blk in payload.get("content", []):
                if blk.get("type") == "text":
                    return (blk.get("text") or "").strip()
            return "Empty response from Bedrock."
        except Exception as e:
            return f"[bedrock error] {e}"
