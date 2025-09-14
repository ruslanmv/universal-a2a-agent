from __future__ import annotations
import httpx
from typing import Dict, Any

class A2AClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def send(self, text: str, use_jsonrpc: bool = False, timeout: float = 20.0) -> str:
        if use_jsonrpc:
            payload: Dict[str, Any] = {
                "jsonrpc": "2.0", "id": "1", "method": "message/send",
                "params": {"message": {"role": "user", "messageId": "cli", "parts": [{"type": "text", "text": text}]}}
            }
            url = f"{self.base_url}/rpc"
        else:
            payload = {"method": "message/send", "params": {"message": {"role": "user", "messageId": "cli", "parts": [{"type": "text", "text": text}]}}}
            url = f"{self.base_url}/a2a"
        r = httpx.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if "result" in data and isinstance(data["result"], dict):
            data = data["result"]
        parts = (data.get("message") or {}).get("parts", [])
        for p in parts:
            if p.get("type") == "text":
                return p.get("text", "")
        return ""
