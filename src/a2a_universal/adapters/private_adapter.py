"""Private adapter mapping — toggled by env and intentionally generic."""
from __future__ import annotations
import os
from typing import Any, Dict

INPUT_KEY = os.getenv("PRIVATE_ADAPTER_INPUT_KEY", "input")
OUTPUT_KEY = os.getenv("PRIVATE_ADAPTER_OUTPUT_KEY", "output")
TRACE_KEY = os.getenv("PRIVATE_ADAPTER_TRACE_KEY", "traceId")


def extract_user_text(body: Dict[str, Any]) -> str:
    if isinstance(body, dict) and INPUT_KEY in body and isinstance(body[INPUT_KEY], str):
        return body[INPUT_KEY]
    messages = body.get("messages") if isinstance(body, dict) else None
    if isinstance(messages, list):
        for m in reversed(messages):
            role = (m or {}).get("role")
            content = (m or {}).get("content")
            if role == "user":
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            return p.get("text", "")
                return ""
    return ""


def make_response(text: str, req_body: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {OUTPUT_KEY: text, "ok": True}
    if isinstance(req_body, dict) and TRACE_KEY in req_body:
        out[TRACE_KEY] = req_body.get(TRACE_KEY)
    return out

