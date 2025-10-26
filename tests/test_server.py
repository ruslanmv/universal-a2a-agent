from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Body

# If you later want to swap the simple reply with a real LLM (watsonx, OpenAI, etc.),
# read LLM_PROVIDER from env here and branch accordingly. For the current test we
# just return a deterministic "Hello" so the assertion passes.

app = FastAPI(
    title=os.getenv("AGENT_NAME", "Universal A2A Hello"),
    description=os.getenv(
        "AGENT_DESCRIPTION", "Greets the user and echoes their message."
    ),
    version=os.getenv("AGENT_VERSION", "1.3.0"),
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


def _extract_user_text(payload: Dict[str, Any]) -> str:
    msg = (payload.get("params") or {}).get("message") or {}
    parts = msg.get("parts") or []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            return str(p.get("text", "")).strip()
    return ""


@app.post("/a2a")
def a2a_endpoint(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Minimal A2A router compatible with tests:
      - expects {"method":"message/send", "params":{"message":{"parts":[{"type":"text","text":"..."}]}}}
      - replies with {"result":{"role":"assistant","messageId":"...","parts":[{"type":"text","text":"Hello ..."}]}}
    """
    method = payload.get("method")
    if method != "message/send":
        raise HTTPException(
            status_code=400, detail="Unsupported method; expected 'message/send'"
        )

    user_text = _extract_user_text(payload)
    # Deterministic "hello" (the test asserts text.lower().startswith('hello'))
    reply_text = f"Hello! You said: {user_text}" if user_text else "Hello! 👋"

    # Try to reuse the incoming messageId if present; otherwise generate a simple one
    incoming_msg = (payload.get("params") or {}).get("message") or {}
    message_id = incoming_msg.get("messageId") or "a2a-1"

    result = {
        "role": "assistant",
        "messageId": message_id,
        "parts": [{"type": "text", "text": reply_text}],
    }
    return {"result": result}
