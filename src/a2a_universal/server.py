from __future__ import annotations
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from typing import Optional
import uuid
import os
import time

from .models import (
    TextPart, Message,
    A2AResponse,
    JSONRPCRequest, JSONRPCSuccess, JSONRPCError
)
from .card import agent_card
from .adapters import private_adapter as pad

app = FastAPI(title="Universal A2A Agent")

# --- helpers ---

def make_agent_message(text: str) -> Message:
    return Message(
        role="agent",
        messageId=str(uuid.uuid4()),
        parts=[TextPart(text=text)]
    )

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/.well-known/agent-card.json")
async def card():
    return agent_card()

# ---- Raw A2A ----
@app.post("/a2a")
async def a2a_endpoint(req: Request):
    body = await req.json()

    def normalize_parts(msg):
        if not msg:
            return msg
        parts = msg.get("parts", [])
        norm = []
        for p in parts:
            if p.get("type") == "text" or p.get("kind") == "text":
                norm.append({"type": "text", "text": p.get("text", "")})
        msg["parts"] = norm
        return msg

    if isinstance(body, dict) and body.get("method") == "message/send":
        params = body.get("params", {})
        msg = normalize_parts(params.get("message", {}))
        user_text = None
        for p in msg.get("parts", []):
            if p.get("type") == "text":
                user_text = p.get("text")
                break
        greeting = "Hello, World!" if not user_text else f"Hello, you said: {user_text}"
        return A2AResponse(message=make_agent_message(greeting)).model_dump()
    raise HTTPException(400, detail="Unsupported A2A payload")

# ---- JSON-RPC 2.0 ----
@app.post("/rpc")
async def jsonrpc(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(JSONRPCError(id=None, error={"code": -32700, "message": "Parse error"}).model_dump())

    try:
        rpc = JSONRPCRequest.model_validate(body)
    except ValidationError as e:
        return JSONResponse(JSONRPCError(id=body.get("id"), error={"code": -32600, "message": f"Invalid Request: {e}"}).model_dump())

    if rpc.method != "message/send":
        return JSONResponse(JSONRPCError(id=rpc.id, error={"code": -32601, "message": "Method not found"}).model_dump())

    user_text = None
    for p in rpc.params.message.parts:
        if p.type == "text":
            user_text = p.text
            break
    greeting = "Hello, World!" if not user_text else f"Hello, you said: {user_text}"
    return JSONResponse(JSONRPCSuccess(id=rpc.id, result=A2AResponse(message=make_agent_message(greeting))).model_dump())

# ---- OpenAI Chat Completions (for UI/Orchestrators) ----
@app.post("/openai/v1/chat/completions")
async def openai_chat_completions(req: Request):
    body = await req.json()

    def last_user_text(messages):
        if not isinstance(messages, list):
            return ""
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

    user_text = last_user_text(body.get("messages", []))
    reply_text = "Hello, World!" if not user_text else f"Hello, you said: {user_text}"
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": now,
        "model": body.get("model", "universal-a2a-hello"),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": reply_text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

# ---- Private adapter (enterprise) ----
_PRIV_ENABLED = os.getenv("PRIVATE_ADAPTER_ENABLED", "false").lower() == "true"
_PRIV_SCHEME = os.getenv("PRIVATE_ADAPTER_AUTH_SCHEME", "NONE").upper()
_PRIV_TOKEN = os.getenv("PRIVATE_ADAPTER_AUTH_TOKEN", "")
_PRIV_PATH = os.getenv("PRIVATE_ADAPTER_PATH", "/enterprise/v1/agent")


def _check_private_auth(req: Request) -> None:
    if not _PRIV_ENABLED:
        raise HTTPException(status_code=404, detail="Not Found")
    if _PRIV_SCHEME == "NONE":
        return
    auth = req.headers.get("Authorization", "")
    if _PRIV_SCHEME == "BEARER":
        if not auth.startswith("Bearer ") or auth.split(" ", 1)[1] != _PRIV_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")
    elif _PRIV_SCHEME == "API_KEY":
        if req.headers.get("X-API-Key") != _PRIV_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")

@app.post(_PRIV_PATH)
async def private_adapter_endpoint(req: Request):
    _check_private_auth(req)
    body = await req.json()
    user_text = pad.extract_user_text(body)
    reply_text = "Hello, World!" if not user_text else f"Hello, you said: {user_text}"
    return pad.make_response(reply_text, body)
