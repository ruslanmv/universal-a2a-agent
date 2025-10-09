# SPDX-License-Identifier: Apache-2.0
"""
a2a_universal.appkit
--------------------

Tiny, production-safe helper to turn a simple `handler(text) -> str | awaitable[str]`
into a full A2A service:

  - GET  /healthz
  - GET  /readyz
  - GET  /.well-known/agent-card.json
  - POST /a2a                       (raw A2A "message/send" envelope)
  - POST /rpc                       (JSON-RPC 2.0 "message/send")
  - POST /openai/v1/chat/completions (OpenAI-compatible chat completions)

Design goals:
  * Minimal user surface (one function).
  * Robust parsing for A2A/OpenAI message bodies.
  * Consistent responses (match universal server shapes).
  * Clear diagnostics (X-Request-ID + no-store).
  * Optional root_path for reverse proxies.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

# Public type for user handler
Handler = Union[Callable[[str], str], Callable[[str], Awaitable[str]]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_coro(fn: Handler) -> bool:
    # type: ignore[arg-type]
    return asyncio.iscoroutinefunction(fn)  # type: ignore


def _request_id(req: Request) -> str:
    return req.headers.get("x-request-id") or f"gen-{uuid.uuid4()}"


def _diag_headers(rid: str) -> Dict[str, str]:
    return {
        "X-Request-ID": rid,
        "Cache-Control": "no-store",
    }


def _require_json(req: Request) -> None:
    ctype = (req.headers.get("content-type") or "").lower()
    if "application/json" not in ctype:
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")


def _extract_text_from_a2a(body: Dict[str, Any]) -> str:
    """
    Pull the first text part from the A2A envelope:
      {"method":"message/send","params":{"message":{"parts":[{"type":"text","text":"..."}]}}}
    """
    msg = ((body or {}).get("params") or {}).get("message") or {}
    parts = msg.get("parts") or []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            t = p.get("text")
            if isinstance(t, str) and t.strip():
                return t.strip()
    # Fallback: some clients might send content at top-level (non-standard)
    text = (msg.get("content") or "").strip() if isinstance(msg.get("content"), str) else ""
    return text


def _extract_text_from_openai(body: Dict[str, Any]) -> str:
    """
    Pull the last user text from an OpenAI chat-completions payload.
    Supports both string content and list-of-parts content.
    """
    msgs = (body or {}).get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text", "")
                        if isinstance(t, str) and t.strip():
                            return t.strip()
    return ""


def _wrap_a2a_message(text: str, *, context_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Build the standard A2A agent message body that universal server returns under "result".
    """
    msg: Dict[str, Any] = {
        "role": "agent",
        "messageId": str(uuid.uuid4()),
        "parts": [{"type": "text", "text": text}],
    }
    if context_id:
        msg["contextId"] = context_id
    return msg


async def _call_handler(handler: Handler, text: str) -> str:
    # If the handler is async, await it; else offload to threadpool (so we don't block event loop)
    if _is_coro(handler):  # type: ignore[arg-type]
        return await handler(text)  # type: ignore[misc]
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, handler, text)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build(
    handler: Handler,
    *,
    name: str = os.getenv("AGENT_NAME", "Universal A2A Agent"),
    description: str = os.getenv("AGENT_DESCRIPTION", "A2A-compatible agent"),
    version: str = os.getenv("AGENT_VERSION", "0.1.0"),
    protocol_version: str = os.getenv("PROTOCOL_VERSION", "0.3.0"),
    root_path: Optional[str] = None,  # helpful when served behind a sub-path gateway
    preferred_transport: str = "JSONRPC",
    skills: Optional[List[Dict[str, Any]]] = None,
    readiness_check: Optional[Callable[[], bool]] = None,  # simple hook; return True if ready
) -> FastAPI:
    """
    Build a FastAPI app exposing a production-friendly A2A surface around a single text handler.

    Args:
        handler: Callable that accepts a single text string and returns/awaits a string reply.
        name, description, version: Metadata for the service (also reflected in agent-card).
        protocol_version: Agent card protocol version (default "0.3.0").
        root_path: Optional ASGI root_path (also available via env A2A_ROOT_PATH).
        preferred_transport: "JSONRPC" by default (what the card advertises).
        skills: Optional list of skills metadata to include in the agent-card.
        readiness_check: Optional function to gate /readyz readiness.

    Returns:
        FastAPI application.
    """
    rp = root_path if root_path is not None else os.getenv("A2A_ROOT_PATH", "")

    app = FastAPI(
        title=name,
        description=description,
        version=version,
        root_path=rp,
        # Keep default docs enabled; production gateways may disable them.
        openapi_url="/openapi.json",
    )

    # ---------------------
    # Meta / Health routes
    # ---------------------

    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/docs", status_code=307)

    @app.get("/healthz", tags=["Monitoring"])
    async def _healthz(req: Request) -> JSONResponse:
        rid = _request_id(req)
        return JSONResponse({"status": "ok"}, headers=_diag_headers(rid))

    @app.get("/readyz", tags=["Monitoring"])
    async def _readyz(req: Request) -> JSONResponse:
        rid = _request_id(req)
        ok = True
        if readiness_check is not None:
            try:
                ok = bool(readiness_check())
            except Exception:
                ok = False
        payload = {"status": "ready" if ok else "not_ready"}
        return JSONResponse(payload, status_code=200 if ok else 503, headers=_diag_headers(rid))

    @app.get("/.well-known/agent-card.json", tags=["Discovery"])
    async def _agent_card(req: Request) -> JSONResponse:
        rid = _request_id(req)
        base = (os.getenv("PUBLIC_URL") or str(req.base_url)).rstrip("/")
        card = {
            "protocolVersion": protocol_version,
            "name": name,
            "description": description,
            "version": version,
            "url": f"{base}/rpc",
            "preferredTransport": preferred_transport,
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "capabilities": {"streaming": False, "pushNotifications": False},
            "skills": skills
            or [
                {
                    "id": "say-hello",
                    "name": "Say Hello",
                    "description": "Responds with a friendly greeting.",
                    "tags": ["hello", "greeting"],
                }
            ],
        }
        return JSONResponse(card, headers=_diag_headers(rid))

    # ---------------------
    # Core protocol routes
    # ---------------------

    @app.post("/a2a", tags=["A2A"])
    async def _a2a(req: Request) -> JSONResponse:
        """
        Raw A2A envelope. Expect:
          {"method":"message/send","params":{"message":{"role":"user","parts":[{"type":"text","text":"..."}]}}}
        Response mirrors universal server shape:
          {"result": {<agent_message>}}
        """
        rid = _request_id(req)
        _require_json(req)
        try:
            body = await req.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        if not (isinstance(body, dict) and body.get("method") == "message/send"):
            raise HTTPException(status_code=400, detail="Unsupported A2A payload structure")

        params = body.get("params") or {}
        user_msg = params.get("message") or {}
        context_id = user_msg.get("contextId")
        text = _extract_text_from_a2a(body)

        reply = await _call_handler(handler, text)
        agent_message = _wrap_a2a_message(reply, context_id=context_id)

        return JSONResponse({"result": agent_message}, headers=_diag_headers(rid))

    @app.post("/rpc", tags=["A2A"])
    async def _rpc(req: Request) -> JSONResponse:
        """
        JSON-RPC 2.0:
          {"jsonrpc":"2.0","id":"1","method":"message/send","params":{"message":{...}}}
        """
        rid = _request_id(req)
        _require_json(req)
        try:
            body = await req.json()
        except Exception:
            # Parse error
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
                headers=_diag_headers(rid),
            )

        mid = body.get("id")
        method = body.get("method")
        if method != "message/send":
            return JSONResponse(
                {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "Method not found"}},
                headers=_diag_headers(rid),
            )

        params = body.get("params") or {}
        user_msg = params.get("message") or {}
        context_id = user_msg.get("contextId")
        text = _extract_text_from_a2a({"params": params})

        try:
            reply = await _call_handler(handler, text)
            result = _wrap_a2a_message(reply, context_id=context_id)
            return JSONResponse(
                {"jsonrpc": "2.0", "id": mid, "result": result},
                headers=_diag_headers(rid),
            )
        except Exception as e:  # safety
            return JSONResponse(
                {"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": f"Server error: {e}"}},
                headers=_diag_headers(rid),
            )

    @app.post("/openai/v1/chat/completions", tags=["OpenAI"])
    async def _openai_chat(req: Request, body: Dict[str, Any] = Body(...)) -> JSONResponse:
        """
        OpenAI-compatible chat completions (minimal subset).
        Input:
          {"model":"...", "messages":[{"role":"user","content":"..."}]}
        """
        rid = _request_id(req)
        _require_json(req)

        # Defensive extraction
        text = _extract_text_from_openai(body)
        reply = await _call_handler(handler, text)

        payload = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "a2a-generic"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return JSONResponse(payload, headers=_diag_headers(rid))

    # ---------------------
    # Global exception guard
    # ---------------------

    @app.exception_handler(Exception)
    async def _unhandled(req: Request, exc: Exception) -> JSONResponse:
        rid = _request_id(req)
        # Do not leak internals; keep a generic 500 but include diag headers.
        return JSONResponse(
            {"error": "Internal Server Error"},
            status_code=500,
            headers=_diag_headers(rid),
        )

    return app
