# src/a2a_universal/server.py
from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ValidationError

from .config import settings
from .logging_config import configure_logging
from .providers import ProviderBase, build_provider
from .frameworks import FrameworkBase, build_framework
from .models import (
    TextPart,
    Message,
    A2AResponse,
    JSONRPCRequest,
    JSONRPCSuccess,
    JSONRPCError,
)
from .card import agent_card
from .adapters import private_adapter as pad


# =============================================================================
# Logging & Application Setup
# =============================================================================

configure_logging()
log = logging.getLogger("a2a.server")

def _log(level: str, event: str, **fields: Any) -> None:
    """Structured logging helper: avoids double JSON encoding."""
    fn = getattr(log, level, log.info)
    fn(event, extra=fields)


def _request_id(req: Request) -> str:
    """Return incoming X-Request-ID or generate a new one."""
    rid = req.headers.get("x-request-id")
    return rid if rid else str(uuid.uuid4())


def _with_diag_headers(rid: str) -> Dict[str, str]:
    """Standard headers we attach to all responses."""
    return {
        "X-Request-ID": rid,
        "Cache-Control": "no-store",
    }


def _require_json(req: Request) -> None:
    """Ensure Content-Type is application/json."""
    ctype = (req.headers.get("content-type") or "").lower()
    if "application/json" not in ctype:
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")


# FastAPI application
app = FastAPI(
    title=settings.AGENT_NAME or "Universal A2A Agent",
    version=settings.AGENT_VERSION or "0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS (configurable; permissive defaults suitable for local/dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=(settings.CORS_ALLOW_ORIGINS or ["*"]),
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=(settings.CORS_ALLOW_METHODS or ["*"]),
    allow_headers=(settings.CORS_ALLOW_HEADERS or ["*"]),
)

# (Optional) Trusted hosts — if you wish to lock down Host headers in prod,
# configure a list via env and uncomment below.
# app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts or ["*"])


# =============================================================================
# Provider & Framework Initialization (Runtime Injection)
# =============================================================================

PROVIDER: ProviderBase = build_provider()
FRAMEWORK: FrameworkBase = build_framework(PROVIDER)

def _prov_meta(p: ProviderBase) -> Dict[str, Any]:
    return {
        "id": getattr(p, "id", "unknown"),
        "name": getattr(p, "name", "unknown"),
        "ready": bool(getattr(p, "ready", False)),
        "reason": getattr(p, "reason", ""),
    }

def _fw_meta(f: FrameworkBase) -> Dict[str, Any]:
    return {
        "id": getattr(f, "id", "unknown"),
        "name": getattr(f, "name", "unknown"),
        "ready": bool(getattr(f, "ready", False)),
        "reason": getattr(f, "reason", ""),
    }


@app.on_event("startup")
async def _on_startup() -> None:
    _log("info", "startup", provider=_prov_meta(PROVIDER), framework=_fw_meta(FRAMEWORK))


# =============================================================================
# Models & Helpers
# =============================================================================

def make_agent_message(text: str) -> Message:
    return Message(
        role="agent",
        messageId=str(uuid.uuid4()),
        parts=[TextPart(text=text)],
    )


def _extract_text_part(msg: Dict[str, Any]) -> str:
    """Extract first text part from an A2A message (dict)."""
    for p in (msg or {}).get("parts", []) or []:
        if isinstance(p, dict) and (p.get("type") == "text" or p.get("kind") == "text"):
            return p.get("text", "")
    return ""


# Minimal OpenAI chat schema (tolerant)
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Union[str, Dict[str, Any]]]]] = None


class ChatRequest(BaseModel):
    model: Optional[str] = "universal-a2a-hello"
    messages: List[ChatMessage]


def _to_text(content: Any) -> str:
    """Normalize OpenAI-style content into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join([t for t in parts if t])
    return ""


# =============================================================================
# Meta & Health
# =============================================================================

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    # Simple convenience: redirect to API docs
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/healthz")
async def healthz(req: Request) -> JSONResponse:
    rid = _request_id(req)
    return JSONResponse({"status": "ok"}, headers=_with_diag_headers(rid))


@app.get("/readyz")
async def readyz(req: Request) -> JSONResponse:
    rid = _request_id(req)
    provider_ready = bool(getattr(PROVIDER, "ready", False))
    framework_ready = bool(getattr(FRAMEWORK, "ready", False))
    ok = provider_ready and framework_ready
    payload = {
        "status": "ready" if ok else "not-ready",
        "provider": _prov_meta(PROVIDER),
        "framework": _fw_meta(FRAMEWORK),
    }
    return JSONResponse(payload, status_code=200 if ok else 503, headers=_with_diag_headers(rid))


@app.get("/.well-known/agent-card.json")
async def card(req: Request) -> JSONResponse:
    rid = _request_id(req)
    return JSONResponse(agent_card(), headers=_with_diag_headers(rid))


# =============================================================================
# A2A (Raw) Endpoint
# =============================================================================

@app.post("/a2a")
async def a2a_endpoint(req: Request) -> JSONResponse:
    rid = _request_id(req)
    _require_json(req)

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not (isinstance(body, dict) and body.get("method") == "message/send"):
        raise HTTPException(status_code=400, detail="Unsupported A2A payload")

    params = body.get("params", {}) or {}
    user_text = _extract_text_part(params.get("message", {}))

    # Execute via framework
    reply_text = await FRAMEWORK.execute([{"role": "user", "content": user_text}])
    resp = A2AResponse(message=make_agent_message(reply_text)).model_dump()

    _log("info", "a2a.request",
         request_id=rid,
         method="message/send",
         user_text_len=len(user_text),
         provider=_prov_meta(PROVIDER),
         framework=_fw_meta(FRAMEWORK))

    return JSONResponse(resp, headers=_with_diag_headers(rid))


# =============================================================================
# JSON-RPC 2.0
# =============================================================================

@app.post("/rpc")
async def jsonrpc(req: Request) -> JSONResponse:
    rid = _request_id(req)
    _require_json(req)

    try:
        body = await req.json()
    except Exception:
        return JSONResponse(
            JSONRPCError(id=None, error={"code": -32700, "message": "Parse error"}).model_dump(),
            status_code=200,  # JSON-RPC spec uses 200 with error body
            headers=_with_diag_headers(rid),
        )

    try:
        rpc = JSONRPCRequest.model_validate(body)
    except ValidationError as e:
        return JSONResponse(
            JSONRPCError(
                id=(body.get("id") if isinstance(body, dict) else None),
                error={"code": -32600, "message": f"Invalid Request: {e}"},
            ).model_dump(),
            status_code=200,
            headers=_with_diag_headers(rid),
        )

    if rpc.method != "message/send":
        return JSONResponse(
            JSONRPCError(id=rpc.id, error={"code": -32601, "message": "Method not found"}).model_dump(),
            status_code=200,
            headers=_with_diag_headers(rid),
        )

    # Extract first text part
    user_text = ""
    for p in rpc.params.message.parts:
        if p.type == "text":
            user_text = p.text or ""
            break

    reply_text = await FRAMEWORK.execute([{"role": "user", "content": user_text}])

    _log("info", "rpc.request",
         request_id=rid,
         method=rpc.method,
         user_text_len=len(user_text),
         provider=_prov_meta(PROVIDER),
         framework=_fw_meta(FRAMEWORK))

    return JSONResponse(
        JSONRPCSuccess(id=rpc.id, result=A2AResponse(message=make_agent_message(reply_text))).model_dump(),
        status_code=200,
        headers=_with_diag_headers(rid),
    )


# =============================================================================
# OpenAI Chat Completions (for UIs / Orchestrators)
# =============================================================================

@app.post("/openai/v1/chat/completions")
async def openai_chat_completions(req: Request) -> JSONResponse:
    rid = _request_id(req)
    _require_json(req)

    raw = await req.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty JSON body")

    try:
        payload = ChatRequest.model_validate_json(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Convert to internal message shape
    messages = [{"role": m.role, "content": _to_text(m.content)} for m in payload.messages]
    reply_text = await FRAMEWORK.execute(messages)
    now = int(time.time())

    _log("info", "openai.request",
         request_id=rid,
         model=payload.model or "universal-a2a-hello",
         turns=len(messages),
         provider=_prov_meta(PROVIDER),
         framework=_fw_meta(FRAMEWORK))

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": now,
            "model": payload.model or "universal-a2a-hello",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        headers=_with_diag_headers(rid),
    )


# =============================================================================
# Private Adapter (Enterprise)
# =============================================================================

_PRIV_ENABLED = settings.PRIVATE_ADAPTER_ENABLED
_PRIV_SCHEME = (settings.PRIVATE_ADAPTER_AUTH_SCHEME or "NONE").upper()
_PRIV_TOKEN = settings.PRIVATE_ADAPTER_AUTH_TOKEN or ""
_PRIV_PATH = settings.PRIVATE_ADAPTER_PATH or "/enterprise/v1/agent"


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
async def private_adapter_endpoint(req: Request) -> JSONResponse:
    rid = _request_id(req)
    _check_private_auth(req)
    _require_json(req)

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_text = pad.extract_user_text(body)
    reply_text = await FRAMEWORK.execute([{"role": "user", "content": user_text}])
    resp = pad.make_response(reply_text, body)

    _log("info", "private.request",
         request_id=rid,
         payload_shape="enterprise",
         provider=_prov_meta(PROVIDER),
         framework=_fw_meta(FRAMEWORK))

    return JSONResponse(resp, headers=_with_diag_headers(rid))


# =============================================================================
# Global Exception Handlers (polish)
# =============================================================================

@app.exception_handler(ValidationError)
async def _validation_error_handler(_: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse({"error": "validation_error", "detail": str(exc)}, status_code=400)


@app.exception_handler(HTTPException)
async def _http_error_handler(req: Request, exc: HTTPException) -> JSONResponse:
    rid = _request_id(req)
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=_with_diag_headers(rid))


@app.exception_handler(Exception)
async def _unhandled_error_handler(req: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(req)
    _log("error", "unhandled.exception", request_id=rid, error=str(exc))
    # Avoid leaking internals; log has details.
    return JSONResponse({"error": "internal_error"}, status_code=500, headers=_with_diag_headers(rid))
