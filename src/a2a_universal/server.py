# SPDX-License-Identifier: Apache-2.0
"""
FastAPI server for the Universal A2A Agent.

- Reads environment from `.env`, or falls back to `.env.example` if `.env` is missing.
- Serves the core Universal A2A APIs.
- Optionally mounts /knowledge (RAG) when A2A_ENABLE_KNOWLEDGE=1.

Endpoints:
  * POST /a2a           — Universal A2A envelope
  * POST /rpc           — JSON-RPC 2.0 wrapper
  * POST /openai/...    — OpenAI-compatible chat completions
  * GET  /healthz       — liveness
  * GET  /readyz        — readiness
  * GET  /.well-known/agent-card.json  — discovery metadata
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

# --- Load environment from .env, with fallback to .env.example (no override) ---
from dotenv import load_dotenv, find_dotenv


def _load_env_with_fallback() -> str:
    loaded_from = ""
    env_path = find_dotenv(filename=".env", usecwd=True)
    if env_path and load_dotenv(env_path, override=False):
        loaded_from = env_path
    if not loaded_from:
        example_path = find_dotenv(filename=".env.example", usecwd=True)
        if example_path and load_dotenv(example_path, override=False):
            loaded_from = example_path
    return loaded_from


_loaded_env_path = _load_env_with_fallback()

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ValidationError
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# --- Local Application Imports (after env is loaded) ---
from .card import agent_card
from .config import settings
from .frameworks import FrameworkBase, build_framework, list_frameworks
from .logging_config import configure_logging
from .models import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCSuccess,
    Message,
    TextPart,
)
from .providers import ProviderBase, build_provider, list_providers

# =============================================================================
# APPLICATION SETUP
# =============================================================================

configure_logging()
log = structlog.get_logger("a2a.server")
log.info("Environment loaded", env_file=_loaded_env_path or "(none; OS env only)")

PROVIDER: ProviderBase = build_provider()
FRAMEWORK: FrameworkBase = build_framework(PROVIDER)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Server startup sequence initiated...")

    all_providers = list_providers()
    all_frameworks = list_frameworks()
    log.info(
        "Component discovery complete",
        available_providers=all_providers,
        available_frameworks=all_frameworks,
    )
    log.info(
        "Active components initialized",
        provider={"id": PROVIDER.id, "name": PROVIDER.name, "ready": PROVIDER.ready},
        framework={
            "id": FRAMEWORK.id,
            "name": FRAMEWORK.name,
            "ready": FRAMEWORK.ready,
        },
    )
    yield
    log.info("Server shutdown sequence complete.")


app = FastAPI(
    title=settings.AGENT_NAME or "Universal A2A Agent",
    version=settings.AGENT_VERSION or "0.1.0",
    lifespan=lifespan,
    root_path=os.getenv("A2A_ROOT_PATH", ""),
    openapi_url="/openapi.json",
)

# Optionally mount /knowledge router (RAG)
if os.getenv("A2A_ENABLE_KNOWLEDGE", "0") == "1":
    try:
        from .routers import knowledge as knowledge_router

        app.include_router(knowledge_router.router)
        log.info("Knowledge API mounted", enabled=True, prefix="/knowledge")
    except Exception as e:  # pragma: no cover
        log.error("Failed to mount /knowledge router", error=str(e))
else:
    log.info("Knowledge API not mounted (A2A_ENABLE_KNOWLEDGE!=1)")

# =============================================================================
# MIDDLEWARE
# =============================================================================

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=getattr(settings, "ALLOWED_HOSTS", ["*"]),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=(settings.CORS_ALLOW_ORIGINS or ["*"]),
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=(settings.CORS_ALLOW_METHODS or ["*"]),
    allow_headers=(settings.CORS_ALLOW_HEADERS or ["*"]),
)

# =============================================================================
# HELPERS & MODELS
# =============================================================================


def _get_request_id(req: Request) -> str:
    return req.headers.get("x-request-id", f"gen-{uuid.uuid4()}")


def _get_diag_headers(request_id: str) -> Dict[str, str]:
    return {"X-Request-ID": request_id, "Cache-Control": "no-store"}


def _require_json_content_type(req: Request) -> None:
    content_type = (req.headers.get("content-type") or "").lower()
    if "application/json" not in content_type:
        raise HTTPException(
            status_code=415, detail="Content-Type must be application/json"
        )


def _extract_text_from_message(msg: Dict[str, Any]) -> str:
    for part in (msg or {}).get("parts", []):
        if isinstance(part, dict) and part.get("type") == "text":
            return part.get("text", "")
    return ""


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None


class ChatRequest(BaseModel):
    model: Optional[str] = "universal-a2a-agent"
    messages: List[ChatMessage]


# =============================================================================
# META & HEALTH
# =============================================================================


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/health", include_in_schema=False)
async def health_alias() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz", tags=["Monitoring"])
async def healthz(req: Request) -> JSONResponse:
    request_id = _get_request_id(req)
    return JSONResponse({"status": "ok"}, headers=_get_diag_headers(request_id))


@app.get("/readyz", tags=["Monitoring"])
async def readyz(req: Request) -> JSONResponse:
    request_id = _get_request_id(req)
    provider_meta = {
        "id": PROVIDER.id,
        "ready": PROVIDER.ready,
        "reason": PROVIDER.reason,
    }
    framework_meta = {
        "id": FRAMEWORK.id,
        "ready": FRAMEWORK.ready,
        "reason": FRAMEWORK.reason,
    }
    is_ready = PROVIDER.ready and FRAMEWORK.ready
    payload = {
        "status": "ready" if is_ready else "not_ready",
        "provider": provider_meta,
        "framework": framework_meta,
    }
    return JSONResponse(
        payload,
        status_code=(200 if is_ready else 503),
        headers=_get_diag_headers(request_id),
    )


@app.get("/.well-known/agent-card.json", tags=["Discovery"])
async def get_agent_card(req: Request) -> JSONResponse:
    request_id = _get_request_id(req)
    return JSONResponse(agent_card(), headers=_get_diag_headers(request_id))


# =============================================================================
# CORE API
# =============================================================================


@app.post("/a2a", tags=["A2A"])
async def a2a_endpoint(req: Request) -> JSONResponse:
    request_id = _get_request_id(req)
    _require_json_content_type(req)
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not (isinstance(body, dict) and body.get("method") == "message/send"):
        raise HTTPException(status_code=400, detail="Unsupported A2A payload structure")

    user_msg = body.get("params", {}).get("message", {})
    user_text = _extract_text_from_message(user_msg)
    context_id = user_msg.get("contextId")
    reply_text = await FRAMEWORK.execute([{"role": "user", "content": user_text}])
    agent_message = Message(
        role="agent", parts=[TextPart(text=reply_text)], contextId=context_id
    )
    return JSONResponse(
        {"result": agent_message.model_dump(exclude_none=True)},
        headers=_get_diag_headers(request_id),
    )


@app.post("/rpc", tags=["A2A"])
async def jsonrpc_endpoint(req: Request) -> JSONResponse:
    request_id = _get_request_id(req)
    _require_json_content_type(req)

    body = {}
    try:
        body = await req.json()
        rpc_request = JSONRPCRequest.model_validate(body)
    except ValidationError:
        error_response = JSONRPCError(
            id=body.get("id"), error={"code": -32600, "message": "Invalid Request"}
        )
        return JSONResponse(
            error_response.model_dump(exclude_none=True),
            status_code=200,
            headers=_get_diag_headers(request_id),
        )
    except Exception:
        error_response = JSONRPCError(
            id=None, error={"code": -32700, "message": "Parse error"}
        )
        return JSONResponse(
            error_response.model_dump(exclude_none=True),
            status_code=200,
            headers=_get_diag_headers(request_id),
        )

    if rpc_request.method != "message/send":
        error_response = JSONRPCError(
            id=rpc_request.id, error={"code": -32601, "message": "Method not found"}
        )
        return JSONResponse(
            error_response.model_dump(exclude_none=True),
            status_code=200,
            headers=_get_diag_headers(request_id),
        )

    user_msg = rpc_request.params.message
    user_text = _extract_text_from_message(user_msg.model_dump())
    context_id = user_msg.contextId
    reply_text = await FRAMEWORK.execute([{"role": "user", "content": user_text}])
    agent_message = Message(
        role="agent", parts=[TextPart(text=reply_text)], contextId=context_id
    )
    success_response = JSONRPCSuccess(id=rpc_request.id, result=agent_message)
    return JSONResponse(
        success_response.model_dump(exclude_none=True),
        status_code=200,
        headers=_get_diag_headers(request_id),
    )


# --- Friendly GET/HEAD/OPTIONS for /rpc (avoid 405 noise) ---


@app.get("/rpc", include_in_schema=False)
async def rpc_info(req: Request) -> JSONResponse:
    request_id = _get_request_id(req)
    try:
        post_url = str(req.url_for("jsonrpc_endpoint"))
    except Exception:
        post_url = "/rpc"
    return JSONResponse(
        {
            "status": "ok",
            "message": (
                "This is a JSON-RPC 2.0 endpoint. Use POST with body: "
                '{"jsonrpc":"2.0","method":"message/send","params":{...},"id":"..."}'
            ),
            "post_url": post_url,
            "methods": ["POST"],
        },
        headers={**_get_diag_headers(request_id), "Allow": "POST, OPTIONS"},
    )


@app.head("/rpc", include_in_schema=False)
async def rpc_head() -> Response:
    return Response(status_code=204, headers={"Allow": "POST, OPTIONS"})


@app.options("/rpc", include_in_schema=False)
async def rpc_options() -> Response:
    return Response(status_code=204, headers={"Allow": "POST, OPTIONS"})


# =============================================================================
# OPENAI-COMPATIBLE ENDPOINT
# =============================================================================


@app.post("/openai/v1/chat/completions", tags=["OpenAI"])
async def openai_chat_completions(req: Request) -> JSONResponse:
    request_id = _get_request_id(req)
    _require_json_content_type(req)

    try:
        payload = ChatRequest.model_validate_json(await req.body())
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    messages = [
        {"role": m.role, "content": str(m.content or "")} for m in payload.messages
    ]
    reply_text = await FRAMEWORK.execute(messages)

    response_payload = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return JSONResponse(response_payload, headers=_get_diag_headers(request_id))


# =============================================================================
# GLOBAL EXCEPTION HANDLERS
# =============================================================================


@app.exception_handler(Exception)
async def unhandled_exception_handler(req: Request, exc: Exception) -> JSONResponse:
    request_id = _get_request_id(req)
    log.error(
        "Unhandled exception caught",
        request_id=request_id,
        path=req.url.path,
        client=req.client.host if req.client else "unknown",
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        {"error": "Internal Server Error"},
        status_code=500,
        headers=_get_diag_headers(request_id),
    )


# -----------------------  Compatibility middleware (additive)  -----------------------


@app.middleware("http")
async def _a2a_rpc_compat_normalizer(request: Request, call_next):
    import json as _json

    path = request.url.path
    method = (request.method or "GET").upper()

    if method == "POST" and path in ("/a2a", "/rpc"):
        try:
            raw = await request.body()
            data = _json.loads(raw.decode("utf-8") or "{}")
            msg = None
            if isinstance(data, dict):
                if isinstance(data.get("params"), dict) and isinstance(
                    data["params"].get("message"), dict
                ):
                    msg = data["params"]["message"]
                elif isinstance(data.get("message"), dict):
                    msg = data["message"]
            changed = False
            if isinstance(msg, dict) and isinstance(msg.get("parts"), list):
                for p in msg["parts"]:
                    if isinstance(p, dict) and "text" in p and p.get("type") != "text":
                        if p.get("kind") == "text":
                            p.pop("kind", None)
                        p["type"] = "text"
                        changed = True
            if changed:
                new_raw = _json.dumps(data).encode("utf-8")

                async def _receive():
                    return {"type": "http.request", "body": new_raw, "more_body": False}

                request = Request(request.scope, _receive)
        except Exception:
            pass

    response = await call_next(request)

    if method == "POST" and path == "/a2a":
        try:
            body_bytes = b""
            async for chunk in response.body_iterator:
                body_bytes += chunk

            from fastapi.responses import JSONResponse as _JSONResponse

            payload = {}
            try:
                payload = _json.loads(body_bytes.decode("utf-8") or "{}")
            except Exception:
                return Response(
                    content=body_bytes,
                    status_code=response.status_code,
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() != "content-length"
                    },
                    media_type=response.media_type,
                )

            if "message" not in payload and "result" in payload:
                payload["message"] = payload["result"]

            headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() != "content-length"
            }
            return _JSONResponse(
                content=payload, status_code=response.status_code, headers=headers
            )
        except Exception:
            return response

    return response


# =============================================================================
# DEVELOPMENT SERVER LAUNCHER
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("A2A_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_PORT", "8000"))
    log.info("Starting server in development mode...", host=host, port=port)
    uvicorn.run(
        "a2a_universal.server:app",
        host=host,
        port=port,
        log_level="info",
        reload=True,
    )
