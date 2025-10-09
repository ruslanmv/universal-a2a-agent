# SPDX-License-Identifier: Apache-2.0
"""
FastAPI server for the Universal A2A Agent.

This module sets up and runs the main web server, handling various API endpoints
including A2A, JSON-RPC, and an OpenAI-compatible chat completion endpoint.
It integrates dynamic provider and framework loading, structured logging,
and production-ready security middleware.
"""

from __future__ import annotations

import os  # <-- Added: for A2A_ROOT_PATH support
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import structlog
from fastapi import FastAPI, HTTPException, Request, Response  # <-- Added Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ValidationError
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# --- Local Application Imports ---
from .adapters import private_adapter as pad
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

# Initialize structured logging. This should be the first action.
configure_logging()
log = structlog.get_logger("a2a.server")

# Load the selected provider and framework at startup.
# This follows a "fail-fast" approach; if essential components cannot be
# loaded, the application will not start correctly.
PROVIDER: ProviderBase = build_provider()
FRAMEWORK: FrameworkBase = build_framework(PROVIDER)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    This is the modern replacement for @app.on_event("startup").
    """
    log.info("Server startup sequence initiated...")

    # Log discovered and active components for diagnostics.
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
        framework={"id": FRAMEWORK.id, "name": FRAMEWORK.name, "ready": FRAMEWORK.ready},
    )

    yield

    log.info("Server shutdown sequence complete.")


# Initialize the FastAPI application.
# Added: root_path honors a deployment prefix when running behind a proxy/gateway.
app = FastAPI(
    title=settings.AGENT_NAME or "Universal A2A Agent",
    version=settings.AGENT_VERSION or "0.1.0",
    lifespan=lifespan,
    root_path=os.getenv("A2A_ROOT_PATH", ""),  # <--- NEW
    # In a secure production environment, you might disable the docs:
    # docs_url=None,
    # redoc_url=None,
    openapi_url="/openapi.json",
)


# =============================================================================
# MIDDLEWARE CONFIGURATION
# =============================================================================
# Middleware is processed in the reverse order it's added.

# IMPORTANT: Add TrustedHostMiddleware to prevent Host header attacks.
# In production, set the ALLOWED_HOSTS environment variable to a comma-separated
# list of your domain names (e.g., "example.com,api.example.com").
#
# THE FIX: Use getattr to safely access ALLOWED_HOSTS. If the attribute
# doesn't exist in the config, it defaults to ["*"] for backward compatibility.
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=getattr(settings, "ALLOWED_HOSTS", ["*"]),
)

# Configure CORS (Cross-Origin Resource Sharing).
# For production, `CORS_ALLOW_ORIGINS` should be a specific list of domains,
# not the wildcard "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=(settings.CORS_ALLOW_ORIGINS or ["*"]),
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=(settings.CORS_ALLOW_METHODS or ["*"]),
    allow_headers=(settings.CORS_ALLOW_HEADERS or ["*"]),
)


# =============================================================================
# HELPER FUNCTIONS & MODELS
# =============================================================================


def _get_request_id(req: Request) -> str:
    """
    Retrieves the X-Request-ID header or generates a new one.
    This is crucial for request tracing and debugging across services.
    """
    return req.headers.get("x-request-id", f"gen-{uuid.uuid4()}")


def _get_diag_headers(request_id: str) -> Dict[str, str]:
    """
    Returns standard diagnostic and security headers for all responses.
    """
    return {
        "X-Request-ID": request_id,
        "Cache-Control": "no-store",  # Prevents caching of sensitive API responses.
    }


def _require_json_content_type(req: Request) -> None:
    """
    Raises an HTTPException if the request Content-Type is not application/json.
    """
    content_type = (req.headers.get("content-type") or "").lower()
    if "application/json" not in content_type:
        log.warning(
            "Unsupported Content-Type",
            content_type=content_type,
            client_host=req.client.host if req.client else "unknown",
        )
        raise HTTPException(
            status_code=415, detail="Content-Type must be application/json"
        )


def _extract_text_from_message(msg: Dict[str, Any]) -> str:
    """
    Safely extracts the first text part from a standard message dictionary.
    """
    for part in (msg or {}).get("parts", []):
        # FIX: The model uses 'type', not 'kind'.
        if isinstance(part, dict) and part.get("type") == "text":
            return part.get("text", "")
    return ""


class ChatMessage(BaseModel):
    """Represents a single message in an OpenAI-compatible chat request."""

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None


class ChatRequest(BaseModel):
    """Represents the body of an OpenAI-compatible chat completions request."""

    model: Optional[str] = "universal-a2a-agent"
    messages: List[ChatMessage]


# =============================================================================
# META & HEALTH ENDPOINTS
# =============================================================================


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirects the root path to the API documentation for convenience."""
    return RedirectResponse(url="/docs", status_code=307)


# NEW: convenience alias (kept out of schema) for platforms probing /health
@app.get("/health", include_in_schema=False)
async def health_alias() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz", tags=["Monitoring"])
async def healthz(req: Request) -> JSONResponse:
    """
    A simple health check endpoint. Returns 200 OK if the server is running.
    Does not check dependencies.
    """
    request_id = _get_request_id(req)
    return JSONResponse({"status": "ok"}, headers=_get_diag_headers(request_id))


@app.get("/readyz", tags=["Monitoring"])
async def readyz(req: Request) -> JSONResponse:
    """
    A readiness probe endpoint. Checks if the core components (Provider and
    Framework) are ready to accept traffic.
    """
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
    status_code = 200 if is_ready else 503  # Service Unavailable

    payload = {
        "status": "ready" if is_ready else "not_ready",
        "provider": provider_meta,
        "framework": framework_meta,
    }
    return JSONResponse(
        payload, status_code=status_code, headers=_get_diag_headers(request_id)
    )


@app.get("/.well-known/agent-card.json", tags=["Discovery"])
async def get_agent_card(req: Request) -> JSONResponse:
    """Returns the agent's discovery card."""
    request_id = _get_request_id(req)
    return JSONResponse(agent_card(), headers=_get_diag_headers(request_id))


# =============================================================================
# CORE API ENDPOINTS
# =============================================================================


@app.post("/a2a", tags=["A2A"])
async def a2a_endpoint(req: Request) -> JSONResponse:
    """Handles raw A2A message/send requests."""
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

    log.info(
        "A2A request processed",
        request_id=request_id,
        user_text_len=len(user_text),
        reply_text_len=len(reply_text),
    )
    return JSONResponse(
        {"result": agent_message.model_dump(exclude_none=True)},
        headers=_get_diag_headers(request_id),
    )


@app.post("/rpc", tags=["A2A"])
async def jsonrpc_endpoint(req: Request) -> JSONResponse:
    """Handles JSON-RPC 2.0 message/send requests."""
    request_id = _get_request_id(req)
    _require_json_content_type(req)

    body = {}
    try:
        body = await req.json()
        rpc_request = JSONRPCRequest.model_validate(body)
    except ValidationError as e:
        log.warning(
            "Invalid JSON-RPC request",
            request_id=request_id,
            error=str(e),
            body=body,
        )
        error_response = JSONRPCError(
            id=body.get("id"),
            error={"code": -32600, "message": "Invalid Request"},
        )
        return JSONResponse(
            error_response.model_dump(exclude_none=True),
            status_code=200,
            headers=_get_diag_headers(request_id),
        )
    except Exception:
        error_response = JSONRPCError(
            id=None,
            error={"code": -32700, "message": "Parse error"},
        )
        return JSONResponse(
            error_response.model_dump(exclude_none=True),
            status_code=200,
            headers=_get_diag_headers(request_id),
        )

    if rpc_request.method != "message/send":
        error_response = JSONRPCError(
            id=rpc_request.id,
            error={"code": -32601, "message": "Method not found"},
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

    log.info(
        "JSON-RPC request processed",
        request_id=request_id,
        user_text_len=len(user_text),
        reply_text_len=len(reply_text),
    )
    return JSONResponse(
        success_response.model_dump(exclude_none=True),
        status_code=200,
        headers=_get_diag_headers(request_id),
    )


# --- Minor patch: friendly GET/HEAD/OPTIONS for /rpc to avoid 405 noise -------

@app.get("/rpc", include_in_schema=False)
async def rpc_info(req: Request) -> JSONResponse:
    """
    Informational endpoint for browsers/health probes that hit GET /rpc.
    Real JSON-RPC calls must use POST /rpc with a JSON body.
    """
    request_id = _get_request_id(req)
    try:
        post_url = str(req.url_for("jsonrpc_endpoint"))
    except Exception:
        post_url = "/rpc"
    return JSONResponse(
        {
            "status": "ok",
            "message": (
                "This is a JSON-RPC 2.0 endpoint. "
                "Use POST with body: "
                '{"jsonrpc":"2.0","method":"message/send","params":{...},"id":"..."}'
            ),
            "post_url": post_url,
            "methods": ["POST"],
        },
        headers={**_get_diag_headers(request_id), "Allow": "POST, OPTIONS"},
    )


@app.head("/rpc", include_in_schema=False)
async def rpc_head() -> Response:
    """Fast path for load-balancer/monitor checks that send HEAD to /rpc."""
    return Response(status_code=204, headers={"Allow": "POST, OPTIONS"})


@app.options("/rpc", include_in_schema=False)
async def rpc_options() -> Response:
    """Explicit OPTIONS (CORS middleware usually covers this)."""
    return Response(status_code=204, headers={"Allow": "POST, OPTIONS"})


# =============================================================================
# OPENAI-COMPATIBLE ENDPOINT
# =============================================================================


@app.post("/openai/v1/chat/completions", tags=["OpenAI"])
async def openai_chat_completions(req: Request) -> JSONResponse:
    """Provides an OpenAI-compatible endpoint for chat completions."""
    request_id = _get_request_id(req)
    _require_json_content_type(req)

    try:
        payload = ChatRequest.model_validate_json(await req.body())
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    # Convert to the internal message format for the framework.
    messages = [{"role": m.role, "content": str(m.content or "")} for m in payload.messages]

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

    log.info(
        "OpenAI completion request processed",
        request_id=request_id,
        model=payload.model,
        num_messages=len(messages),
    )
    return JSONResponse(response_payload, headers=_get_diag_headers(request_id))


# =============================================================================
# GLOBAL EXCEPTION HANDLERS
# =============================================================================


@app.exception_handler(Exception)
async def unhandled_exception_handler(req: Request, exc: Exception) -> JSONResponse:
    """
    Catches any unhandled exceptions and returns a generic 500 error.
    This prevents leaking internal implementation details to the client.
    The full traceback is logged for debugging.
    """
    request_id = _get_request_id(req)
    log.error(
        "Unhandled exception caught",
        request_id=request_id,
        path=req.url.path,
        client=req.client.host if req.client else "unknown",
        error=str(exc),
        exc_info=True,  # This is critical for logging the stack trace.
    )
    return JSONResponse(
        {"error": "Internal Server Error"},
        status_code=500,
        headers=_get_diag_headers(request_id),
    )


# -----------------------  Compatibility middleware (additive)  -----------------------

@app.middleware("http")
async def _a2a_rpc_compat_normalizer(request: Request, call_next):
    """
    Additive compatibility shim:
      - Normalizes POST bodies for /a2a and /rpc so parts with {"text": "..."} or {"kind":"text"}
        are converted to {"type":"text","text":"..."} BEFORE your handlers parse the body.
      - Augments /a2a responses by mirroring top-level {"message": <result>} for clients/tests
        that expect 'message' instead of 'result'. Existing payload remains untouched.
    """
    import json as _json

    path = request.url.path
    method = (request.method or "GET").upper()

    # ---- Normalize request bodies for /a2a and /rpc (input text extraction fix) ----
    if method == "POST" and path in ("/a2a", "/rpc"):
        try:
            raw = await request.body()
            data = _json.loads(raw.decode("utf-8") or "{}")
            # Locate the 'message' envelope (A2A or JSON-RPC params)
            msg = None
            if isinstance(data, dict):
                if isinstance(data.get("params"), dict) and isinstance(data["params"].get("message"), dict):
                    msg = data["params"]["message"]
                elif isinstance(data.get("message"), dict):
                    msg = data["message"]
            # Normalize parts -> ensure {"type":"text","text":"..."}
            changed = False
            if isinstance(msg, dict) and isinstance(msg.get("parts"), list):
                for p in msg["parts"]:
                    if isinstance(p, dict) and "text" in p:
                        if p.get("type") != "text":
                            # Convert {"kind":"text"} or bare {"text": "..."} into canonical shape
                            if p.get("kind") == "text":
                                p.pop("kind", None)
                            p["type"] = "text"
                            changed = True
            if changed:
                new_raw = _json.dumps(data).encode("utf-8")
                # Rebuild the request's receive() so downstream sees normalized JSON
                async def _receive():
                    return {"type": "http.request", "body": new_raw, "more_body": False}
                request = Request(request.scope, _receive)
        except Exception:
            # If anything goes wrong, fall back to original request
            pass

    # Call downstream handler
    response = await call_next(request)

    # ---- Augment /a2a responses with top-level 'message' (additive, keeps 'result') ----
    if method == "POST" and path == "/a2a":
        try:
            # Drain the response body (it's a stream), then rebuild
            body_bytes = b""
            async for chunk in response.body_iterator:
                body_bytes += chunk

            import typing as _t
            from fastapi.responses import JSONResponse as _JSONResponse

            payload = {}
            try:
                payload = _json.loads(body_bytes.decode("utf-8") or "{}")
            except Exception:
                # Not JSON? Return original bytes
                return Response(
                    content=body_bytes,
                    status_code=response.status_code,
                    headers={k: v for k, v in response.headers.items() if k.lower() != "content-length"},
                    media_type=response.media_type,
                )

            # If top-level 'message' is missing but 'result' exists, mirror it.
            if "message" not in payload and "result" in payload:
                payload["message"] = payload["result"]

            # Rebuild JSON response, preserving headers (except content-length)
            headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
            return _JSONResponse(
                content=payload,
                status_code=response.status_code,
                headers=headers,
            )
        except Exception:
            # On any failure, return the original response as-is
            return response

    return response
# ---------------------  End Compatibility middleware (additive)  ---------------------


# =============================================================================
# DEVELOPMENT SERVER LAUNCHER
# =============================================================================

if __name__ == "__main__":
    # This block is for local development only.
    # In production, use a process manager like Gunicorn with Uvicorn workers
    # to run the 'app' object. Example:
    # gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.a2a_universal.server:app
    import uvicorn

    log.info("Starting server in development mode...")
    uvicorn.run(
        "src.a2a_universal.server:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=True,
    )
