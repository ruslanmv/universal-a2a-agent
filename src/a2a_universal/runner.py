from __future__ import annotations

import importlib
import json
import os
import sys
import warnings
from typing import Any, Optional, Union, Callable, Awaitable

from starlette.types import ASGIApp, Scope, Receive, Send

# Reuse the production FastAPI app from the package (providers, frameworks, card, health, etc.)
from .server import app as _universal_app


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _load_dotenv() -> None:
    """
    Best-effort load of a local .env file if python-dotenv is installed.
    Does nothing (silently) if not installed, keeping this optional.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    # Do not override explicit environment, just add from .env if missing.
    load_dotenv(override=False)


def _resolve_app(app_or_path: Optional[Union[str, ASGIApp]]) -> Optional[ASGIApp]:
    """
    Resolve:
      - ASGI app instance (FastAPI/Starlette or any ASGI callable)
      - "module:attr" string (e.g., "__main__:app", "myapp.web:app")
      - None (means: no user app provided)
    """
    if app_or_path is None:
        return None
    if not isinstance(app_or_path, str):
        return app_or_path

    mod_name, sep, obj_name = app_or_path.partition(":")
    if not mod_name or not obj_name or sep != ":":
        raise ValueError("app path must be 'module:attr', e.g., '__main__:app'")

    # Support __main__ (running inside the same file)
    if mod_name == "__main__":
        mod = sys.modules.get("__main__")
        if mod is None:
            mod = importlib.import_module("__main__")
    else:
        mod = importlib.import_module(mod_name)

    app_obj = getattr(mod, obj_name, None)
    if app_obj is None:
        raise AttributeError(f"Object {obj_name!r} not found in module {mod_name!r}")

    return app_obj  # type: ignore[return-value]


# ------------------------------------------------------------------------------
# RPC shim + A2A body normalizer
# ------------------------------------------------------------------------------

class _RpcShimAndNormalizer:
    """
    ASGI wrapper that:
      1) Adds friendly GET/HEAD/OPTIONS handling for /rpc so browsers/monitors don't see 405.
      2) Normalizes POST bodies for /rpc and /a2a to be tolerant of different text-part shapes:
         - {"text": "..."}                 -> {"type":"text","text":"..."}
         - {"kind": "text", "text": "..."} -> {"type":"text","text":"..."}
    This keeps the underlying app untouched but more interoperable.
    """

    def __init__(self, app: ASGIApp, *, rpc_path: str = "/rpc", a2a_path: str = "/a2a"):
        self.app = app
        self.rpc_path = rpc_path
        self.a2a_path = a2a_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET").upper()
        path = scope.get("path") or "/"

        # 1) Friendly GET/HEAD/OPTIONS for /rpc
        if path == self.rpc_path and method in ("GET", "HEAD", "OPTIONS"):
            if method == "GET":
                payload = {
                    "status": "ok",
                    "message": (
                        "This is a JSON-RPC 2.0 endpoint. "
                        "Use POST with body: "
                        '{"jsonrpc":"2.0","method":"message/send","params":{...},"id":"..."}'
                    ),
                    "methods": ["POST"],
                }
                body_bytes = json.dumps(payload).encode("utf-8")
                headers = [
                    (b"content-type", b"application/json"),
                    (b"cache-control", b"no-store"),
                    (b"allow", b"POST, OPTIONS"),
                ]
                await send({"type": "http.response.start", "status": 200, "headers": headers})
                await send({"type": "http.response.body", "body": body_bytes})
                return
            # HEAD/OPTIONS: no body, just Allow
            status = 204
            headers = [(b"allow", b"POST, OPTIONS"), (b"cache-control", b"no-store")]
            await send({"type": "http.response.start", "status": status, "headers": headers})
            await send({"type": "http.response.body", "body": b""})
            return

        # 2) Normalize POST bodies for /rpc and /a2a
        if method == "POST" and path in (self.rpc_path, self.a2a_path):
            body = b""
            more = True
            while more:
                message = await receive()
                if message["type"] != "http.request":
                    continue
                body += message.get("body", b"")
                more = message.get("more_body", False)

            new_body = body
            try:
                data = json.loads(body.decode("utf-8") or "{}")
                if isinstance(data, dict):
                    # JSON-RPC: params.message.parts
                    msg = None
                    if isinstance(data.get("params"), dict):
                        msg = data["params"].get("message")
                    # Canonical A2A: sometimes directly {"message": {...}}
                    if msg is None and isinstance(data.get("message"), dict):
                        msg = data["message"]

                    if isinstance(msg, dict):
                        parts = msg.get("parts")
                        if isinstance(parts, list):
                            for p in parts:
                                if isinstance(p, dict) and "text" in p:
                                    # Convert common variants to {"type":"text","text":"..."}
                                    if not p.get("type") and p.get("kind") == "text":
                                        p.pop("kind", None)
                                        p["type"] = "text"
                                    elif not p.get("type"):
                                        p["type"] = "text"
                new_body = json.dumps(data).encode("utf-8")
            except Exception:
                # If anything goes wrong, forward original body untouched.
                new_body = body

            async def new_receive() -> dict:
                return {"type": "http.request", "body": new_body, "more_body": False}

            await self.app(scope, new_receive, send)
            return

        # Otherwise, pass through
        await self.app(scope, receive, send)


def _wrap_with_rpc_shim(app: Optional[ASGIApp]) -> Optional[ASGIApp]:
    return _RpcShimAndNormalizer(app) if app is not None else None


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def compose(
    user_app: Optional[ASGIApp],
    *,
    mode: str = "attach",
    a2a_prefix: str = "/a2a",
    user_prefix: str = "/app",
    # NEW: optionally inject a custom A2A surface
    handler: Optional[Callable[[str], Union[str, Awaitable[str]]]] = None,
    a2a_app: Optional[ASGIApp] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
) -> ASGIApp:
    """
    Build a composite ASGI application with both Universal A2A + your app.

    Modes:
      - "attach" (default): Your app at '/', Universal A2A mounted under 'a2a_prefix' (default: /a2a)
      - "primary"         : Universal A2A at '/', your app mounted under 'user_prefix' (default: /app)
      - "solo"            : Only Universal A2A (ignore user app)

    Notes:
      * We DO NOT mutate your app. We return a composite FastAPI root and mount both.
      * FastAPI is required only to compose roots. Your own app can be Starlette or plain ASGI.
    """
    try:
        from fastapi import FastAPI
    except Exception as e:
        raise RuntimeError("FastAPI is required to compose apps.") from e

    # Normalize prefixes
    if not a2a_prefix.startswith("/"):
        a2a_prefix = "/" + a2a_prefix
    if not user_prefix.startswith("/"):
        user_prefix = "/" + user_prefix

    # Decide which A2A app to use:
    #  - explicit a2a_app
    #  - built from handler via app.build()
    #  - packaged universal app (default)
    selected_a2a = _universal_app
    if a2a_app is not None:
        selected_a2a = a2a_app
    elif handler is not None:
        from .app import build as _build  # lazy to avoid cycles
        selected_a2a = _build(
            handler=handler,
            name=name or os.getenv("AGENT_NAME", "Universal A2A Agent"),
            description=description or os.getenv("AGENT_DESCRIPTION", "A2A-compatible agent"),
            version=version or os.getenv("AGENT_VERSION", "0.1.0"),
        )

    # Solo = just the chosen A2A app
    if mode == "solo" or user_app is None:
        # Even in solo, add the RPC shim to silence GET /rpc 405 and normalize bodies.
        return _wrap_with_rpc_shim(selected_a2a)  # type: ignore[return-value]

    # Build a clean root that mounts both apps without altering either
    root = FastAPI(
        title="Universal A2A — Composite",
        # Avoid confusing duplicate docs when composing; your app's docs remain intact under its prefix
        docs_url=None, redoc_url=None, openapi_url=None,
    )

    # Wrap both sides with the shim
    user_app_wrapped = _wrap_with_rpc_shim(user_app)
    a2a_app_wrapped = _wrap_with_rpc_shim(selected_a2a)

    if mode == "primary":
        # Universal A2A at '/', your app under '/app' (or custom user_prefix)
        root.mount("/", a2a_app_wrapped)           # type: ignore[arg-type]
        root.mount(user_prefix, user_app_wrapped)  # type: ignore[arg-type]
        return root

    if mode == "attach":
        # Your app at '/', A2A under '/a2a' (or custom a2a_prefix)
        root.mount("/", user_app_wrapped)          # type: ignore[arg-type]
        root.mount(a2a_prefix, a2a_app_wrapped)    # type: ignore[arg-type]
        return root

    raise ValueError("mode must be one of: 'primary' | 'attach' | 'solo'")


def mount(
    app_or_handler: Any = None,
    *,
    prefix: str = "/a2a",
    # "function mode" (build a full app from a single handler)
    handler: Any = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
) -> Any:
    """
    Dual-purpose 'mount':

    1) Function mode (simple): build a full A2A app from a single handler(text)->str|awaitable[str]
         app = mount(handler=my_handler, name="X", description="Y")

       This delegates to `a2a_universal.app.build()` and returns a fully-formed FastAPI app
       exposing: /a2a, /rpc, /openai/v1/chat/completions, /healthz, /readyz, /.well-known/agent-card.json.

    2) Mount-under-prefix mode (legacy/advanced): mount the Universal A2A server under an existing app
         mount(existing_fastapi_app, prefix="/a2a")

       This mutates and returns the provided app.

    The presence of the `handler` keyword OR a first positional argument that is callable
    selects "function mode". Otherwise, we assume you passed an app to mount under `prefix`.
    """
    # Detect "function mode"
    effective_handler = handler if handler is not None else (
        app_or_handler if callable(app_or_handler) else None
    )
    if effective_handler is not None:
        from .app import build as _build  # lazy import to avoid cycles
        return _build(
            handler=effective_handler,
            name=name or os.getenv("AGENT_NAME", "Universal A2A Agent"),
            description=description or os.getenv("AGENT_DESCRIPTION", "A2A-compatible agent"),
            version=version or os.getenv("AGENT_VERSION", "0.1.0"),
        )

    # Otherwise we expect an ASGI app to mount under the given prefix
    if app_or_handler is None:
        raise TypeError(
            "mount() expected a FastAPI/Starlette app when used without 'handler='. "
            "Example: mount(app, prefix='/a2a') or mount(handler=my_handler, name='X')."
        )
    try:
        # Wrap the universal app to silence GET/HEAD/OPTIONS on /rpc when mounted.
        app_or_handler.mount(prefix, _wrap_with_rpc_shim(_universal_app))  # type: ignore[arg-type]
        return app_or_handler
    except Exception as e:
        raise RuntimeError(
            "App does not support .mount(prefix, app). Use compose() instead."
        ) from e


def run(
    app: Optional[Union[str, ASGIApp]] = None,
    *,
    mode: str = "attach",
    a2a_prefix: str = "/a2a",
    user_prefix: str = "/app",
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info",
    workers: Optional[int] = None,
    # NEW: same injection knobs as compose()
    handler: Optional[Callable[[str], Union[str, Awaitable[str]]]] = None,
    a2a_app: Optional[ASGIApp] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
) -> None:
    """
    Start an ASGI server with Universal A2A + your app (optional).

    Args:
        app: ASGI app instance OR "module:attr" string. None => run Universal A2A solo.
        mode:
            - "attach" : your app at '/', A2A under 'a2a_prefix' (default)
            - "primary": Universal A2A at '/', your app under 'user_prefix'
            - "solo"   : Universal A2A only
        a2a_prefix: mount point for A2A when mode="attach" (default: /a2a)
        user_prefix: mount point for your app when mode="primary" (default: /app)
        host, port, reload, log_level, workers: forwarded to uvicorn.run()
        handler: optional text handler -> builds a custom A2A app on the fly (via app.build)
        a2a_app: optional pre-built A2A FastAPI app to use instead of the packaged one
        name/description/version: metadata for the A2A app when building from handler

    Behavior:
        - Loads .env if available (without overriding explicit env).
        - Uses the production Universal A2A app from the package by default.
        - When `reload=True`:
            * Supported cleanly when mode='solo' AND you passed an import string (e.g. '__main__:app')
              and you did not inject a custom A2A via handler/a2a_app.
            * For composed modes ('attach'/'primary'), Uvicorn cannot reload an in-memory composite object.
              Prefer `reload=False` or run your own uvicorn command with import strings.
    """
    _load_dotenv()

    # Keep original param to decide how to pass to uvicorn for reload/workers
    raw_app = app

    user_app = _resolve_app(app)
    application = compose(
        user_app,
        mode=mode,
        a2a_prefix=a2a_prefix,
        user_prefix=user_prefix,
        handler=handler,
        a2a_app=a2a_app,
        name=name,
        description=description,
        version=version,
    )

    try:
        import uvicorn  # lazy import
    except Exception as e:
        raise RuntimeError(
            "uvicorn is required to run the server. Install with `pip install uvicorn`."
        ) from e

    # If we are in SOLO mode and caller provided import string, forward it to uvicorn
    # so reload/workers are fully supported without warnings. Skip if a custom A2A was injected.
    if mode == "solo" and isinstance(raw_app, str) and not (handler or a2a_app):
        uvicorn.run(
            raw_app,  # import string here!
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            workers=workers,
        )
        return

    # In composed modes, passing an object is fine; but uvicorn reload requires import strings.
    if reload and not (mode == "solo" and isinstance(raw_app, str) and not (handler or a2a_app)):
        warnings.warn(
            "reload=True is only fully supported when mode='solo' and you pass the app as an import string "
            "(and no custom A2A was injected). Continuing without reload support for a composed in-memory app.",
            RuntimeWarning,
            stacklevel=2,
        )

    uvicorn.run(
        application,
        host=host,
        port=port,
        reload=False if reload and not (mode == "solo" and isinstance(raw_app, str) and not (handler or a2a_app)) else reload,
        log_level=log_level,
        workers=workers,
    )
