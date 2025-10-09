# SPDX-License-Identifier: Apache-2.0
"""
Universal A2A Agent — Python package entry point.

What this gives you
-------------------
- A production-ready FastAPI service (see `a2a_universal.server:app`) that speaks:
  * POST /a2a           — Universal A2A envelope
  * POST /rpc           — JSON-RPC 2.0 wrapper
  * POST /openai/...    — OpenAI-compatible chat completions
  * GET  /healthz       — liveness
  * GET  /readyz        — readiness
  * GET  /.well-known/agent-card.json  — discovery metadata

- A thin HTTP client: `A2AClient` for calling any A2A service.
- A tiny “runner” API so *any* FastAPI/ASGI app can become A2A-enabled in seconds:
    from a2a_universal import run, compose, mount

Quick starts
------------
# 1) Run the Universal A2A server by itself (echo provider by default)
>>> import a2a_universal as a2a
>>> a2a.run(host="0.0.0.0", port=8000)

# 2) Attach A2A under your existing FastAPI app (your routes at '/', A2A at '/a2a')
>>> from fastapi import FastAPI
>>> app = FastAPI()
>>> import a2a_universal as a2a
>>> app = a2a.mount(app, prefix="/a2a")

# 3) Make A2A primary and keep your own app under '/app'
>>> a2a.run("__main__:app", mode="primary", user_prefix="/app", port=8080)

Set behavior via environment (see README for full list):
- LLM_PROVIDER=echo|watsonx|openai|anthropic|gemini|ollama|azure|bedrock|...
- AGENT_FRAMEWORK=langgraph|crewai|langchain|native
- Provider-specific credentials (e.g., WATSONX_API_KEY, OPENAI_API_KEY, ...)
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Version resolution (robust across normal and editable installs)
# ---------------------------------------------------------------------------

__version__: str

try:
    # Python 3.10+ stdlib
    from importlib.metadata import PackageNotFoundError, version as _pkg_version  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Backport for older environments
        from importlib_metadata import PackageNotFoundError, version as _pkg_version  # type: ignore
    except Exception:  # pragma: no cover
        PackageNotFoundError = Exception  # type: ignore[assignment]
        _pkg_version = None  # type: ignore[assignment]

if _pkg_version is not None:
    try:
        __version__ = _pkg_version("universal-a2a-agent")
    except PackageNotFoundError:  # pragma: no cover
        # Fallback for editable/unknown context
        __version__ = "0.0.0.dev0"
else:  # pragma: no cover
    __version__ = "0.0.0.dev0"


def get_version() -> str:
    """Return the installed package version string."""
    return __version__


# ---------------------------------------------------------------------------
# Public client API
# ---------------------------------------------------------------------------

from .client import A2AClient  # re-export


# ---------------------------------------------------------------------------
# Runner helpers (compose your app with the Universal A2A service)
# ---------------------------------------------------------------------------

# We keep imports lazy/friendly: if the runner module is missing for any reason,
# expose stubs that raise a clear, actionable error on use.
try:
    from .runner import run, compose, mount  # type: ignore
except Exception as _e:  # pragma: no cover
    _runner_import_error = _e

    def _runner_stub(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "The 'runner' helpers are unavailable. Ensure the package is installed correctly "
            "and its optional server dependencies (FastAPI/uvicorn) are present.\n"
            f"Underlying error: {_runner_import_error!r}"
        )

    run = _runner_stub            # type: ignore
    compose = _runner_stub        # type: ignore
    mount = _runner_stub          # type: ignore


__all__ = [
    # version
    "__version__", "get_version",
    # client
    "A2AClient",
    # runner helpers
    "run", "compose", "mount",
]
