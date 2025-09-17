# SPDX-License-Identifier: Apache-2.0
"""
LangChain tool(s) for calling the Universal A2A Agent.

Design goals (prod-ready):
- Single-input tool signature (compatible with Chat agents).
- Reads config from environment (no leaking params into the tool signature).
- Sensible defaults + overridable via env vars.
- Input validation, clear errors, minimal & safe logging.
- Small retry loop for transient network hiccups.
- One-time /readyz preflight with framework mismatch warnings.
- Backwards-compatible export: `a2a_hello`.

ENV VARS (all optional unless noted):
- A2A_BASE_URL         : Base URL of the A2A server (default: http://localhost:8000)
- A2A_USE_JSONRPC      : "true"/"false" to use /rpc JSON-RPC route (default: false -> /a2a route)
- A2A_TIMEOUT_SEC      : Request timeout in seconds (default: 30)
- A2A_RETRIES          : Number of quick retries on failure (default: 2)
- A2A_TOOL_DEBUG       : "true" to enable DEBUG logs for this module (default: false)
- A2A_TOOL_PREFLIGHT   : "true" to run a one-time /readyz preflight (default: true)

Usage:
    from a2a_universal.adapters.langchain_tool import a2a_hello
    agent = initialize_agent(tools=[a2a_hello], llm=llm, ...)

You can also create a custom-named tool:
    from a2a_universal.adapters.langchain_tool import make_a2a_tool
    mytool = make_a2a_tool(name="call_universal_agent", description="...")
"""

from __future__ import annotations

import json
import logging
import os
import time
from threading import Lock
from typing import Callable

import httpx
from langchain.tools import tool

from ..client import A2AClient

__all__ = ["a2a_hello", "a2a_call", "make_a2a_tool"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG = logging.getLogger(__name__)
if os.getenv("A2A_TOOL_DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}:
    # Don't clobber root handlers; just set level on this logger.
    _LOG.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y"}


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _base_url() -> str:
    return os.getenv("A2A_BASE_URL", "http://localhost:8000").rstrip("/")


# ---------------------------------------------------------------------------
# One-time /readyz preflight (warn if server uses CrewAI framework)
# ---------------------------------------------------------------------------

_PREFLIGHT_DONE = False
_PREFLIGHT_LOCK = Lock()


def _preflight_readyz(base_url: str, *, context: str = "LangChain tool") -> None:
    """
    Best-effort preflight to detect server framework mismatch.
    Warn (do not raise) if the server looks configured for CrewAI.
    """
    global _PREFLIGHT_DONE
    if _PREFLIGHT_DONE:
        return
    with _PREFLIGHT_LOCK:
        if _PREFLIGHT_DONE:
            return
        if not _bool_env("A2A_TOOL_PREFLIGHT", True):
            _PREFLIGHT_DONE = True
            return
        try:
            r = httpx.get(f"{base_url}/readyz", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                text = json.dumps(data).lower()
                if "crewai" in text:
                    # Match the phrasing used in the LangGraph node, tailored for LangChain.
                    _LOG.warning(
                        "[WARN] A2A server looks configured for CrewAI. "
                        "Run it with AGENT_FRAMEWORK=native for the %s.",
                        context,
                    )
            else:
                _LOG.debug("Preflight /readyz returned HTTP %s; continuing.", r.status_code)
        except Exception as e:  # noqa: BLE001
            _LOG.debug("Preflight /readyz skipped due to error: %r", e)
        finally:
            _PREFLIGHT_DONE = True


# ---------------------------------------------------------------------------
# Core callable (plain function you can also import & use directly)
# ---------------------------------------------------------------------------

def a2a_call(text: str) -> str:
    """
    Send a user message to the Universal A2A Agent and return its text reply.

    Parameters
    ----------
    text : str
        The message to send. Must be a non-empty string.

    Returns
    -------
    str
        The agent's reply text. If the agent returns no text part,
        a short diagnostic string is returned.

    Raises
    ------
    ValueError
        If `text` is empty or not a str.
    RuntimeError
        If all retry attempts fail due to transport or server errors.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("a2a_call(text): `text` must be a non-empty string.")

    base_url = _base_url()
    use_jsonrpc = _bool_env("A2A_USE_JSONRPC", False)
    timeout_sec = _float_env("A2A_TIMEOUT_SEC", 30.0)
    retries = _int_env("A2A_RETRIES", 2)

    # One-time preflight
    _preflight_readyz(base_url, context="LangChain tool")

    client = A2AClient(base_url, timeout=timeout_sec)  # `timeout` supported by client

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if _LOG.isEnabledFor(logging.DEBUG):
                _LOG.debug(
                    "A2A call start: base_url=%s route=%s timeout=%.1fs attempt=%d",
                    base_url,
                    "JSON-RPC" if use_jsonrpc else "A2A",
                    timeout_sec,
                    attempt + 1,
                )
            t0 = time.perf_counter()
            resp = client.send(text, use_jsonrpc=use_jsonrpc)
            dt = (time.perf_counter() - t0) * 1000.0
            if _LOG.isEnabledFor(logging.DEBUG):
                _LOG.debug("A2A call OK in %.1f ms", dt)

            # If server returned a framework error string, log and return a concise hint.
            if isinstance(resp, str) and resp.startswith("[crewai error]"):
                _LOG.warning("A2A server reported CrewAI error; ensure AGENT_FRAMEWORK=native on the server.")
                return "Server is running CrewAI framework; set AGENT_FRAMEWORK=native on the A2A server."

            # Defensive: ensure we return a string
            if isinstance(resp, str) and resp.strip():
                return resp
            # Client guarantees string, but keep a guard:
            return str(resp) if resp is not None else "[A2A: no text in response]"
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                backoff_ms = 100 * (attempt + 1)  # light linear backoff
                _LOG.warning(
                    "A2A call failed (attempt %d/%d): %s — retrying in %d ms",
                    attempt + 1,
                    retries + 1,
                    type(e).__name__,
                    backoff_ms,
                )
                time.sleep(backoff_ms / 1000.0)
                continue
            # Exhausted
            _LOG.error("A2A call failed after %d attempts: %s", retries + 1, repr(e))
            raise RuntimeError(f"A2A call failed: {type(e).__name__}: {e}") from e


# ---------------------------------------------------------------------------
# LangChain Tool(s)
#   - Single input param (text) to be compatible with Chat agents.
#   - `return_direct=False` so the output flows back through the agent.
# ---------------------------------------------------------------------------

@tool("a2a_hello", return_direct=False)
def a2a_hello(text: str) -> str:
    """
    Send free-form text to the Universal A2A Agent and return its reply.

    Input schema:
        text: str  # Required. The message to send to the agent.

    Configuration comes from environment variables; see module docstring.
    """
    return a2a_call(text)


def make_a2a_tool(
    name: str = "a2a_call",
    description: str | None = None,
) -> Callable[[str], str]:
    """
    Factory to create a single-input LangChain tool bound to `a2a_call`.

    Parameters
    ----------
    name : str
        Tool name exposed to the LLM.
    description : str | None
        Human-readable description. If None, a safe default is used.

    Returns
    -------
    Callable[[str], str]
        A LangChain-compatible tool function (decorated) that accepts one
        string argument and returns the agent's reply text.
    """
    if not description:
        description = (
            "Send a user message to the Universal A2A Agent and return the reply. "
            "Input: the exact text to send."
        )

    # We create a new function and decorate it with @tool at runtime.
    # This avoids multi-input signatures and keeps per-tool naming.
    def _impl(text: str) -> str:
        return a2a_call(text)

    _impl.__name__ = name  # helps with logs/debugging
    _impl.__doc__ = description
    return tool(name=name, return_direct=False)(_impl)
