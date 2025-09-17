# SPDX-License-Identifier: Apache-2.0
"""
LangGraph node that calls the Universal A2A service.

Goals (prod-ready):
- Works across LangGraph versions (typed state fallback if MessagesState moved).
- Safe defaults, overridable via environment variables.
- Clear logging + small retry for transient HTTP hiccups.
- No multi-input surprises; only reads configuration from env/constructor.
- Optional offline fallback for tests/CI (disabled by default for prod).

ENV (optional):
- A2A_BASE_URL          : Base URL for the A2A server (default: http://localhost:8000)
- A2A_USE_JSONRPC       : "true"/"false" to use /rpc JSON-RPC route (default: false -> /a2a)
- A2A_NODE_TIMEOUT_SEC  : Per-request timeout hint (not passed to client; used for logging) (default: 30)
- A2A_NODE_RETRIES      : Number of quick retries on failure (default: 2)
- A2A_NODE_DEBUG        : "true" to enable DEBUG logs for this module (default: false)
- A2A_OFFLINE_FALLBACK  : "true" to return a local echo on HTTP errors (default: false)

Usage (example):
    from typing_extensions import Annotated, TypedDict
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages, AnyMessage
    from langchain_core.messages import HumanMessage
    from a2a_universal.adapters.langgraph_agent import A2AAgentNode

    class GraphState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    g = StateGraph(GraphState)
    g.add_node("a2a", A2AAgentNode())  # reads A2A_BASE_URL from env if not given
    g.add_edge("__start__", "a2a")
    g.add_edge("a2a", END)
    app = g.compile()
    out = app.invoke({"messages": [HumanMessage(content="Ping the agent")]})
    print(out["messages"][-1].content)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.messages import AIMessage, BaseMessage

# --- LangGraph state compatibility ------------------------------------------------
# Prefer built-in MessagesState when available; otherwise define a typed state
# compatible with modern LangGraph (Annotated[list[AnyMessage], add_messages]).
try:
    # Older/newer versions may export this here
    from langgraph.graph import MessagesState  # type: ignore
except Exception:  # pragma: no cover - only used on certain versions
    from typing_extensions import Annotated, TypedDict
    from langgraph.graph.message import add_messages, AnyMessage

    class MessagesState(TypedDict):  # type: ignore[no-redef]
        messages: Annotated[List[AnyMessage], add_messages]

from ..client import A2AClient

__all__ = ["A2AAgentNode", "MessagesState"]

# --- Logging ---------------------------------------------------------------------

_LOG = logging.getLogger(__name__)
if os.getenv("A2A_NODE_DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}:
    _LOG.setLevel(logging.DEBUG)


# --- Env helpers -----------------------------------------------------------------

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


def _base_url(default: str = "http://localhost:8000") -> str:
    return os.getenv("A2A_BASE_URL", default).rstrip("/")


# --- Message helpers --------------------------------------------------------------

def _last_user_text(messages: List[Any] | List[BaseMessage]) -> str:
    """
    Extract the most relevant user text from LangChain/BaseMessage or dict-style messages.
    Prefers the last HumanMessage; falls back to the last non-empty text content.
    """
    # Walk backwards: prefer explicit human messages; then any text content.
    for m in reversed(messages or []):
        try:
            # LangChain BaseMessage path
            if hasattr(m, "type") and getattr(m, "type") == "human":
                content = getattr(m, "content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()
            # Dict-like fallback (OpenAI style)
            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    if isinstance(content, list):
                        # [{"type":"text","text":"..."}] etc.
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                                return str(part["text"]).strip()
        except Exception:
            continue
    # If nothing found, best-effort: stringify last element
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "") if hasattr(last, "content") else last
        return str(content) if content else ""
    return ""


# --- Node ------------------------------------------------------------------------

class A2AAgentNode:
    """LangGraph node that wraps the Universal A2A service.

    By default, reads base URL and behavior flags from environment variables.
    See module docstring for the full list of recognized env vars.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        use_jsonrpc: Optional[bool] = None,
        timeout_sec: Optional[float] = None,
        retries: Optional[int] = None,
        offline_fallback: Optional[bool] = None,
    ) -> None:
        # Configuration (env-driven with sane defaults)
        self.base_url = (base_url or _base_url())
        self.use_jsonrpc = (use_jsonrpc if use_jsonrpc is not None else _bool_env("A2A_USE_JSONRPC", False))
        self.timeout_sec = (timeout_sec if timeout_sec is not None else _float_env("A2A_NODE_TIMEOUT_SEC", 30.0))
        self.retries = (retries if retries is not None else _int_env("A2A_NODE_RETRIES", 2))
        self.offline_fallback = (offline_fallback
                                 if offline_fallback is not None
                                 else _bool_env("A2A_OFFLINE_FALLBACK", False))

        # Client: keep constructor minimal to avoid coupling to client kwargs that might not exist.
        self.client = A2AClient(self.base_url)

        _LOG.info(
            "A2AAgentNode initialized base_url=%s jsonrpc=%s retries=%d timeout=%.1fs offline_fallback=%s",
            self.base_url, self.use_jsonrpc, self.retries, self.timeout_sec, self.offline_fallback,
        )

    def __call__(self, state: MessagesState) -> Dict[str, Any]:
        """LangGraph node entry point: consumes state, returns a dict with new messages."""
        messages = state.get("messages", [])  # type: ignore[assignment]
        user_text = _last_user_text(messages)

        # Guard empty prompts (avoid sending empty requests downstream)
        if not isinstance(user_text, str) or not user_text.strip():
            _LOG.debug("No user text found in state; returning empty AI message.")
            return {"messages": [AIMessage(content="")]}

        # Simple retry loop around the A2A call
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                if _LOG.isEnabledFor(logging.DEBUG):
                    _LOG.debug(
                        "A2A request start: route=%s attempt=%d/%d",
                        "JSON-RPC" if self.use_jsonrpc else "A2A",
                        attempt + 1,
                        self.retries + 1,
                    )
                t0 = time.perf_counter()
                reply = self.client.send(user_text, use_jsonrpc=self.use_jsonrpc)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if _LOG.isEnabledFor(logging.DEBUG):
                    _LOG.debug("A2A response OK in %.1f ms", dt_ms)

                # If server returned a framework error string, log and return a concise hint.
                if isinstance(reply, str) and reply.startswith("[crewai error]"):
                    _LOG.warning(
                        "A2A server reported CrewAI error; ensure AGENT_FRAMEWORK=native on the server."
                    )
                    concise = "Server is running CrewAI framework; set AGENT_FRAMEWORK=native on the A2A server."
                    return {"messages": [AIMessage(content=concise)]}

                # Defensive: ensure a string; client should already do this.
                text = reply if isinstance(reply, str) else (str(reply) if reply is not None else "")
                return {"messages": [AIMessage(content=text)]}
            except httpx.HTTPError as e:
                last_exc = e
                _LOG.warning("A2A HTTP error (%s) on attempt %d/%d", e.__class__.__name__, attempt + 1, self.retries + 1)
            except Exception as e:  # noqa: BLE001
                last_exc = e
                _LOG.warning("A2A call failed (%s) on attempt %d/%d", e.__class__.__name__, attempt + 1, self.retries + 1)

            # Backoff before retrying
            if attempt < self.retries:
                backoff_ms = 100 * (attempt + 1)
                time.sleep(backoff_ms / 1000.0)

        # All attempts failed
        if self.offline_fallback:
            _LOG.error("A2A call failed after %d attempts; returning offline fallback. Last error: %r",
                       self.retries + 1, last_exc)
            msg = f"Hello (offline), you said: {user_text}" if user_text else "Hello, World!"
            return {"messages": [AIMessage(content=msg)]}

        # Propagate a clear error message into the graph (no exception to keep graph running)
        err = f"A2A call failed after {self.retries + 1} attempt(s): {type(last_exc).__name__}: {last_exc}"
        _LOG.error(err)
        return {"messages": [AIMessage(content=err)]}
