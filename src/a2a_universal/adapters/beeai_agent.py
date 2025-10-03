# SPDX-License-Identifier: Apache-2.0
"""
BeeAI framework adapter for the Universal A2A Agent.

Goals (prod-ready):
- Minimal, robust factory to create a BeeAI Agent bound to the A2A agent card.
- Safe defaults from environment, easy to override via parameters.
- One-time /readyz preflight with clear warnings if the server is not in `native` framework.
- Version-tolerant construction for BeeAI's A2AAgent (handles multiple constructor/classmethod shapes).
- Post-construction fixups when the agent_card is a dict (wrap as attribute-style object), and
  normalize camelCase -> snake_case keys (e.g., preferredTransport -> preferred_transport).
- Quiet, structured logging without leaking secrets.

Server requirement:
- Run the A2A server with AGENT_FRAMEWORK=native (provider can be watsonx/openai/etc.).

ENV (optional):
- A2A_BASE_URL        : Base URL for the A2A server (default: http://localhost:8000)
- BEEAI_AGENT_DEBUG   : "true" to enable DEBUG logs in this module (default: false)
- BEEAI_PREFLIGHT     : "true" to run a one-time /readyz preflight (default: true)

Usage:
    from a2a_universal.adapters.beeai_agent import make_beeai_agent
    agent = make_beeai_agent()  # or make_beeai_agent(base_url="http://your-host:8000")
    result = await agent.run(UserMessage("ping"))
"""

from __future__ import annotations

import json
import logging
import os
import re
from threading import Lock
from typing import Any, Optional
from types import SimpleNamespace

import httpx

try:
    from beeai_framework.adapters.a2a.agents.agent import A2AAgent as BeeA2AAgent  # type: ignore
    from beeai_framework.memory import UnconstrainedMemory  # type: ignore
except Exception as _e:  # pragma: no cover
    BeeA2AAgent = None  # type: ignore
    UnconstrainedMemory = None  # type: ignore
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

__all__ = ["make_beeai_agent", "preflight_readyz"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG = logging.getLogger(__name__)
if os.getenv("BEEAI_AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}:
    _LOG.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y"}


def _base_url(default: str = "http://localhost:8000") -> str:
    return os.getenv("A2A_BASE_URL", default).rstrip("/")


# ---------------------------------------------------------------------------
# One-time /readyz preflight (warn if server uses CrewAI framework)
# ---------------------------------------------------------------------------

_PREFLIGHT_DONE = False
_PREFLIGHT_LOCK = Lock()


def preflight_readyz(base_url: str, *, context: str = "BeeAI agent") -> None:
    """Best-effort preflight to detect server framework mismatch."""
    global _PREFLIGHT_DONE
    if _PREFLIGHT_DONE:
        return
    with _PREFLIGHT_LOCK:
        if _PREFLIGHT_DONE:
            return
        if not _bool_env("BEEAI_PREFLIGHT", True):
            _PREFLIGHT_DONE = True
            return
        try:
            r = httpx.get(f"{base_url}/readyz", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                text = json.dumps(data).lower()
                if "crewai" in text:
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
# Helpers
# ---------------------------------------------------------------------------

_CAMEL_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _to_snake(s: str) -> str:
    """Convert camelCase/PascalCase/kebab-case to snake_case."""
    if "-" in s or " " in s:
        s = s.replace("-", "_").replace(" ", "_")
    s = _CAMEL_SPLIT_RE.sub("_", s).lower()
    s = re.sub(r"__+", "_", s)
    return s


def _alias_keys(obj: Any) -> Any:
    """
    Recursively add snake_case aliases for dict keys that are camelCase/PascalCase/kebab-case.
    Keeps the original keys as well. For lists, processes each element.
    """
    if isinstance(obj, list):
        return [_alias_keys(x) for x in obj]
    if not isinstance(obj, dict):
        return obj

    out: dict[str, Any] = {}
    for k, v in obj.items():
        out[k] = _alias_keys(v)
        snake = _to_snake(k)
        if snake != k and snake not in out:
            out[snake] = _alias_keys(v)
    return out


def _fetch_agent_card(card_url: str) -> Optional[dict[str, Any]]:
    """Fetch and return the agent-card JSON, or None on failure."""
    try:
        resp = httpx.get(card_url, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return _alias_keys(data)
    except Exception as e:  # noqa: BLE001
        _LOG.warning("Failed to fetch agent card from %s: %r", card_url, e)
    return None


def _ensure_attr_aliases_ns(ns: SimpleNamespace) -> SimpleNamespace:
    """
    For a SimpleNamespace created from a dict possibly containing camelCase keys,
    add snake_case attribute aliases (non-destructively).
    """
    for k, v in vars(ns).copy().items():
        snake = _to_snake(k)
        if snake != k and not hasattr(ns, snake):
            setattr(ns, snake, v)
        # Recurse into nested dicts/namespaces/lists
        if isinstance(v, dict):
            v_alias = _alias_keys(v)
            setattr(ns, snake if snake != k else k, v_alias)
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, dict):
                    new_list.append(_alias_keys(item))
                else:
                    new_list.append(item)
            setattr(ns, k, new_list)
    return ns


def _wrap_card_obj(card_obj: Any) -> SimpleNamespace:
    """
    Convert a card object (possibly dict or SimpleNamespace) into a SimpleNamespace
    with snake_case aliases added for camelCase keys.
    """
    if isinstance(card_obj, SimpleNamespace):
        return _ensure_attr_aliases_ns(card_obj)
    if isinstance(card_obj, dict):
        card_obj = _alias_keys(card_obj)
        return SimpleNamespace(**card_obj)
    return SimpleNamespace(value=card_obj)


def _post_construct_fixups(agent: Any, card_url: str) -> Any:
    """
    Patch common edge-cases across BeeAI versions:

    - Ensure both `agent_card` (public) and `_agent_card` (private) exist
      and point to the same object.
    - If the card is a dict, wrap it with SimpleNamespace for attribute access and
      add snake_case aliases (e.g., preferredTransport -> preferred_transport).
    - Ensure `_url` is set to the card URL when missing.
    """
    try:
        # 1) Find an existing card object, regardless of attribute name
        card_obj = None
        if hasattr(agent, "_agent_card"):
            card_obj = getattr(agent, "_agent_card")
        elif hasattr(agent, "agent_card"):
            card_obj = getattr(agent, "agent_card")

        # 2) If missing entirely, fetch the card from the URL
        if card_obj is None:
            card_dict = _fetch_agent_card(card_url)
            if card_dict:
                card_obj = card_dict

        # 3) Wrap into SimpleNamespace + add aliases
        card_ns = _wrap_card_obj(card_obj) if card_obj is not None else None

        # 4) Ensure BOTH attributes exist and point to the same object
        if card_ns is not None:
            try:
                setattr(agent, "_agent_card", card_ns)
            except Exception as e:
                _LOG.debug("Could not set _agent_card: %r", e)
            try:
                setattr(agent, "agent_card", card_ns)
            except Exception as e:
                _LOG.debug("Could not set agent_card: %r", e)

        # 5) Ensure _url exists for logging/namespacing in BeeAI emitter
        if not getattr(agent, "_url", None):
            setattr(agent, "_url", card_url)

    except Exception as e:  # noqa: BLE001
        _LOG.debug("Post-construct fixups skipped: %r", e)

    return agent


def _construct_bee_agent(card_url: str) -> Any:
    """
    Try multiple construction paths to be compatible with various BeeAI versions.

    Priority:
    1) Classmethods: from_agent_card_url / from_url / from_card / from_card_url
    2) Constructor kwargs: agent_card_url / card_url / url / agent_url / a2a_url
    3) Constructor with preloaded card: agent_card=card
    """
    if BeeA2AAgent is None:  # type: ignore[truthy-function]
        raise RuntimeError(
            f"beeai_framework is not installed correctly: {_IMPORT_ERROR!r}"
        )

    memory = UnconstrainedMemory()  # type: ignore[call-arg]

    # --- 1) Classmethod variants -------------------------------------------------
    for meth_name in ("from_agent_card_url", "from_url", "from_card_url"):
        ctor = getattr(BeeA2AAgent, meth_name, None)
        if callable(ctor):
            try:
                try:
                    agent = ctor(card_url, memory=memory)  # type: ignore[call-arg]
                except TypeError:
                    agent = ctor(card_url)  # type: ignore[call-arg]
                return _post_construct_fixups(agent, card_url)
            except Exception as e:
                _LOG.debug("BeeAI A2AAgent.%s failed: %r", meth_name, e)

    # Some versions may expose .from_card expecting the JSON dict.
    ctor = getattr(BeeA2AAgent, "from_card", None)
    if callable(ctor):
        card = _fetch_agent_card(card_url)
        if card:
            try:
                try:
                    agent = ctor(card, memory=memory)  # type: ignore[call-arg]
                except TypeError:
                    agent = ctor(card)  # type: ignore[call-arg]
                return _post_construct_fixups(agent, card_url)
            except Exception as e:
                _LOG.debug("BeeAI A2AAgent.from_card failed: %r", e)

    # --- 2) Constructor kwargs (URL-style) --------------------------------------
    for kwargs in (
        {"agent_card_url": card_url, "memory": memory},
        {"card_url": card_url, "memory": memory},
        {"url": card_url, "memory": memory},
        {"agent_url": card_url, "memory": memory},
        {"a2a_url": card_url, "memory": memory},
        {"agent_card_url": card_url},
        {"card_url": card_url},
        {"url": card_url},
        {"agent_url": card_url},
        {"a2a_url": card_url},
    ):
        try:
            agent = BeeA2AAgent(**kwargs)  # type: ignore[call-arg]
            return _post_construct_fixups(agent, card_url)
        except TypeError as e:
            _LOG.debug("BeeAI A2AAgent(**%r) signature mismatch: %r", kwargs, e)
        except Exception as e:
            _LOG.debug("BeeAI A2AAgent(**%r) failed: %r", kwargs, e)

    # --- 3) Constructor with preloaded card dict --------------------------------
    card = _fetch_agent_card(card_url)
    if card:
        for kwargs in (
            {"agent_card": card, "memory": memory},
            {"agent_card": card},
        ):
            try:
                agent = BeeA2AAgent(**kwargs)  # type: ignore[call-arg]
                return _post_construct_fixups(agent, card_url)
            except TypeError as e:
                _LOG.debug("BeeAI A2AAgent(**%r) signature mismatch: %r", kwargs, e)
            except Exception as e:
                _LOG.debug("BeeAI A2AAgent(**%r) failed: %r", kwargs, e)

    raise RuntimeError(
        "Could not construct BeeAI A2AAgent with any known signatures. "
        "Please update the adapter or the beeai_framework version."
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_beeai_agent(base_url: str | None = None) -> Any:
    """
    Create a BeeAI agent wired to the Universal A2A agent card.

    Parameters
    ----------
    base_url : str | None
        Base URL of the A2A server. If None, reads from A2A_BASE_URL (defaults to localhost).

    Returns
    -------
    BeeAI A2AAgent-compatible instance
    """
    base = (base_url or _base_url())
    preflight_readyz(base, context="BeeAI agent")

    card_url = f"{base.rstrip('/')}/.well-known/agent-card.json"
    _LOG.info("Initializing BeeAI agent using agent card at %s", card_url)

    return _construct_bee_agent(card_url)
