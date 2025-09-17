# SPDX-License-Identifier: Apache-2.0
"""
Framework loader/dispatcher for the Universal A2A Agent.

Production goals:
- Stable base contract (`FrameworkBase`) with a single async `execute()` method.
- Non-blocking provider calls (sync providers are offloaded to a thread).
- Safe, lazy plugin discovery (builtins + Python entry points).
- Clear readiness/error signaling (NotReadyFramework with actionable `reason`).
- Friendly aliasing for popular framework ids (native, langgraph, crewai, beeai).
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import asyncio
from typing import Callable, Dict, Optional, Any

from .providers import ProviderBase

# ===== Base contract ============================================================

class FrameworkBase:
    """All framework plugins must accept a Provider and implement async execute()."""

    id: str = "base"
    name: str = "BaseFramework"
    ready: bool = False
    reason: str = "Not initialized"

    def __init__(self, provider: ProviderBase, **_: Any) -> None:
        self.provider = provider
        self.ready = True
        self.reason = ""

    async def execute(self, messages: list[dict[str, Any]]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class NotReadyFramework(FrameworkBase):
    """Placeholder framework when discovery/initialization fails."""

    def __init__(self, provider: ProviderBase, framework_id: str, reason: str) -> None:
        super().__init__(provider)
        self.id = framework_id
        self.name = framework_id.capitalize()
        self.ready = False
        self.reason = reason

    async def execute(self, messages: list[dict[str, Any]]) -> str:
        # Pragmatic fallback: call provider directly.
        text = _extract_last_user_text(messages)
        return await _call_provider(self.provider, text, messages)


# ===== Async provider shim ======================================================

async def _call_provider(provider: ProviderBase, prompt: str, messages: list[dict[str, Any]]) -> str:
    """
    Call provider.generate asynchronously, offloading sync providers to a thread.

    IMPORTANT: Do not call .generate() twice. Detect coroutine-ness up-front and
    either await directly or offload the single call to a worker thread.
    """
    try:
        gen = getattr(provider, "generate", None)
        if gen is None or not callable(gen):
            return "[framework/provider error] provider has no callable 'generate'"

        # Bound methods can be inspected for coroutine signature safely.
        if inspect.iscoroutinefunction(gen):
            return await gen(prompt=prompt, messages=messages)  # type: ignore[misc]
        # Fallback: treat as sync, run once in a thread
        return await asyncio.to_thread(gen, prompt, messages)  # type: ignore[misc]
    except Exception as e:  # pragma: no cover
        return f"[framework/provider error] {e}"


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    """
    Best-effort extraction of the latest user text from a universal message array.
    Supports OpenAI-style {'role','content'} where content can be str or list parts.
    """
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        role = (m or {}).get("role")
        content = (m or {}).get("content")
        if role == "user":
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        txt = p.get("text", "")
                        if isinstance(txt, str) and txt.strip():
                            return txt
    return ""


# ===== Plugin discovery =========================================================

PLUGIN_PACKAGE = "a2a_universal.framework_plugins"
Factory = Callable[[ProviderBase], FrameworkBase]


def _safe_factory_from_module(module_name: str, fallback_id: str) -> Factory:
    """Wrap import/instantiation errors into a NotReadyFramework with clear reason."""
    def _stub(provider: ProviderBase) -> FrameworkBase:
        return NotReadyFramework(provider, fallback_id, reason="Module missing Framework/get_framework")

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        def _err(provider: ProviderBase) -> FrameworkBase:
            return NotReadyFramework(provider, fallback_id, reason=f"Import error: {e}")
        return _err

    # Priority 1: get_framework(provider: ProviderBase) -> FrameworkBase
    getf = getattr(mod, "get_framework", None)
    if callable(getf):
        def _ok_fn(provider: ProviderBase) -> FrameworkBase:
            try:
                fw = getf(provider)  # type: ignore[misc]
                if isinstance(fw, FrameworkBase):
                    return fw
                return NotReadyFramework(provider, fallback_id, reason="get_framework() did not return FrameworkBase")
            except Exception as e:
                return NotReadyFramework(provider, fallback_id, reason=f"get_framework() failed: {e}")
        return _ok_fn

    # Priority 2: class Framework(FrameworkBase)
    cls = getattr(mod, "Framework", None)
    if inspect.isclass(cls) and issubclass(cls, FrameworkBase):
        def _ok_cls(provider: ProviderBase) -> FrameworkBase:
            try:
                return cls(provider)  # type: ignore[misc]
            except Exception as e:
                return NotReadyFramework(provider, fallback_id, reason=f"Framework() init failed: {e}")
        return _ok_cls

    return _stub


def _discover_builtin() -> Dict[str, Factory]:
    registry: Dict[str, Factory] = {}
    try:
        pkg = importlib.import_module(PLUGIN_PACKAGE)
        prefix = pkg.__name__ + "."
        for _, name, ispkg in pkgutil.iter_modules(pkg.__path__, prefix):  # type: ignore[attr-defined]
            if ispkg:
                continue
            short = name.rsplit(".", 1)[-1]
            registry[short] = _safe_factory_from_module(name, short)
    except Exception:
        pass
    return registry


# Entry points for external frameworks
try:  # Python 3.12 style
    from importlib.metadata import entry_points as _eps  # type: ignore

    def _discover_entry_points() -> Dict[str, Factory]:
        out: Dict[str, Factory] = {}
        try:
            eps = _eps().get("a2a_universal.frameworks", [])  # type: ignore[call-arg]
        except Exception:
            try:
                eps = [ep for ep in _eps(group="a2a_universal.frameworks")]  # type: ignore[call-arg]
            except Exception:
                eps = []

        for ep in eps:
            fid = ep.name

            def _factory(provider: ProviderBase, ep=ep, fid=fid) -> FrameworkBase:
                try:
                    obj = ep.load()
                    # Object may be an instance, class, or factory.
                    if isinstance(obj, FrameworkBase):
                        return obj
                    if inspect.isclass(obj) and issubclass(obj, FrameworkBase):
                        return obj(provider)  # type: ignore[misc]
                    if callable(obj):
                        fw = obj(provider)
                        if isinstance(fw, FrameworkBase):
                            return fw
                    return NotReadyFramework(provider, fid, reason="entry point did not yield FrameworkBase")
                except Exception as e:
                    return NotReadyFramework(provider, fid, reason=f"entry point load error: {e}")

            out[fid] = _factory
        return out
except Exception:  # pragma: no cover
    def _discover_entry_points() -> Dict[str, Factory]:
        return {}


_REGISTRY: Dict[str, Factory] = {}
_REGISTRY.update(_discover_builtin())
_REGISTRY.update(_discover_entry_points())

_ALIASES: Dict[str, str] = {
    "native": "native",
    "direct": "native",
    "langgraph": "langgraph",
    "lg": "langgraph",
    "crewai": "crewai",
    "crew": "crewai",
    "beeai": "beeai",
    "bee.ai": "beeai",
    "beeai_framework": "beeai",
}


def list_frameworks() -> Dict[str, str]:
    """Return a map of discovered framework ids -> source ('builtin' or 'entrypoint')."""
    out: Dict[str, str] = {}
    for k in _discover_builtin().keys():
        out[k] = "builtin"
    for k in _discover_entry_points().keys():
        out[k] = "entrypoint"
    return out


def build_framework(provider: ProviderBase) -> FrameworkBase:
    """
    Build the framework selected by env `AGENT_FRAMEWORK` (defaults to 'native').
    Falls back to 'native' when the requested framework is unavailable.
    """
    want = (os.getenv("AGENT_FRAMEWORK", "native") or "native").lower().strip()
    want = _ALIASES.get(want, want)
    factory = _REGISTRY.get(want)
    if factory:
        return factory(provider)

    # Fallback chain
    if "native" in _REGISTRY:
        return _REGISTRY["native"](provider)
    return NotReadyFramework(provider, want or "unknown", reason="No frameworks discovered")
