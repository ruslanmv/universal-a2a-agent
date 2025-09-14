from __future__ import annotations
import importlib
import pkgutil
import inspect
import os
import asyncio
from typing import Callable, Dict, Optional, Any, Awaitable

from .providers import ProviderBase, NotReadyProvider

# ===== Base contract =====
class FrameworkBase:
    """All framework plugins must accept a Provider and implement async execute()."""

    id: str = "base"
    name: str = "BaseFramework"
    ready: bool = False
    reason: str = "Not initialized"

    def __init__(self, provider: ProviderBase, **kwargs: Any) -> None:
        self.provider = provider
        self.ready = True
        self.reason = ""

    async def execute(self, messages: list[dict[str, Any]]) -> str:
        raise NotImplementedError


class NotReadyFramework(FrameworkBase):
    def __init__(self, provider: ProviderBase, framework_id: str, reason: str) -> None:
        super().__init__(provider)
        self.id = framework_id
        self.name = framework_id.capitalize()
        self.ready = False
        self.reason = reason

    async def execute(self, messages: list[dict[str, Any]]) -> str:
        # Simple fallback: call provider directly for a response
        text = _extract_last_user_text(messages)
        return await _call_provider(self.provider, text, messages)


# ===== Async provider shim =====
async def _call_provider(provider: ProviderBase, prompt: str, messages: list[dict[str, Any]]) -> str:
    """Call provider.generate asynchronously, offloading sync providers to a thread."""
    try:
        res = provider.generate(prompt, messages)  # may be sync or async
        if inspect.isawaitable(res):
            return await res  # type: ignore[no-any-return]
        # run sync generate in a thread to avoid blocking event loop
        return await asyncio.to_thread(provider.generate, prompt, messages)  # type: ignore[misc]
    except Exception as e:  # pragma: no cover
        return f"[framework/provider error] {e}"


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        role = (m or {}).get("role")
        content = (m or {}).get("content")
        if role == "user":
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        return p.get("text", "")
    return ""


# ===== Plugin discovery =====
PLUGIN_PACKAGE = "a2a_universal.framework_plugins"
Factory = Callable[[ProviderBase], FrameworkBase]


def _safe_factory_from_module(module_name: str, fallback_id: str) -> Callable[[ProviderBase], FrameworkBase]:
    def _stub(provider: ProviderBase) -> FrameworkBase:
        return NotReadyFramework(provider, fallback_id, reason="Module missing Provider/Factory")

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        def _err(provider: ProviderBase) -> FrameworkBase:
            return NotReadyFramework(provider, fallback_id, reason=f"Import error: {e}")
        return _err

    # Priority 1: get_framework(provider: ProviderBase) -> FrameworkBase
    if hasattr(mod, "get_framework") and callable(mod.get_framework):  # type: ignore[attr-defined]
        def _ok_fn(provider: ProviderBase) -> FrameworkBase:
            try:
                fw = mod.get_framework(provider)  # type: ignore[attr-defined]
                if isinstance(fw, FrameworkBase):
                    return fw
                return NotReadyFramework(provider, fallback_id, reason="get_framework() did not return FrameworkBase")
            except Exception as e:
                return NotReadyFramework(provider, fallback_id, reason=f"get_framework() failed: {e}")
        return _ok_fn

    # Priority 2: class Framework(FrameworkBase)
    if hasattr(mod, "Framework"):
        cls = getattr(mod, "Framework")
        if inspect.isclass(cls) and issubclass(cls, FrameworkBase):
            def _ok_cls(provider: ProviderBase) -> FrameworkBase:
                try:
                    return cls(provider)  # type: ignore[call-arg]
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
                    if isinstance(obj, FrameworkBase):
                        return obj
                    if inspect.isclass(obj) and issubclass(obj, FrameworkBase):
                        return obj(provider)  # type: ignore[call-arg]
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
    "langgraph": "langgraph",
    "crewai": "crewai",
}


def list_frameworks() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in _discover_builtin().keys():
        out[k] = "builtin"
    for k in _discover_entry_points().keys():
        out[k] = "entrypoint"
    return out


def build_framework(provider: ProviderBase) -> FrameworkBase:
    want = (os.getenv("AGENT_FRAMEWORK", "native") or "native").lower().strip()
    want = _ALIASES.get(want, want)
    factory = _REGISTRY.get(want)
    if factory:
        return factory(provider)
    # fallback
    if "native" in _REGISTRY:
        return _REGISTRY["native"](provider)
    # last resort pass-through
    return NotReadyFramework(provider, want or "unknown", reason="No frameworks discovered")
