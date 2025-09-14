from __future__ import annotations
import importlib
import pkgutil
import inspect
import os
from typing import Callable, Dict, Optional, Type, Any

try:
    # Python 3.10+: importlib.metadata is stdlib
    from importlib.metadata import entry_points, EntryPoint  # type: ignore
except Exception:  # pragma: no cover
    entry_points = None  # type: ignore
    EntryPoint = Any  # type: ignore


# ===== Base contract =====
class ProviderBase:
    """
    Base provider contract. Implementations should override:
      - id (short identifier, e.g. 'openai', 'watsonx', 'echo')
      - name (human-friendly)
      - ready (bool)
      - reason (why not ready)
      - supports_messages (True if accepts chat messages directly)
      - generate(prompt, messages) -> str   (messages is optional)

    Minimal implementations may ignore messages and use prompt only.
    """
    id: str = "base"
    name: str = "BaseProvider"
    ready: bool = False
    reason: str = "Not initialized"
    supports_messages: bool = True

    def generate(self, prompt: str = "", messages: Optional[list] = None) -> str:
        raise NotImplementedError


class NotReadyProvider(ProviderBase):
    """A stub provider returned when a plugin fails to load or init."""
    def __init__(self, provider_id: str, reason: str) -> None:
        self.id = provider_id
        self.name = provider_id.capitalize()
        self.ready = False
        self.reason = reason
        self.supports_messages = True

    def generate(self, prompt: str = "", messages: Optional[list] = None) -> str:
        base = (prompt or "").strip()
        prefix = f"[{self.id} not ready: {self.reason}] "
        return prefix + (f"You said: {base}" if base else "Hello, World!")


# ===== Plugin discovery =====
PLUGIN_PACKAGE = "a2a_universal.provider_plugins"
Factory = Callable[[], ProviderBase]


def _safe_factory_from_module(module_name: str, fallback_id: str) -> Factory:
    """
    Build a zero-arg factory for a provider module. If import fails or the module
    does not expose a usable provider, return a factory that makes a NotReadyProvider.
    """
    def _stub() -> ProviderBase:
        return NotReadyProvider(fallback_id, reason=(
            "Module did not expose a valid Provider class or get_provider function."
        ))

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        def _err() -> ProviderBase:
            return NotReadyProvider(fallback_id, reason=f"Import error: {e}")
        return _err

    # Prefer function get_provider()
    if hasattr(mod, "get_provider") and callable(mod.get_provider):  # type: ignore[attr-defined]
        def _ok_fn() -> ProviderBase:
            try:
                p = mod.get_provider()  # type: ignore[attr-defined]
                if isinstance(p, ProviderBase):
                    return p
                return NotReadyProvider(fallback_id, reason="get_provider() did not return ProviderBase")
            except Exception as e:
                return NotReadyProvider(fallback_id, reason=f"get_provider() failed: {e}")
        return _ok_fn

    # Fallback: class Provider(ProviderBase)
    if hasattr(mod, "Provider"):
        cls = getattr(mod, "Provider")
        if inspect.isclass(cls) and issubclass(cls, ProviderBase):
            def _ok_cls() -> ProviderBase:
                try:
                    return cls()  # type: ignore[call-arg]
                except Exception as e:
                    return NotReadyProvider(fallback_id, reason=f"Provider() init failed: {e}")
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
            short = name.rsplit(".", 1)[-1]   # e.g., 'openai', 'watsonx'
            registry[short] = _safe_factory_from_module(name, short)
    except Exception:
        # Carry on (entry points may still work)
        pass
    return registry


def _discover_entry_points() -> Dict[str, Factory]:
    registry: Dict[str, Factory] = {}
    if entry_points is None:
        return registry

    try:
        # Python ≥3.12 style
        eps = entry_points().get("a2a_universal.providers", [])  # type: ignore[call-arg]
    except Exception:
        # Python 3.10/3.11 style
        try:
            eps = [ep for ep in entry_points(group="a2a_universal.providers")]  # type: ignore[call-arg]
        except Exception:
            eps = []

    for ep in eps:
        pid = ep.name  # advertised provider id
        def _factory(ep=ep, pid=pid) -> ProviderBase:
            try:
                obj = ep.load()
                # Accept: instance, subclass, or zero-arg callable returning instance
                if isinstance(obj, ProviderBase):
                    return obj
                if inspect.isclass(obj) and issubclass(obj, ProviderBase):
                    return obj()  # type: ignore[call-arg]
                if callable(obj):
                    p = obj()
                    if isinstance(p, ProviderBase):
                        return p
                return NotReadyProvider(pid, reason="entry point did not yield ProviderBase")
            except Exception as e:
                return NotReadyProvider(pid, reason=f"entry point load error: {e}")
        registry[pid] = _factory
    return registry


# Cache registry at import time (fast, deterministic)
_REGISTRY: Dict[str, Factory] = {}
_REGISTRY.update(_discover_builtin())
_REGISTRY.update(_discover_entry_points())

# Aliases allow friendly names (e.g., "azure" -> "azure_openai")
_ALIASES: Dict[str, str] = {
    "echo": "echo",
    "openai": "openai",
    "azure": "azure_openai",
    "azure-openai": "azure_openai",
    "azure_openai": "azure_openai",
    "watsonx": "watsonx",
    "ollama": "ollama",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
    "bedrock": "bedrock",
}


def list_providers() -> Dict[str, str]:
    """
    Returns a dict of {provider_id: 'builtin'|'entrypoint'} for discoverability.
    """
    out: Dict[str, str] = {}
    # Builtins
    for k in _discover_builtin().keys():
        out[k] = "builtin"
    # Entry points
    for k in _discover_entry_points().keys():
        out[k] = "entrypoint"
    return out


def build_provider() -> ProviderBase:
    """
    Build the provider selected by env var LLM_PROVIDER.
    Falls back to 'echo' if unspecified or missing.
    """
    want = (os.getenv("LLM_PROVIDER", "echo") or "echo").lower().strip()
    want = _ALIASES.get(want, want)

    factory = _REGISTRY.get(want)
    if factory is not None:
        return factory()

    # Fallback: echo if available, else first available
    if "echo" in _REGISTRY:
        return _REGISTRY["echo"]()
    if _REGISTRY:
        any_factory = next(iter(_REGISTRY.values()))
        return any_factory()

    # Final fallback: hard NotReady
    return NotReadyProvider("unknown", reason="No providers discovered")
