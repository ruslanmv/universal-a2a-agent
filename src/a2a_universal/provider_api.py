# src/a2a_universal/provider_api.py
"""
Convenience API and framework adapters built on top of the core provider registry.

This module keeps `providers.py` generic and adds a high-level, ergonomic surface:

- provider()       -> cached active ProviderBase instance
- provider_id()    -> active provider id string (after aliasing)
- framework_id()   -> selected agent framework (after aliasing)
- llm()            -> general LLM factory (auto from env AGENT_FRAMEWORK/LLM_PROVIDER)
- crew_llm()       -> CrewAI LLM wired to the active provider (via LiteLLM)
- langchain_llm()  -> LangChain ChatModel for the active provider
- langgraph_llm()  -> Alias of langchain_llm (LangGraph uses LangChain models)
- autogen_llm()    -> pyautogen config dict for the active provider
- native_llm()     -> Small `.call()` adapter backed by ProviderBase
- orchestrate_llm()-> placeholder for IBM watsonx Orchestrate (raises with guidance)

Design goals:
- Multi-provider support with one-liners.
- Multi-framework orchestration via env (`AGENT_FRAMEWORK`).
- Lazy imports for optional dependencies.
- Clear, actionable errors (missing deps/creds).
- Easy to extend (add providers/frameworks in one place).

Environment variables (common):
- LLM_PROVIDER: watsonx|openai|anthropic|gemini|ollama|azure|bedrock|...
- AGENT_FRAMEWORK: crewai|langchain|langgraph|autogen|native|watsonx_orchestrate|...

Provider-specific env keys are read in the respective adapter sections.
"""

from __future__ import annotations

import os
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, Optional

# Import internals from the core registry (read-only)
from .providers import (  # type: ignore
    ProviderBase,
    NotReadyProvider,
    build_provider,
    _REGISTRY,   # internal registry; safe to use within package
    _ALIASES,    # alias map; safe to use within package
)

__all__ = [
    # base helpers
    "provider",
    "provider_id",
    "framework_id",
    # general entry point
    "llm",
    # framework-specific factories
    "crew_llm",
    "langchain_llm",
    "langgraph_llm",
    "autogen_llm",
    "native_llm",
    "orchestrate_llm",
    # compatibility alias
    "make_crewai_llm_from_providers",
]

# -----------------------------------------------------------------------------
# Provider convenience
# -----------------------------------------------------------------------------

_PROVIDER_SINGLETON: Optional[ProviderBase] = None
_PROVIDER_LOCK = RLock()


def provider(name: Optional[str] = None, *, fresh: bool = False) -> ProviderBase:
    """
    Return the active provider instance.

    - If `name` is provided, resolve it via aliases/registry.
    - If `name` is None, use env var LLM_PROVIDER (same behavior as build_provider()).
    - Returns a cached singleton unless `fresh=True` is specified.
    """
    global _PROVIDER_SINGLETON
    with _PROVIDER_LOCK:
        if not fresh and _PROVIDER_SINGLETON is not None and name is None:
            return _PROVIDER_SINGLETON

        if name is None:
            p = build_provider()
        else:
            want = _ALIASES.get(name.lower().strip(), name.lower().strip())
            factory = _REGISTRY.get(want)
            if factory is not None:
                p = factory()
            else:
                # Fallbacks mirror build_provider()
                if "echo" in _REGISTRY:
                    p = _REGISTRY["echo"]()
                elif _REGISTRY:
                    p = next(iter(_REGISTRY.values()))()
                else:
                    p = NotReadyProvider("unknown", reason="No providers discovered")

        if name is None and not fresh:
            _PROVIDER_SINGLETON = p
        return p


def provider_id(default: str = "echo") -> str:
    """Resolve the active provider id string (after aliasing), or `default`."""
    want = (os.getenv("LLM_PROVIDER", default) or default).lower().strip()
    return _ALIASES.get(want, want)


# -----------------------------------------------------------------------------
# Framework normalization
# -----------------------------------------------------------------------------

_FRAMEWORK_ALIASES: Dict[str, str] = {
    "crewai": "crewai",
    "crew": "crewai",
    "crew.ai": "crewai",
    "langgraph": "langgraph",
    "lg": "langgraph",
    "langchain": "langchain",
    "lc": "langchain",
    "autogen": "autogen",
    "native": "native",
    "direct": "native",
    "watsonx_orchestrate": "watsonx_orchestrate",
    "orchestrate": "watsonx_orchestrate",
}


def framework_id(default: str = "crewai") -> str:
    """Return normalized framework id from env AGENT_FRAMEWORK (or default)."""
    raw = (os.getenv("AGENT_FRAMEWORK", default) or default).lower().strip()
    return _FRAMEWORK_ALIASES.get(raw, raw)


# -----------------------------------------------------------------------------
# General LLM factory (auto from env)
# -----------------------------------------------------------------------------

def llm():
    """
    General, framework-agnostic LLM factory chosen automatically via env:

        LLM_PROVIDER=watsonx|openai|anthropic|gemini|ollama|azure|bedrock|...
        AGENT_FRAMEWORK=crewai|langchain|langgraph|autogen|native|...

    Returns:
        - CrewAI LLM (if framework=crewai)
        - LangChain ChatModel (if framework=langchain or langgraph)
        - dict for pyautogen config (if framework=autogen)
        - Small adapter exposing `.call()` (if framework=native)
        - Raises NotImplementedError for watsonx_orchestrate (with guidance)
    """
    fw = framework_id()

    if fw == "crewai":
        return crew_llm()

    if fw in ("langchain", "langgraph"):
        return langchain_llm()  # langgraph uses LC models

    if fw == "autogen":
        return autogen_llm()

    if fw == "native":
        return native_llm()

    if fw == "watsonx_orchestrate":
        return orchestrate_llm()

    # Fallback: native adapter
    return native_llm()


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _require_env(env: Dict[str, Optional[str]], keys: list[str]) -> None:
    missing = [k for k in keys if not env.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def _active_provider_ready() -> ProviderBase:
    p = provider()  # cached by default
    pid = (getattr(p, "id", None) or "").lower()
    if not pid:
        raise RuntimeError("No provider id resolved from provider().")
    if not getattr(p, "ready", False):
        raise RuntimeError(f"Provider '{pid}' not ready: {getattr(p, 'reason', 'unknown')}")
    return p


def _sync_watsonx_env_aliases() -> None:
    """
    LiteLLM expects WATSONX_APIKEY (no underscore). Support .env using WATSONX_API_KEY.

    .env example:
        WATSONX_API_KEY=...
        WATSONX_URL=https://us-south.ml.cloud.ibm.com
        WATSONX_PROJECT_ID=...

    This will automatically set WATSONX_APIKEY for LiteLLM/CrewAI.
    """
    if not os.getenv("WATSONX_APIKEY"):
        alt = os.getenv("WATSONX_API_KEY")
        if alt:
            os.environ["WATSONX_APIKEY"] = alt


# -----------------------------------------------------------------------------
# CrewAI adapter (via LiteLLM model strings)
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # typing only; avoid runtime import
    from crewai import LLM as _CrewLLM


def crew_llm() -> "_CrewLLM":
    """
    Return a CrewAI LLM configured for the active provider.
    Uses LiteLLM model strings and lazy-imports CrewAI.
    """
    try:
        from crewai import LLM as CrewLLM  # lazy import
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CrewAI is not installed. Install with `pip install crewai` "
            "or `pip install universal-a2a-agent[crewai]`."
        ) from e

    p = _active_provider_ready()
    pid = (getattr(p, "id", None) or "").lower()

    # IBM watsonx.ai (LiteLLM needs env; pass ONLY the model string)
    if pid == "watsonx":
        _sync_watsonx_env_aliases()
        env = {
            "WATSONX_APIKEY": os.getenv("WATSONX_APIKEY"),
            "WATSONX_URL": os.getenv("WATSONX_URL"),
            "WATSONX_PROJECT_ID": os.getenv("WATSONX_PROJECT_ID"),
        }
        _require_env(env, ["WATSONX_APIKEY", "WATSONX_URL", "WATSONX_PROJECT_ID"])
        model_id = os.getenv("MODEL_ID", "ibm/granite-3-3-8b-instruct")
        # IMPORTANT: ONLY pass the litellm model string to CrewLLM;
        # credentials come from environment variables.
        return CrewLLM(
            model=f"watsonx/{model_id}",
            temperature=0.0,
            max_tokens=768,
        )

    # OpenAI
    if pid == "openai":
        env = {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
        _require_env(env, ["OPENAI_API_KEY"])
        model_id = os.getenv("MODEL_ID", "gpt-4o-mini")
        return CrewLLM(model=f"openai/{model_id}", temperature=0.0, max_tokens=768)

    # Azure OpenAI
    if pid in ("azure_openai", "azure-openai", "azure"):
        env = {
            # LiteLLM commonly uses these names; align your .env accordingly
            "AZURE_API_KEY": os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
            "AZURE_API_BASE": os.getenv("AZURE_API_BASE") or os.getenv("AZURE_OPENAI_API_BASE"),
            "AZURE_API_VERSION": os.getenv("AZURE_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION"),
            "AZURE_DEPLOYMENT": os.getenv("AZURE_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        }
        _require_env(env, ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_DEPLOYMENT"])
        model = env["AZURE_DEPLOYMENT"]
        return CrewLLM(model=f"azure/{model}", temperature=0.0, max_tokens=768)

    # Anthropic / Claude
    if pid in ("anthropic", "claude"):
        env = {"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")}
        _require_env(env, ["ANTHROPIC_API_KEY"])
        model_id = os.getenv("MODEL_ID", "claude-3-5-sonnet-latest")
        return CrewLLM(model=f"anthropic/{model_id}", temperature=0.0, max_tokens=768)

    # Google Gemini
    if pid in ("gemini", "google"):
        env = {"GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")}
        _require_env(env, ["GEMINI_API_KEY"])
        model_id = os.getenv("MODEL_ID", "gemini-1.5-pro")
        return CrewLLM(model=f"gemini/{model_id}", temperature=0.0, max_tokens=768)

    # Ollama (local)
    if pid == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        os.environ.setdefault("OLLAMA_BASE_URL", base)
        model_id = os.getenv("MODEL_ID", "llama3.1")
        return CrewLLM(model=f"ollama/{model_id}", temperature=0.0, max_tokens=768)

    # AWS Bedrock (LiteLLM supports bedrock/<model>)
    if pid == "bedrock":
        # LiteLLM reads AWS creds from env; ensure region at least
        env = {"AWS_REGION": os.getenv("AWS_REGION")}
        _require_env(env, ["AWS_REGION"])
        model_id = os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
        return CrewLLM(model=f"bedrock/{model_id}", temperature=0.0, max_tokens=768)

    raise RuntimeError(f"Provider '{pid}' is not mapped to a CrewAI/LiteLLM config.")


# Backward-compat alias
make_crewai_llm_from_providers = crew_llm


# -----------------------------------------------------------------------------
# LangChain / LangGraph adapter (returns a ChatModel)
# -----------------------------------------------------------------------------

def langchain_llm():
    """
    Return a LangChain ChatModel configured for the active provider.
    Lazy-imports each provider's LC package to avoid hard deps.
    """
    p = _active_provider_ready()
    pid = (getattr(p, "id", None) or "").lower()

    # IBM watsonx.ai
    if pid == "watsonx":
        try:
            from langchain_ibm import ChatWatsonx  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Install `langchain-ibm` for watsonx integration: `pip install langchain-ibm`."
            ) from e
        _sync_watsonx_env_aliases()
        env = {
            "WATSONX_APIKEY": os.getenv("WATSONX_APIKEY"),
            "WATSONX_URL": os.getenv("WATSONX_URL"),
            "WATSONX_PROJECT_ID": os.getenv("WATSONX_PROJECT_ID"),
        }
        _require_env(env, ["WATSONX_APIKEY", "WATSONX_URL", "WATSONX_PROJECT_ID"])
        model_id = os.getenv("MODEL_ID", "ibm/granite-3-3-8b-instruct")
        return ChatWatsonx(
            model_id=model_id,
            project_id=env["WATSONX_PROJECT_ID"],
            base_url=env["WATSONX_URL"],
            apikey=env["WATSONX_APIKEY"],
            temperature=0.0,
        )

    # OpenAI
    if pid == "openai":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install `langchain-openai`: `pip install langchain-openai`.") from e
        env = {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
        _require_env(env, ["OPENAI_API_KEY"])
        model_id = os.getenv("MODEL_ID", "gpt-4o-mini")
        return ChatOpenAI(api_key=env["OPENAI_API_KEY"], model=model_id, temperature=0.0)

    # Azure OpenAI
    if pid in ("azure_openai", "azure-openai", "azure"):
        try:
            from langchain_openai import AzureChatOpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install `langchain-openai`: `pip install langchain-openai`.") from e
        env = {
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY"),
            "AZURE_OPENAI_API_BASE": os.getenv("AZURE_OPENAI_API_BASE") or os.getenv("AZURE_API_BASE"),
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_API_VERSION", "2024-05-01-preview"),
            "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_DEPLOYMENT"),
        }
        _require_env(env, ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_BASE", "AZURE_OPENAI_DEPLOYMENT_NAME"])
        return AzureChatOpenAI(
            openai_api_key=env["AZURE_OPENAI_API_KEY"],
            azure_endpoint=env["AZURE_OPENAI_API_BASE"],
            api_version=env["AZURE_OPENAI_API_VERSION"],
            deployment_name=env["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=0.0,
        )

    # Anthropic / Claude
    if pid in ("anthropic", "claude"):
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install `langchain-anthropic`: `pip install langchain-anthropic`.") from e
        env = {"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")}
        _require_env(env, ["ANTHROPIC_API_KEY"])
        model_id = os.getenv("MODEL_ID", "claude-3-5-sonnet-latest")
        return ChatAnthropic(api_key=env["ANTHROPIC_API_KEY"], model=model_id, temperature=0.0)

    # Google Gemini
    if pid in ("gemini", "google"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install `langchain-google-genai`: `pip install langchain-google-genai`.") from e
        env = {"GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")}
        _require_env(env, ["GEMINI_API_KEY"])
        model_id = os.getenv("MODEL_ID", "gemini-1.5-pro")
        return ChatGoogleGenerativeAI(google_api_key=env["GEMINI_API_KEY"], model=model_id, temperature=0.0)

    # Ollama (local)
    if pid == "ollama":
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception:
            try:
                from langchain_community.chat_models import ChatOllama  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Install `langchain-ollama` or `langchain-community` for Ollama."
                ) from e
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_id = os.getenv("MODEL_ID", "llama3.1")
        try:
            return ChatOllama(model=model_id, base_url=base, temperature=0.0)
        except TypeError:
            return ChatOllama(model=model_id, temperature=0.0)

    # AWS Bedrock
    if pid == "bedrock":
        try:
            from langchain_aws import ChatBedrock  # type: ignore
            use_aws = True
        except Exception:
            try:
                from langchain_community.chat_models import BedrockChat as ChatBedrock  # type: ignore
                use_aws = False
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Install `langchain-aws` or `langchain-community` for Bedrock."
                ) from e
        region = os.getenv("AWS_REGION")
        _require_env({"AWS_REGION": region}, ["AWS_REGION"])
        model_id = os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
        if use_aws:
            return ChatBedrock(model_id=model_id, region_name=region, temperature=0.0)
        return ChatBedrock(model_id=model_id, temperature=0.0)

    raise RuntimeError(f"Provider '{pid}' is not mapped to a LangChain config.")


def langgraph_llm():
    """
    Return a ChatModel for LangGraph.
    LangGraph uses LangChain chat models, so we reuse langchain_llm().
    """
    return langchain_llm()


# -----------------------------------------------------------------------------
# AutoGen adapter (returns config dict)
# -----------------------------------------------------------------------------

def autogen_llm() -> Dict[str, Any]:
    """
    Return a config dict suitable for `pyautogen` agent initialization.
    """
    p = _active_provider_ready()
    pid = (getattr(p, "id", None) or "").lower()

    cfg: Dict[str, Any] = {"temperature": 0.0, "max_tokens": 768}

    if pid == "openai":
        env = {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
        _require_env(env, ["OPENAI_API_KEY"])
        model_id = os.getenv("MODEL_ID", "gpt-4o-mini")
        cfg["config_list"] = [{
            "provider": "openai",
            "model": model_id,
            "api_key": env["OPENAI_API_KEY"],
        }]
        return cfg

    if pid in ("azure_openai", "azure-openai", "azure"):
        env = {
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY"),
            "AZURE_OPENAI_API_BASE": os.getenv("AZURE_OPENAI_API_BASE") or os.getenv("AZURE_API_BASE"),
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_API_VERSION", "2024-05-01-preview"),
            "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_DEPLOYMENT"),
        }
        _require_env(env, ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_BASE", "AZURE_OPENAI_DEPLOYMENT_NAME"])
        cfg["config_list"] = [{
            "provider": "azure_openai",
            "model": env["AZURE_OPENAI_DEPLOYMENT_NAME"],
            "api_key": env["AZURE_OPENAI_API_KEY"],
            "api_base": env["AZURE_OPENAI_API_BASE"],
            "api_version": env["AZURE_OPENAI_API_VERSION"],
        }]
        return cfg

    if pid in ("gemini", "google"):
        env = {"GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")}
        _require_env(env, ["GEMINI_API_KEY"])
        model_id = os.getenv("MODEL_ID", "gemini-1.5-pro")
        cfg["config_list"] = [{
            "provider": "gemini",
            "model": model_id,
            "api_key": env["GEMINI_API_KEY"],
        }]
        return cfg

    if pid == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_id = os.getenv("MODEL_ID", "llama3.1")
        cfg["config_list"] = [{
            "provider": "ollama",
            "model": model_id,
            "base_url": base,
        }]
        return cfg

    if pid in ("anthropic", "claude"):
        env = {"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")}
        _require_env(env, ["ANTHROPIC_API_KEY"])
        model_id = os.getenv("MODEL_ID", "claude-3-5-sonnet-latest")
        cfg["config_list"] = [{
            "provider": "anthropic",
            "model": model_id,
            "api_key": env["ANTHROPIC_API_KEY"],
        }]
        return cfg

    if pid == "bedrock":
        region = os.getenv("AWS_REGION")
        _require_env({"AWS_REGION": region}, ["AWS_REGION"])
        model_id = os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
        cfg["config_list"] = [{
            "provider": "bedrock",
            "model": model_id,
            "region_name": region,
        }]
        return cfg

    if pid == "watsonx":
        raise RuntimeError(
            "AutoGen mapping for IBM watsonx is not supported in this template. "
            "Consider routing watsonx via an OpenAI-compatible gateway or use CrewAI/LangChain."
        )

    raise RuntimeError(f"Provider '{pid}' is not mapped to an AutoGen config.")


# -----------------------------------------------------------------------------
# Native adapter: small `.call()` wrapper over ProviderBase
# -----------------------------------------------------------------------------

class _NativeLLMAdapter:
    """Tiny adapter exposing `.call()` and delegating to ProviderBase.generate()."""

    def __init__(self, prov: ProviderBase) -> None:
        self._p = prov

    def __repr__(self) -> str:
        return f"<NativeLLMAdapter provider={getattr(self._p, 'id', 'unknown')}>"

    def call(self, messages: Any, **_: Any) -> str:
        if isinstance(messages, str):
            return self._p.generate(prompt=messages)
        return self._p.generate(messages=messages)


def native_llm() -> _NativeLLMAdapter:
    p = _active_provider_ready()
    return _NativeLLMAdapter(p)


# -----------------------------------------------------------------------------
# IBM watsonx Orchestrate (placeholder)
# -----------------------------------------------------------------------------

def orchestrate_llm():
    """
    Placeholder for IBM watsonx Orchestrate integration.
    """
    raise NotImplementedError(
        "AGENT_FRAMEWORK=watsonx_orchestrate is a placeholder. "
        "Provide your Orchestrate client + mapping and return a callable object."
    )
