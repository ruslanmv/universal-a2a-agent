# src/a2a_universal/config.py
from __future__ import annotations

from typing import Any, List, Optional, Literal
from pydantic import Field, AliasChoices, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


def _parse_bool(value: Any) -> bool:
    """
    Robust bool parser:
    - handles actual bools
    - strips inline comments like 'false   # note'
    - accepts common truthy/falsey tokens
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.split("#", 1)[0].strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    # Fallback: python truthiness
    return bool(value)


def _parse_list(value: Any) -> List[str]:
    """
    Accept:
      - native list (already parsed)
      - JSON list string: '["*","https://a.com"]'
      - CSV string: '*,https://a.com'
    Returns a list of trimmed strings; empty -> []
    """
    if isinstance(value, list):
        return [str(x).strip() for x in value]
    if isinstance(value, str):
        raw = value.split("#", 1)[0].strip()
        if not raw:
            return []
        # Try JSON array first
        if (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("(") and raw.endswith(")")):
            try:
                data = json.loads(raw.replace("(", "[").replace(")", "]"))
                if isinstance(data, list):
                    return [str(x).strip() for x in data]
            except Exception:
                # fall through to CSV
                pass
        # CSV fallback
        return [s.strip() for s in raw.split(",") if s.strip()]
    # Fallback to string representation in a single-item list
    return [str(value).strip()]


def _normalize_auth_scheme(value: Any) -> str:
    """
    Normalize to NONE | BEARER | API_KEY; unknown -> NONE
    """
    if not isinstance(value, str):
        return "NONE"
    v = value.split("#", 1)[0].strip().upper()
    if v in {"NONE", "BEARER", "API_KEY"}:
        return v
    return "NONE"


class Settings(BaseSettings):
    """
    Pydantic v2 settings with:
    - lowercase field names (idiomatic)
    - env var aliases so UPPERCASE (legacy) works too
    - robust parsers for booleans and lists
    - uppercase properties for backward compatibility
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # env names not case-sensitive
        extra="ignore",         # ignore unknown env keys
    )

    # ------------------------------------------------------------------
    # Identity & protocol
    # ------------------------------------------------------------------
    agent_name: str = Field(
        default="Universal A2A Hello",
        validation_alias=AliasChoices("AGENT_NAME", "agent_name"),
    )
    agent_description: str = Field(
        default="Greets the user and echoes their message.",
        validation_alias=AliasChoices("AGENT_DESCRIPTION", "agent_description"),
    )
    agent_version: str = Field(
        default="1.2.0",
        validation_alias=AliasChoices("AGENT_VERSION", "agent_version"),
    )
    protocol_version: str = Field(
        default="0.3.0",
        validation_alias=AliasChoices("PROTOCOL_VERSION", "protocol_version"),
    )

    # ------------------------------------------------------------------
    # Network / URLs
    # ------------------------------------------------------------------
    a2a_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("A2A_HOST", "a2a_host"),
    )
    a2a_port: int = Field(
        default=8000,
        validation_alias=AliasChoices("A2A_PORT", "a2a_port"),
    )
    # Keep this as str to avoid strict URL validation breaking on 'http://localhost'
    public_url: Optional[str] = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("PUBLIC_URL", "public_url"),
    )

    # ------------------------------------------------------------------
    # Provider & Framework selection
    # ------------------------------------------------------------------
    llm_provider: str = Field(
        default="echo",
        validation_alias=AliasChoices("LLM_PROVIDER", "llm_provider"),
    )
    agent_framework: str = Field(
        default="langgraph",
        validation_alias=AliasChoices("AGENT_FRAMEWORK", "agent_framework"),
    )

    # ------------------------------------------------------------------
    # CORS (strings or lists; '*' means allow all)
    # ------------------------------------------------------------------
    cors_allow_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        validation_alias=AliasChoices("CORS_ALLOW_ORIGINS", "cors_allow_origins"),
    )
    cors_allow_methods: List[str] = Field(
        default_factory=lambda: ["*"],
        validation_alias=AliasChoices("CORS_ALLOW_METHODS", "cors_allow_methods"),
    )
    cors_allow_headers: List[str] = Field(
        default_factory=lambda: ["*"],
        validation_alias=AliasChoices("CORS_ALLOW_HEADERS", "cors_allow_headers"),
    )
    cors_allow_credentials: bool = Field(
        default=False,
        validation_alias=AliasChoices("CORS_ALLOW_CREDENTIALS", "cors_allow_credentials"),
    )

    # ------------------------------------------------------------------
    # Private adapter (enterprise)
    # ------------------------------------------------------------------
    private_adapter_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("PRIVATE_ADAPTER_ENABLED", "private_adapter_enabled"),
    )
    private_adapter_auth_scheme: Literal["NONE", "BEARER", "API_KEY"] = Field(
        default="NONE",
        validation_alias=AliasChoices("PRIVATE_ADAPTER_AUTH_SCHEME", "private_adapter_auth_scheme"),
    )
    private_adapter_auth_token: str = Field(
        default="",
        validation_alias=AliasChoices("PRIVATE_ADAPTER_AUTH_TOKEN", "private_adapter_auth_token"),
    )
    private_adapter_input_key: str = Field(
        default="input",
        validation_alias=AliasChoices("PRIVATE_ADAPTER_INPUT_KEY", "private_adapter_input_key"),
    )
    private_adapter_output_key: str = Field(
        default="output",
        validation_alias=AliasChoices("PRIVATE_ADAPTER_OUTPUT_KEY", "private_adapter_output_key"),
    )
    private_adapter_trace_key: str = Field(
        default="traceId",
        validation_alias=AliasChoices("PRIVATE_ADAPTER_TRACE_KEY", "private_adapter_trace_key"),
    )
    private_adapter_path: str = Field(
        default="/enterprise/v1/agent",
        validation_alias=AliasChoices("PRIVATE_ADAPTER_PATH", "private_adapter_path"),
    )

    # -------------------------
    # Validators (robust input)
    # -------------------------
    @field_validator(
        "cors_allow_origins",
        "cors_allow_methods",
        "cors_allow_headers",
        mode="before",
    )
    @classmethod
    def _val_lists(cls, v: Any) -> List[str]:
        return _parse_list(v)

    @field_validator("cors_allow_credentials", mode="before")
    @classmethod
    def _val_cors_bool(cls, v: Any) -> bool:
        return _parse_bool(v)

    @field_validator("private_adapter_enabled", mode="before")
    @classmethod
    def _val_priv_enabled(cls, v: Any) -> bool:
        return _parse_bool(v)

    @field_validator("private_adapter_auth_scheme", mode="before")
    @classmethod
    def _val_auth_scheme(cls, v: Any) -> str:
        return _normalize_auth_scheme(v)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Backward-compatible UPPERCASE properties
    # ------------------------------------------------------------------
    @property
    def AGENT_NAME(self) -> str: return self.agent_name

    @property
    def AGENT_DESCRIPTION(self) -> str: return self.agent_description

    @property
    def AGENT_VERSION(self) -> str: return self.agent_version

    @property
    def PROTOCOL_VERSION(self) -> str: return self.protocol_version

    @property
    def A2A_HOST(self) -> str: return self.a2a_host

    @property
    def A2A_PORT(self) -> int: return self.a2a_port

    @property
    def PUBLIC_URL(self) -> Optional[str]: return self.public_url

    @property
    def LLM_PROVIDER(self) -> str: return self.llm_provider

    @property
    def AGENT_FRAMEWORK(self) -> str: return self.agent_framework

    @property
    def CORS_ALLOW_ORIGINS(self) -> List[str]: return self.cors_allow_origins

    @property
    def CORS_ALLOW_METHODS(self) -> List[str]: return self.cors_allow_methods

    @property
    def CORS_ALLOW_HEADERS(self) -> List[str]: return self.cors_allow_headers

    @property
    def CORS_ALLOW_CREDENTIALS(self) -> bool: return self.cors_allow_credentials

    @property
    def PRIVATE_ADAPTER_ENABLED(self) -> bool: return self.private_adapter_enabled

    @property
    def PRIVATE_ADAPTER_AUTH_SCHEME(self) -> str: return self.private_adapter_auth_scheme

    @property
    def PRIVATE_ADAPTER_AUTH_TOKEN(self) -> str: return self.private_adapter_auth_token

    @property
    def PRIVATE_ADAPTER_INPUT_KEY(self) -> str: return self.private_adapter_input_key

    @property
    def PRIVATE_ADAPTER_OUTPUT_KEY(self) -> str: return self.private_adapter_output_key

    @property
    def PRIVATE_ADAPTER_TRACE_KEY(self) -> str: return self.private_adapter_trace_key

    @property
    def PRIVATE_ADAPTER_PATH(self) -> str: return self.private_adapter_path


# Singleton settings instance
settings = Settings()
