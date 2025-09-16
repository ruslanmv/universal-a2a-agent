# src/a2a_universal/adapters/crewai_tool.py
"""
CrewAI-compatible wrapper for invoking Universal A2A Agent skills.

- Uses Pydantic model fields (no custom __init__) to avoid attribute errors.
- Provides both sync (_run) and async (_arun) execution paths.
- Declares an args_schema so CrewAI knows how to pass inputs.
- Includes sane defaults, timeouts, and robust error handling.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Type

import httpx
from crewai.tools import BaseTool

# Pydantic v2 first; fall back to v1 for broader compatibility.
try:  # Pydantic v2
    from pydantic import BaseModel, Field, model_validator

    _PYDANTIC_V2 = True
except Exception:  # Pydantic v1
    from pydantic import BaseModel, Field, root_validator  # type: ignore

    _PYDANTIC_V2 = False


__all__ = ["A2ATool", "A2AInput", "a2a_hello"]


# ----------------------------- Constants -------------------------------------

_DEFAULT_BASE_URL = os.getenv("A2A_BASE", "http://localhost:8000")
_DEFAULT_ENDPOINT_PATH = "/a2a"
_DEFAULT_TIMEOUT_SECONDS = 30.0


# ------------------------------ Schemas --------------------------------------

class A2AInput(BaseModel):
    """Input schema for the tool."""
    prompt: str = Field(..., description="User request or instruction for the A2A skill.")


# ------------------------------- Tool ----------------------------------------

class A2ATool(BaseTool):
    """
    Generic CrewAI-compatible tool that calls the Universal A2A Agent.

    This class allows you to wrap *any* A2A skill/intent as a CrewAI tool.
    Provide a `name` and `description` (from BaseTool), and (optionally) a `skill`
    that maps to the Universal A2A backend skill/intent identifier.
    """

    # Configuration fields (pydantic-managed)
    base_url: str = Field(
        default=_DEFAULT_BASE_URL,
        description="Base URL of the Universal A2A Agent service (e.g., 'http://localhost:8000').",
    )
    endpoint_path: str = Field(
        default=_DEFAULT_ENDPOINT_PATH,
        description="Relative HTTP path for the A2A message endpoint.",
    )
    request_timeout: float = Field(
        default=_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        description="HTTP request timeout in seconds.",
    )
    # Declare 'skill' as a Pydantic field so it can be set/validated
    skill: Optional[str] = Field(
        default=None,
        description="A2A skill/intent identifier. If omitted, defaults to the tool's name.",
    )

    # Let CrewAI know the expected input shape
    args_schema: Type[BaseModel] = A2AInput

    # ----------------------- Pydantic validators -----------------------------

    if _PYDANTIC_V2:
        @model_validator(mode="after")  # type: ignore[no-redef]
        def _default_skill(self) -> "A2ATool":
            """If 'skill' isn't provided, default it to the tool's name."""
            if not self.skill:
                self.skill = self.name
            # Normalize base_url and endpoint path
            self.base_url = self.base_url.rstrip("/")
            self.endpoint_path = "/" + self.endpoint_path.lstrip("/")
            return self
    else:  # Pydantic v1 compatibility
        @root_validator(pre=False)  # type: ignore[misc]
        def _default_skill_v1(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            skill = values.get("skill")
            name = values.get("name")
            if not skill and name:
                values["skill"] = name
            base_url = (values.get("base_url") or _DEFAULT_BASE_URL).rstrip("/")
            endpoint = "/" + (values.get("endpoint_path") or _DEFAULT_ENDPOINT_PATH).lstrip("/")
            values["base_url"] = base_url
            values["endpoint_path"] = endpoint
            return values

    # ------------------------ Internal helpers -------------------------------

    def _make_payload(self, prompt: str) -> Dict[str, Any]:
        """Construct the A2A message payload."""
        return {
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": f"{self.name}-tool",
                    "parts": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "data",
                            "mimeType": "application/vnd.a2a-skill",
                            "data": self.skill,
                        },
                    ],
                }
            },
        }

    def _build_url(self) -> str:
        """Join base_url and endpoint_path safely."""
        return f"{self.base_url}{self.endpoint_path}"

    @staticmethod
    def _extract_text_from_response(data: Dict[str, Any]) -> str:
        """
        Try to extract the most useful text from a few likely shapes.
        """
        # Preferred A2A message.parts[text]
        message = (data.get("message") or {})
        for part in message.get("parts", []):
            if part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text

        # Common fallbacks
        for key in ("result", "output", "text", "data"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val

        # OpenAI-like choices
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            # handle message.content or text fields
            msg = first.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
            text = first.get("text")
            if isinstance(text, str) and text.strip():
                return text

        return "[No text part in A2A response]"

    # ----------------------------- Execution ---------------------------------

    def _run(self, prompt: str) -> str:
        """
        Synchronous execution of the tool.

        CrewAI calls this when the tool is invoked in a non-async context.
        """
        payload = self._make_payload(prompt)
        try:
            response = httpx.post(
                self._build_url(),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=httpx.Timeout(self.request_timeout),
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            return f"[A2A call failed: timeout after {self.request_timeout:.1f}s: {exc}]"
        except httpx.HTTPStatusError as exc:
            # Include status code and a short response excerpt for diagnostics
            text_excerpt = (exc.response.text or "")[:512]
            return f"[A2A call failed: HTTP {exc.response.status_code}: {text_excerpt}]"
        except httpx.RequestError as exc:
            return f"[A2A call failed: network error: {exc}]"
        except Exception as exc:  # Defensive catch-all
            return f"[A2A call failed: unexpected error: {exc}]"

        # Parse structured response
        try:
            data = response.json()
        except ValueError:
            return "[A2A call failed: invalid JSON response]"

        return self._extract_text_from_response(data)

    async def _arun(self, prompt: str) -> str:
        """
        Async execution of the tool.

        CrewAI calls this when the tool is invoked in an async context.
        """
        payload = self._make_payload(prompt)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._build_url(),
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=httpx.Timeout(self.request_timeout),
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            return f"[A2A call failed: timeout after {self.request_timeout:.1f}s: {exc}]"
        except httpx.HTTPStatusError as exc:
            text_excerpt = (exc.response.text or "")[:512]
            return f"[A2A call failed: HTTP {exc.response.status_code}: {text_excerpt}]"
        except httpx.RequestError as exc:
            return f"[A2A call failed: network error: {exc}]"
        except Exception as exc:
            return f"[A2A call failed: unexpected error: {exc}]"

        try:
            data = response.json()
        except ValueError:
            return "[A2A call failed: invalid JSON response]"

        return self._extract_text_from_response(data)


# -------------------------------------------------------------------
# Example pre-built tool (hello-world)
# -------------------------------------------------------------------
a2a_hello = A2ATool(
    name="a2a_hello",
    description="Send a greeting to the Universal A2A Agent and return its reply.",
    skill="hello",
)
