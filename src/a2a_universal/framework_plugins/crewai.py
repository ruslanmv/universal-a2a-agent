from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from ..frameworks import FrameworkBase, _call_provider, _extract_last_user_text

# Optional: structured logging if your logging_config set it up
try:
    import structlog  # pragma: no cover
    _log = structlog.get_logger("a2a.framework.crewai")
except Exception:  # pragma: no cover
    _log = None


def _log_info(msg: str, **kwargs: Any) -> None:
    if _log:
        _log.info(msg, **kwargs)


def _to_text(result: Any) -> str:
    """Normalize various CrewAI result shapes to a non-empty string."""
    for attr in ("raw", "output", "final_output", "result", "completion"):
        v = getattr(result, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if isinstance(result, dict):
        for k in ("output", "final_output", "text", "content", "result"):
            v = result.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    s = str(result or "").strip()
    return s


class Framework(FrameworkBase):
    id = "crewai"
    name = "CrewAI Framework"

    def __init__(self, provider, **kwargs):
        super().__init__(provider)
        load_dotenv()
        try:
            from crewai import Agent, Task, Crew, LLM
            self.Agent = Agent
            self.Task = Task
            self.Crew = Crew
            self.LLM = LLM
            self.ready = True
            self.reason = ""
        except Exception as e:
            self.Agent = None
            self.Task = None
            self.Crew = None
            self.LLM = None
            self.ready = True
            self.reason = f"CrewAI unavailable, fallback active: {e}"
            _log_info("CrewAI import failed; will use provider fallback", error=str(e))

    def _make_llm(self):
        from ..provider_api import crew_llm
        return crew_llm()

    async def execute(self, messages: list[dict[str, Any]]) -> str:
        """
        Execute a minimal CrewAI flow asynchronously.
        """
        text = _extract_last_user_text(messages)
        if self.Agent and self.Task and self.Crew:
            try:
                llm_instance = self._make_llm()

                # --- FIX APPLIED ---
                # The agent should not have tools that call back to itself.
                # By providing an empty tools list, the agent will rely solely on its
                # LLM to generate the final answer, preventing the infinite loop.
                researcher = self.Agent(
                    role="Researcher",
                    goal="Provide a concise and accurate answer to the user's query.",
                    backstory="A helpful assistant that directly answers questions.",
                    tools=[],  # Empty list prevents self-referential tool calls
                    llm=llm_instance,
                    allow_delegation=False,
                )
                task = self.Task(
                    description=text or "Say hello.",
                    agent=researcher,
                    expected_output="A concise answer to the user's query."
                )
                crew = self.Crew(agents=[researcher], tasks=[task], verbose=False)

                # Run the synchronous kickoff method in a thread to avoid blocking.
                result = await asyncio.to_thread(crew.kickoff)

                _log_info("CrewAI kickoff completed",
                          result_type=type(result).__name__,
                          prompt_len=len(text or ""))
                out = _to_text(result)
                if not out:
                    out = "[crewai] Model returned empty output."
                _log_info("CrewAI finalize", output_len=len(out))
                return out
            except Exception as e:
                _log_info("CrewAI execution error", error=str(e))
                return f"[crewai error] {e}"

        _log_info("CrewAI unavailable, using provider fallback")
        return await _call_provider(self.provider, text, messages)

