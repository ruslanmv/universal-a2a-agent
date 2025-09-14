from __future__ import annotations
from typing import Any

from ..frameworks import FrameworkBase, _call_provider, _extract_last_user_text

class Framework(FrameworkBase):
    id = "crewai"
    name = "CrewAI Framework"

    def __init__(self, provider, **kwargs):
        super().__init__(provider)
        try:
            from crewai import Agent, Task, Crew  # type: ignore
            # Build a tiny two-step crew that ultimately calls the provider
            self.Agent = Agent
            self.Task = Task
            self.Crew = Crew
            self.ready = True
            self.reason = ""
        except Exception as e:
            # CrewAI not installed; we'll fallback in execute
            self.Agent = None
            self.Task = None
            self.Crew = None
            self.ready = True
            self.reason = f"CrewAI unavailable, fallback active: {e}"

    async def execute(self, messages: list[dict[str, Any]]) -> str:
        text = _extract_last_user_text(messages)
        # If CrewAI is available, execute a simple flow; otherwise call provider directly
        if self.Agent and self.Task and self.Crew:
            try:
                # Define a simple tool that uses our provider
                async def a2a_tool(query: str) -> str:
                    return await _call_provider(self.provider, query, messages)

                # CrewAI tools are usually sync; provide a sync adapter
                def a2a_tool_sync(query: str) -> str:
                    import asyncio
                    return asyncio.get_event_loop().run_until_complete(a2a_tool(query))

                researcher = self.Agent(role="Researcher", goal="Answer user input via A2A provider", backstory="", tools=[a2a_tool_sync])
                task = self.Task(description=text or "Say hello.", agent=researcher)
                result = self.Crew(agents=[researcher], tasks=[task]).kickoff()
                return str(result)
            except Exception as e:
                return f"[crewai error] {e}"
        # Fallback path
        return await _call_provider(self.provider, text, messages)
