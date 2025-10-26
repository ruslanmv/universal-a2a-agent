# src/a2a_universal/adapters/crewai_knowledge_tool.py
from __future__ import annotations

import os
import requests
from crewai import Tool


class A2AKnowledgeTool(Tool):
    """CrewAI tool to query the Universal A2A /knowledge API (shared index)."""

    def __init__(self, base_url: str | None = None, timeout: int = 60):
        super().__init__(
            name="a2a_knowledge", description="Query the shared A2A knowledge index."
        )
        self.base = base_url or os.getenv("A2A_BASE", "http://localhost:8000")
        self.timeout = timeout

    def _run(self, query: str, k: int = 8, score_threshold: float = 0.65) -> str:
        r = requests.post(
            f"{self.base}/knowledge/query",
            json={"q": query, "k": k, "score_threshold": score_threshold},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.text
