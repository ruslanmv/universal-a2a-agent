# SPDX-License-Identifier: Apache-2.0
"""
Minimal BeeAI Framework client that calls the Universal A2A Agent via the agent card.

Server requirement:
  Run the A2A server with AGENT_FRAMEWORK=native.

ENV (optional):
- A2A_BASE_URL : Base URL of the A2A server (default: http://localhost:8000)

Usage:
  python examples/beeai_framework_agent.py
  # or
  A2A_BASE_URL=http://your-host:8000 python examples/beeai_framework_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from beeai_framework.backend import UserMessage  # type: ignore

from a2a_universal.adapters.beeai_agent import make_beeai_agent, preflight_readyz


async def main(prompt: Optional[str] = None) -> None:
    load_dotenv()
    base_url = os.getenv("A2A_BASE_URL", "http://localhost:8000").rstrip("/")

    # Best-effort preflight (warn if server is running CrewAI framework)
    preflight_readyz(base_url, context="BeeAI agent")

    agent = make_beeai_agent(base_url)

    text = prompt or "ping from BeeAI"
    try:
        result = await agent.run(UserMessage(text))
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if msg.startswith("[crewai error]"):
            print("A2A server reported CrewAI error; ensure AGENT_FRAMEWORK=native on the server.")
            print("Server is running CrewAI framework; set AGENT_FRAMEWORK=native on the A2A server.")
        raise

    # If the result itself is a CrewAI error string, surface the concise hint.
    if isinstance(result, str) and result.startswith("[crewai error]"):
        print("A2A server reported CrewAI error; ensure AGENT_FRAMEWORK=native on the server.")
        print("Server is running CrewAI framework; set AGENT_FRAMEWORK=native on the A2A server.")

    print(result)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(arg))
