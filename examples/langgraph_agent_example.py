# SPDX-License-Identifier: Apache-2.0
"""
Minimal LangGraph graph that calls the Universal A2A service via the built-in node.

Server requirement:
  Run the A2A server with AGENT_FRAMEWORK=native (provider can be watsonx or others).
Client (this script) can use LangGraph freely.

Env (optional):
  A2A_BASE_URL   : default http://localhost:8000
"""

from __future__ import annotations

import os
import json
import httpx
from typing_extensions import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.messages import HumanMessage

from a2a_universal.adapters.langgraph_agent import A2AAgentNode

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def _preflight_readyz(base: str) -> None:
    """Warn (don’t fail) if the server isn’t running with AGENT_FRAMEWORK=native."""
    try:
        r = httpx.get(f"{base.rstrip('/')}/readyz", timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            # best-effort sniffing; shape can vary by server version
            text = json.dumps(data).lower()
            if "crewai" in text and ("ready" in text or "framework" in text):
                print("[WARN] A2A server looks configured for CrewAI. "
                      "Run it with AGENT_FRAMEWORK=native for the LangGraph node.")
        else:
            print(f"[INFO] /readyz returned HTTP {r.status_code}; continuing.")
    except Exception:
        # Not fatal; keep going.
        pass

def main() -> None:
    load_dotenv()
    base_url = os.getenv("A2A_BASE_URL", "http://localhost:8000")
    _preflight_readyz(base_url)

    g = StateGraph(GraphState)
    g.add_node("a2a", A2AAgentNode(base_url=base_url))
    g.add_edge("__start__", "a2a")
    g.add_edge("a2a", END)
    app = g.compile()

    out = app.invoke({"messages": [HumanMessage(content="What is the capital of Italy?")]})
    # The node returns an AIMessage with the model's text
    print(out["messages"][-1].content)

if __name__ == "__main__":
    main()
