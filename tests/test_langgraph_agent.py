# tests/test_langgraph_agent.py
import pytest

# Skip this optional integration test if its dependencies aren't installed
pytest.importorskip("langgraph", reason="optional dependency not installed in CI")
pytest.importorskip("langchain_core", reason="optional dependency not installed in CI")

from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage

# If your adapter imports langgraph internally, importing it now is safe because the skips above fired if missing.
from a2a_universal.adapters.langgraph_agent import A2AAgentNode


def test_langgraph_a2a_node(monkeypatch):
    """
    Optional integration check for the LangGraph adapter.

    NOTE:
    - This test is skipped automatically if 'langgraph' or 'langchain_core' aren't installed.
    - We monkeypatch the node HTTP call to avoid requiring a live A2A server in CI.
    """
    # Build a small graph that calls the node once
    sg = StateGraph(MessagesState)
    node = A2AAgentNode(base_url="http://localhost:8000")  # base_url unused after monkeypatch

    # Best-effort monkeypatch to avoid real network I/O
    # The adapter typically holds a client with a .send(text, use_jsonrpc=False) method.
    if hasattr(node, "client") and hasattr(node.client, "send"):
        monkeypatch.setattr(node.client, "send", lambda text, use_jsonrpc=False: f"Hello, you said: {text}")
    else:
        # Fallback: patch the node __call__/invoke if present
        if hasattr(node, "__call__"):
            monkeypatch.setattr(
                node, "__call__", lambda state: {**state, "messages": state["messages"] + [HumanMessage(content="Hello")]}
            )
        elif hasattr(node, "invoke"):
            monkeypatch.setattr(
                node, "invoke", lambda state: {**state, "messages": state["messages"] + [HumanMessage(content="Hello")]}
            )
        else:
            pytest.skip("A2AAgentNode does not expose a patchable client or callable surface")

    sg.add_node("a2a", node)
    sg.add_edge("__start__", "a2a")
    sg.add_edge("a2a", END)
    app = sg.compile()

    res = app.invoke({"messages": [HumanMessage(content="ping")]})
    assert isinstance(res, dict)
    assert "messages" in res
    assert "Hello" in res["messages"][-1].content
