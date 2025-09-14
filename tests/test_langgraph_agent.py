from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage
from a2a_universal.adapters.langgraph_agent import A2AAgentNode

def test_langgraph_a2a_node():
    sg = StateGraph(MessagesState)
    sg.add_node("a2a", A2AAgentNode(base_url="http://localhost:8000"))
    sg.add_edge("__start__", "a2a")
    sg.add_edge("a2a", END)
    app = sg.compile()
    res = app.invoke({"messages": [HumanMessage(content="ping")]})
    assert "Hello" in res["messages"][-1].content
