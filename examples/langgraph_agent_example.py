from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage
from a2a_universal.adapters.langgraph_agent import A2AAgentNode

g = StateGraph(MessagesState)
g.add_node("a2a", A2AAgentNode(base_url="http://localhost:8000"))
g.add_edge("__start__", "a2a")
g.add_edge("a2a", END)
app = g.compile()

if __name__ == "__main__":
    out = app.invoke({"messages": [HumanMessage(content="ping from LangGraph")]})
    print(out["messages"][-1].content)
