from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from a2a_universal.adapters.langgraph_agent import A2AAgentNode
from typing import Any, List, Dict

def _as_text(msg: BaseMessage | Dict[str, Any]) -> str:
    """Robustly extract text from LangChain/BaseMessage or dict-like messages."""
    content = getattr(msg, "content", msg)
    # Simple string
    if isinstance(content, str):
        return content
    # OpenAI-style/structured content: list of parts
    if isinstance(content, list):
        buf: List[str] = []
        for part in content:
            if isinstance(part, str):
                buf.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text", "")
                if isinstance(t, str):
                    buf.append(t)
        return "\n".join([t for t in buf if t.strip()])
    # Fallback
    return str(content) if content is not None else ""

def run_debugger_graph():
    sg = StateGraph(MessagesState)

    # If your A2A server runs in Docker/WSL/VM, pick the right host/IP here
    node = A2AAgentNode(base_url="http://127.0.0.1:8001")
    sg.add_node("debugger", node)
    sg.add_edge("__start__", "debugger")
    sg.add_edge("debugger", END)

    app = sg.compile()

    res = app.invoke({
        "messages": [
            HumanMessage(content=(
                "Error: TypeError: unsupported operand type(s) for +: 'int' and 'str'\n\n"
                "Code:\nprint(5 + 'hello')"
            ))
        ]
    })

    # Always print something meaningful
    last = res["messages"][-1]
    text = _as_text(last)
    print(text if text.strip() else "[empty reply]")

if __name__ == "__main__":
    run_debugger_graph()
