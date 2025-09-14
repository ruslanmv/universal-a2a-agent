# examples/quickstart_langgraph_watsonx.py
import asyncio
import httpx
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

BASE = "http://localhost:8000"

async def a2a_send(text: str) -> str:
    payload = {
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": "lg-node",
                "parts": [{"type": "text", "text": text}],
            }
        },
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{BASE}/a2a", json=payload)
        r.raise_for_status()
        data = r.json()
        for p in (data.get("message") or {}).get("parts", []):
            if p.get("type") == "text":
                return p.get("text", "")
    return ""

async def a2a_node(state: dict) -> dict:
    last = state["messages"][-1]
    user_text = getattr(last, "content", "")
    reply = await a2a_send(user_text)
    return {"messages": [AIMessage(content=reply)]}

# Build a tiny graph: start -> a2a_node -> END
g = StateGraph(MessagesState)
g.add_node("a2a", a2a_node)
g.add_edge("__start__", "a2a")
g.add_edge("a2a", END)
app = g.compile()

async def main():
    out = await app.ainvoke({"messages": [HumanMessage(content="ping from LangGraph")]})
    print(out["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
