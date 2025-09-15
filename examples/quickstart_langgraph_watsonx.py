import asyncio
import httpx
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

BASE = "http://localhost:8000"

# -------------------------------------------------------------------
# A2A async call
# -------------------------------------------------------------------
async def a2a_send(text: str) -> str:
    """Send a message to Universal A2A and return its reply text."""
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
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{BASE}/a2a", json=payload)
            r.raise_for_status()
            data = r.json()  # FIX: httpx .json() is synchronous
            for p in (data.get("message") or {}).get("parts", []):
                if p.get("type") == "text":
                    return p.get("text", "")
        return "[No text part in A2A response]"
    except httpx.HTTPError as e:
        return f"[A2A HTTP Error: {e}]"
    except Exception as e:
        return f"[A2A call failed: {e}]"

# -------------------------------------------------------------------
# LangGraph node: forward message to A2A
# -------------------------------------------------------------------
async def a2a_node(state: MessagesState) -> MessagesState:
    last_message = state["messages"][-1]
    user_text = getattr(last_message, "content", "")
    reply_text = await a2a_send(user_text)
    return {"messages": [AIMessage(content=reply_text)]}

# -------------------------------------------------------------------
# Build LangGraph workflow
# -------------------------------------------------------------------
graph = StateGraph(MessagesState)
graph.add_node("a2a", a2a_node)
graph.set_entry_point("a2a")
app = graph.compile()

# -------------------------------------------------------------------
# Run example
# -------------------------------------------------------------------
async def main():
    result = await app.ainvoke({"messages": [HumanMessage(content="Tell me about Genova?")]})
    print("\n[Final Answer]:", result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
