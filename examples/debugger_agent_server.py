from __future__ import annotations
import os
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()


from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

from a2a_universal import mount, run
from a2a_universal.provider_api import langgraph_llm

# 1) Build a tiny "Debugger" graph
llm = langgraph_llm()  # uses LLM_PROVIDER & credentials from env


async def debug_node(state):
    user_text = state["messages"][-1].content
    prompt = (
        "You are a senior Python debugging assistant. "
        "Explain the root cause of the error, then show corrected code.\n\n"
        f"{user_text}"
    )
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=resp.content)]}


sg = StateGraph(MessagesState)
sg.add_node("debugger", debug_node)
sg.add_edge("__start__", "debugger")
sg.add_edge("debugger", END)
graph = sg.compile()


# 2) Expose a text handler for universal A2A
async def handle_text(text: str) -> str:
    out = await graph.ainvoke({"messages": [HumanMessage(content=text)]})
    return out["messages"][-1].content


# 3) Build A2A app (agent-card, A2A, RPC, OpenAI, health)
app = mount(
    handler=handle_text,
    name=os.getenv("AGENT_NAME", "Debugger Agent"),
    description="Explains Python stack traces and suggests fixes.",
)

# 4) Run it
if __name__ == "__main__":
    run(
        "__main__:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=True
    )
