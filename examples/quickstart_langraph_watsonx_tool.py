"""
Example: LangGraph Agent + Watsonx.ai + Universal A2A

- Watsonx.ai acts as the orchestrator (LLM for planning).
- Universal A2A Agent acts as an expert tool for data retrieval.
- This example demonstrates a multi-step workflow:
  1. The orchestrator calls the tool to get facts.
  2. The orchestrator synthesizes those facts into a creative output.
"""

import os
import httpx
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool
from langchain_ibm import ChatWatsonx

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()

BASE = os.getenv("A2A_BASE", "http://localhost:8000")

# -------------------------------------------------------------------
# Universal A2A tool wrapper
# -------------------------------------------------------------------
def a2a_call(prompt: str) -> str:
    """Send a user message to the Universal A2A /a2a endpoint."""
    payload = {
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": "lc-tool",
                "parts": [{"type": "text", "text": prompt}],
            }
        },
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{BASE}/a2a", json=payload)
            r.raise_for_status()
            data = r.json()
            for p in (data.get("message") or {}).get("parts", []):
                if p.get("type") == "text":
                    return p.get("text", "")
            return "[No text part in A2A response]"
    except httpx.HTTPError as e:
        return f"[A2A HTTP Error: {e}]"
    except Exception as e:
        return f"[A2A call failed: {e}]"

# -------------------------------------------------------------------
# Main: Watsonx Orchestrator + LangGraph Agent
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Required environment variables
    model_id = os.getenv("MODEL_ID", "ibm/granite-3-3-8b-instruct")
    project_id = os.environ.get("WATSONX_PROJECT_ID")
    url = os.environ.get("WATSONX_URL")
    api_key = os.environ.get("WATSONX_API_KEY")

    if not all([project_id, url, api_key]):
        raise RuntimeError(
            "Missing Watsonx credentials. Please set WATSONX_PROJECT_ID, WATSONX_URL, and WATSONX_API_KEY."
        )

    # Orchestrator LLM (Watsonx.ai, via LangChain integration)
    llm = ChatWatsonx(
        model_id=model_id,
        project_id=project_id,
        base_url=url,
        apikey=api_key,
        params={"decoding_method": "greedy", "max_new_tokens": 512}, # Increased tokens for a more complex task
        temperature=0.0,
    )

    # NEW: Define a more specialized tool for our workflow
    expert_tool = Tool(
        name="a2a_expert_agent",
        description="Ask the A2A expert agent a question about a specific topic to get detailed facts. Use this to gather information before creating a summary or answering the final question.",
        func=a2a_call,
    )

    # Memory is handled by a checkpointer in LangGraph
    checkpointer = MemorySaver()

    # Build Agent using the modern LangGraph prebuilt helper
    agent_executor = create_react_agent(
        model=llm,
        tools=[expert_tool],
        checkpointer=checkpointer
    )

    # Define a unique ID for the conversation thread to enable memory
    config = {"configurable": {"thread_id": "genoa-research-thread"}}

    # NEW: A more complex, multi-step query for the agent to solve
    query = "First, use the a2a_expert_agent tool to ask 'What are three key facts about Genoa, Italy?'. Then, based ONLY on the answer you receive, write a short, appealing tweet to encourage tourists to visit."

    print(f"Executing workflow for query: '{query}'")

    # Invoke the agent
    response = agent_executor.invoke(
        {"messages": [("human", query)]},
        config=config,
    )
    
    # The final answer is in the content of the last message in the state
    final_answer = response["messages"][-1].content
    print("\n[Final Answer]:\n", final_answer)