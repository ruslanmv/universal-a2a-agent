"""
Example: LangChain Agent + Watsonx.ai + Universal A2A

- Watsonx.ai acts as the orchestrator (LLM for planning).
- Universal A2A Agent acts as the executor (tool backend).
- Demonstrates clean error handling, memory, and environment setup.
"""

import os
import httpx
from dotenv import load_dotenv

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
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
        r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
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
# Main: Watsonx Orchestrator + LangChain Agent
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
        params={"decoding_method": "greedy", "max_new_tokens": 256},
        temperature=0.0,
    )

    # Wrap Universal A2A as a LangChain Tool
    tool = Tool(
        name="a2a_hello",
        description="Send a prompt to the Universal A2A agent and return its reply.",
        func=a2a_call,
    )

    # Memory (required for chat-based agents)
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Build Agent (legacy LangChain agent API)
    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
    )

    # Run an example query
    query = "Use the a2a_hello tool to say hello to LangChain."
    response = agent.invoke({"input": query})
    print("\n[Final Answer]:", response["output"])


"""
Notes:
- You’ll see deprecation warnings because LangChain agents are legacy.
- Migration path: LangGraph (recommended for new apps).
  See: https://langchain-ai.github.io/langgraph/

But this script will continue to run correctly with LangChain.
"""
