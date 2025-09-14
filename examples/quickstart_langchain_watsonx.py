# examples/quickstart_langchain_watsonx.py
import httpx
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI  # any LC LLM is fine here

BASE = "http://localhost:8000"

def a2a_call(prompt: str) -> str:
    """Synchronous tool that posts to the Universal A2A /a2a endpoint."""
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
    r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    for p in (data.get("message") or {}).get("parts", []):
        if p.get("type") == "text":
            return p.get("text", "")
    return ""

tool = Tool(
    name="a2a_hello",
    description="Send a prompt to the Universal A2A agent (watsonx-backed) and return its reply.",
    func=a2a_call,
)

# Any LangChain LLM works for orchestration; output content comes from watsonx via A2A.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if __name__ == "__main__":
    print(agent.run("Use the a2a_hello tool to say hello to LangChain."))
