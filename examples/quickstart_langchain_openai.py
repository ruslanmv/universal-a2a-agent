# File: examples/quickstart_langchain_openai.py
import httpx
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

BASE = "http://localhost:8001"

# Define the function that calls our A2A endpoint
def a2a_call(prompt: str) -> str:
    try:
        payload = {
            "method": "message/send",
            "params": {"message": {
                "role": "user", "messageId": "lc-tool",
                "parts": [{"type": "text", "text": prompt}],
            }},
        }
        r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        # Extract the text response from the A2A message structure
        for p in (data.get("message") or {}).get("parts", []):
            if p.get("type") == "text":
                return p.get("text", "")
        return "[No text part in A2A response]"
    except httpx.HTTPError as e:
        return f"[A2A HTTP Error: {e}]"
    except Exception as e:
        return f"[A2A call failed: {e}]"


# Create the LangChain Tool
tool = Tool(name="a2a_hello", description="Send a prompt to the Universal A2A agent.", func=a2a_call)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # This is the orchestrator LLM
agent = initialize_agent([tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

if __name__ == "__main__":
    response = agent.run("Use the a2a_hello tool to say hello to LangChain.")
    print(response)