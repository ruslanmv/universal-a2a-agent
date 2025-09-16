# examples/quickstart_crewai_watsonx.py
import httpx
from crewai import Agent, Task, Crew
from crewai_tools import Tool  # FIXED: Import the 'Tool' class, not the 'tool' decorator.

BASE = "http://localhost:8000"

# -------------------------------------------------------------------
# Define the function as a standard Python function (no decorator)
# -------------------------------------------------------------------
def a2a_call(prompt: str) -> str:
    """Send a user prompt to the Universal A2A agent and return its reply text."""
    payload = {
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": "crewai-tool",
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
    except Exception as e:
        return f"[A2A call failed: {e}]"

# -------------------------------------------------------------------
# CrewAI setup
# -------------------------------------------------------------------
if __name__ == "__main__":
    # FIXED: Create a Tool instance from the function
    a2a_tool = Tool(
        name="Universal A2A Agent",
        description="Send a user prompt to the Universal A2A agent to get its expert reply.",
        func=a2a_call,
    )

    researcher = Agent(
        role="Researcher",
        goal="Use the A2A tool (watsonx-backed) to answer user prompts",
        backstory="An expert in delegating tasks to other specialized agents.",
        tools=[a2a_tool],  # FIXED: Pass the Tool instance to the agent
        verbose=True
    )

    task = Task(
        description="Use your tool to say hello to CrewAI, then ask it to summarize the word 'ping' in one sentence.",
        expected_output="A friendly greeting followed by a concise, one-sentence definition of 'ping'.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=2)
    result = crew.kickoff()
    print("\n[Final Answer]:", result)