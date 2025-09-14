# examples/quickstart_crewai_watsonx.py
import httpx
from crewai import Agent, Task, Crew

BASE = "http://localhost:8000"

def a2a_call(prompt: str) -> str:
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
    r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    for p in (data.get("message") or {}).get("parts", []):
        if p.get("type") == "text":
            return p.get("text", "")
    return ""

if __name__ == "__main__":
    researcher = Agent(
        role="Researcher",
        goal="Use the A2A tool (watsonx-backed) to answer user prompts",
        backstory="Loves calling external agents.",
        tools=[a2a_call],  # plain Python function works as a CrewAI tool
    )

    task = Task(
        description="Say hello to CrewAI and summarize the word 'ping' in one sentence.",
        agent=researcher,
    )

    result = Crew(agents=[researcher], tasks=[task]).kickoff()
    print(result)
