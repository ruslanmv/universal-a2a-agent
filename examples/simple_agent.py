# examples/tiny_agent.py
from a2a_universal.app import build
from a2a_universal import run


async def handle_text(text: str) -> str:
    return f"Hello from my custom agent. You said: {text[:200]}"


# Build a full A2A app from a single function
app = build(
    handler=handle_text,
    name="Tiny Agent",
    description="One function → full A2A",
)

if __name__ == "__main__":
    run("__main__:app", host="0.0.0.0", port=8080, reload=False)
