## `src/a2a_universal/card.py`
import os

def agent_card():
    base = os.getenv("PUBLIC_URL", "http://localhost:8000")
    return {
        "protocolVersion": os.getenv("PROTOCOL_VERSION", "0.3.0"),
        "name": os.getenv("AGENT_NAME", "Universal A2A Hello"),
        "description": os.getenv("AGENT_DESCRIPTION", "Greets the user and echoes input"),
        "version": os.getenv("AGENT_VERSION", "1.2.0"),
        "preferredTransport": "JSONRPC",
        "url": f"{base}/rpc",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "say-hello",
                "name": "Say Hello",
                "description": "Responds with a friendly greeting.",
                "tags": ["hello", "greeting"]
            }
        ]
    }

