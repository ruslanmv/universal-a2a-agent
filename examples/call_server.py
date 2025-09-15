import httpx, os, sys

BASE = os.getenv("A2A_BASE", "http://localhost:8000")

def call_a2a(text: str) -> str:
    payload = {
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": "poc",
                "parts": [{"type": "text", "text": text}],
            }
        },
    }
    try:
        r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
        r.raise_for_status()
    except httpx.ConnectError:
        return f"[Error] Could not connect to A2A server at {BASE}. Did you run `make run`?"
    except httpx.HTTPStatusError as e:
        return f"[Error] Server returned {e.response.status_code}: {e.response.text}"

    data = r.json()
    for p in (data.get("message") or {}).get("parts", []):
        if p.get("type") == "text":
            return p.get("text", "")
    return "[No text part in A2A response]"

if __name__ == "__main__":
    print(call_a2a("What is the best dish in Genova?"))
