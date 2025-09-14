import httpx, time, os, subprocess, signal

BASE = "http://localhost:8000"

def _ensure_server():
    # If server not up, try to start it for local tests
    try:
        httpx.get(f"{BASE}/healthz", timeout=1.0)
        return None
    except Exception:
        proc = subprocess.Popen(["uvicorn", "a2a_universal.server:app", "--host", "0.0.0.0", "--port", "8000"])
        # wait up to 5s
        for _ in range(50):
            try:
                time.sleep(0.1)
                httpx.get(f"{BASE}/healthz", timeout=0.5)
                return proc
            except Exception:
                continue
        raise RuntimeError("Server failed to start")

def test_a2a_roundtrip():
    proc = _ensure_server()
    try:
        r = httpx.post(f"{BASE}/a2a", json={
            "method": "message/send",
            "params": {"message": {"role": "user", "messageId": "t1", "parts": [{"type": "text", "text": "ping"}]}}
        }, timeout=5.0)
        r.raise_for_status()
        data = r.json()
        assert data["message"]["parts"][0]["text"].startswith("Hello")
    finally:
        if isinstance(proc, subprocess.Popen):
            proc.terminate()
