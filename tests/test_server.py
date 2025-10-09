import httpx
import os
import subprocess
import time

# --- Minimal knobs: let CI override the port/host if needed ---
PORT = int(os.getenv("A2A_PORT", "8000"))
HOST = os.getenv("A2A_HOST", "0.0.0.0")      # where uvicorn binds
BASE = f"http://127.0.0.1:{PORT}"               # where we probe from tests


def _ensure_server():
    """Start the packaged server if /healthz isn't up.
    Keeps it dead simple for GitHub Actions:
      * Adds PYTHONPATH=src so 'a2a_universal' imports in a src/ layout.
      * Waits up to ~6s (60 * 0.1s) for /healthz.
    Returns a Popen if we started the server (so the caller can terminate).
    """
    # Quick probe first
    try:
        httpx.get(f"{BASE}/healthz", timeout=0.5)
        return None
    except Exception:
        pass

    # Spawn uvicorn with minimal env fix for src/ layout
    env = os.environ.copy()
    src = os.path.join(os.getcwd(), "src")
    if os.path.isdir(src):
        env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), src]) if env.get("PYTHONPATH") else src

    cmd = [
        "uvicorn",
        "a2a_universal.server:app",
        "--host", HOST,
        "--port", str(PORT),
    ]
    proc = subprocess.Popen(cmd, env=env)

    # Wait briefly for readiness
    for _ in range(60):  # ~6s
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited early with code {proc.returncode}; check logs")
        try:
            httpx.get(f"{BASE}/healthz", timeout=0.5)
            return proc
        except Exception:
            time.sleep(0.1)

    raise RuntimeError("Server failed to become healthy in time")


def test_a2a_roundtrip():
    proc = _ensure_server()
    try:
        r = httpx.post(
            f"{BASE}/a2a",
            json={
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "messageId": "t1",
                        "parts": [
                            {"type": "text", "text": "ping"}
                        ],
                    }
                },
            },
            timeout=5.0,
        )
        r.raise_for_status()
        data = r.json()

        # Accept either {"result":{...}} or {"message":{...}}
        payload = data.get("result") or data.get("message") or data
        parts = (payload or {}).get("parts", [])
        assert parts and isinstance(parts[0], dict), f"unexpected response: {data!r}"
        text = parts[0].get("text", "")
        assert text and text.lower().startswith("hello"), f"unexpected response: {data!r}"
    finally:
        if isinstance(proc, subprocess.Popen):
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
