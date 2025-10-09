import httpx
import os
import subprocess
import time

# --- Configurable host/port for CI/local runs ---
PORT = int(os.getenv("A2A_PORT", "8000"))
HOST = os.getenv("A2A_HOST", "0.0.0.0")          # bind address for uvicorn
BASE_HOST = os.getenv("A2A_BASE_HOST", "127.0.0.1")  # where the test client connects
BASE = f"http://{BASE_HOST}:{PORT}"


def _env_with_src_on_path() -> dict:
    """Ensure the child process can import `a2a_universal` in a src/ layout.
    Keeps any existing PYTHONPATH and appends ./src.
    """
    env = os.environ.copy()
    src = os.path.join(os.getcwd(), "src")
    parts = [env.get("PYTHONPATH", "")] + ([src] if os.path.isdir(src) else [])
    env["PYTHONPATH"] = os.pathsep.join([p for p in parts if p])
    return env


def _is_up(timeout: float = 0.5) -> bool:
    try:
        httpx.get(f"{BASE}/healthz", timeout=timeout)
        return True
    except Exception:
        return False


def _ensure_server():
    """If the server is not up, try to start it for local/CI tests.

    Returns:
        None if a server is already responding.
        subprocess.Popen if we started a new one (so caller can terminate).
    """
    if _is_up(timeout=0.5):
        return None

    # Spawn uvicorn; add PYTHONPATH=src for src/ layout repos.
    env = _env_with_src_on_path()
    cmd = [
        "uvicorn",
        "a2a_universal.server:app",
        "--host", HOST,
        "--port", str(PORT),
    ]
    proc = subprocess.Popen(cmd, env=env)

    # Wait up to ~10s for the server to come up
    for _ in range(100):  # 100 * 0.1s = 10s
        # If the process died early, fail fast with a helpful message
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"uvicorn exited early with code {rc} while starting test server")
        if _is_up(timeout=0.5):
            return proc
        time.sleep(0.1)

    raise RuntimeError("Server failed to start within 10s")


def _extract_text_from_a2a_response(data: dict) -> str:
    """Be tolerant to slight response-shape differences.

    Accepts one of:
      - {"result": {"parts": [{"type":"text","text":"..."}]}}
      - {"message": {"parts": [{"type":"text","text":"..."}]}}
      - {"parts":   [{"type":"text","text":"..."}]}
    """
    payload = data
    if isinstance(payload, dict) and "result" in payload and isinstance(payload["result"], dict):
        payload = payload["result"]
    if isinstance(payload, dict) and "message" in payload and isinstance(payload["message"], dict):
        payload = payload["message"]

    parts = []
    if isinstance(payload, dict):
        parts = payload.get("parts") or []
    if isinstance(parts, list) and parts:
        first = parts[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
    return ""


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
        text = _extract_text_from_a2a_response(data)
        assert text and text.lower().startswith("hello"), f"unexpected response: {data!r}"
    finally:
        if isinstance(proc, subprocess.Popen):
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            except Exception:
                pass
