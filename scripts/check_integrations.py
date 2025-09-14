#!/usr/bin/env python
"""
Universal A2A Agent — Simple Integration Sanity Check

Usage:
  python scripts/check_integrations.py
  BASE=http://localhost:9000 python scripts/check_integrations.py

What it does:
- Ensures the A2A server is reachable (starts it if needed)
- Runs minimal checks for each optional framework IF it's installed:
    * Base HTTP (/a2a) ping
    * LangChain tool adapter
    * LangGraph native MessagesState node
    * CrewAI BaseTool (no external LLM required)
    * AutoGen registered function
    * BeeAI Framework agent (async)
- Prints PASS/FAIL/SKIP lines + summary and exits 1 on any failure among attempted checks
"""

from __future__ import annotations
import os
import sys
import time
import subprocess
from typing import Tuple, List

# --- Config ---
BASE = os.getenv("BASE", "http://localhost:8000")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# --- Utilities ---
def start_server_if_needed() -> subprocess.Popen | None:
    """If /healthz is not responding, start uvicorn in the background and return the Popen handle."""
    import httpx
    try:
        httpx.get(f"{BASE}/healthz", timeout=1.0)
        return None  # already up
    except Exception:
        pass

    print(f"[-] Server not running at {BASE}, starting uvicorn …")
    proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "a2a_universal.server:app",
        "--host", HOST, "--port", str(PORT)
    ])
    # Wait briefly for health
    for _ in range(50):
        try:
            time.sleep(0.1)
            httpx.get(f"{BASE}/healthz", timeout=0.5)
            print("[+] Server is up")
            return proc
        except Exception:
            continue
    print("[!] WARNING: healthz did not respond; tests may fail.")
    return proc

def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        proc.kill()

def ok_text(text: str) -> bool:
    return isinstance(text, str) and "hello" in text.lower()

# --- Individual checks (return tuple: (name, status_str, details)) ---

def check_base_http() -> Tuple[str, str, str]:
    import httpx
    try:
        r = httpx.post(
            f"{BASE}/a2a",
            json={
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "messageId": "t1",
                        "parts": [{"type": "text", "text": "ping"}],
                    }
                },
            },
            timeout=5.0,
        )
        r.raise_for_status()
        data = r.json()
        txt = data.get("message", {}).get("parts", [{}])[0].get("text", "")
        return ("base-http", "PASS" if ok_text(txt) else "FAIL", txt)
    except Exception as e:
        return ("base-http", "FAIL", repr(e))

def check_langchain() -> Tuple[str, str, str]:
    try:
        import langchain  # noqa: F401
        from a2a_universal.adapters.langchain_tool import a2a_hello
    except Exception as e:
        return ("langchain", "SKIP", f"not installed or import failed: {e!r}")
    try:
        # Tool objects typically expose .invoke for simple string schema.
        if hasattr(a2a_hello, "invoke"):
            txt = a2a_hello.invoke("ping from LC")
        elif callable(a2a_hello):
            txt = a2a_hello("ping from LC")
        else:
            return ("langchain", "FAIL", "tool not callable/invokable")
        return ("langchain", "PASS" if ok_text(str(txt)) else "FAIL", str(txt))
    except Exception as e:
        return ("langchain", "FAIL", repr(e))

def check_langgraph() -> Tuple[str, str, str]:
    try:
        from langgraph.graph import StateGraph, END, MessagesState
        from langchain_core.messages import HumanMessage
        from a2a_universal.adapters.langgraph_agent import A2AAgentNode
    except Exception as e:
        return ("langgraph", "SKIP", f"not installed or import failed: {e!r}")
    try:
        g = StateGraph(MessagesState)
        g.add_node("a2a", A2AAgentNode(base_url=BASE))
        g.add_edge("__start__", "a2a")
        g.add_edge("a2a", END)
        app = g.compile()
        out = app.invoke({"messages": [HumanMessage(content="ping from LangGraph")]})
        msg = out["messages"][-1].content
        return ("langgraph", "PASS" if ok_text(msg) else "FAIL", msg)
    except Exception as e:
        return ("langgraph", "FAIL", repr(e))

def check_crewai() -> Tuple[str, str, str]:
    # Use our BaseTool directly; no LLM keys required.
    try:
        import crewai  # noqa: F401
        from a2a_universal.adapters.crewai_base_tool import A2AHelloTool
    except Exception as e:
        return ("crewai", "SKIP", f"not installed or import failed: {e!r}")
    try:
        tool = A2AHelloTool()
        txt = tool._run("ping from CrewAI")
        return ("crewai", "PASS" if ok_text(txt) else "FAIL", txt)
    except Exception as e:
        return ("crewai", "FAIL", repr(e))

def check_autogen() -> Tuple[str, str, str]:
    try:
        import autogen  # noqa: F401
        from a2a_universal.adapters.autogen_tool import a2a_hello
    except Exception as e:
        return ("autogen", "SKIP", f"not installed or import failed: {e!r}")
    try:
        txt = a2a_hello("ping from AutoGen")
        return ("autogen", "PASS" if ok_text(txt) else "FAIL", str(txt))
    except Exception as e:
        return ("autogen", "FAIL", repr(e))

def check_beeai() -> Tuple[str, str, str]:
    try:
        import asyncio
        from beeai_framework.backend import UserMessage
        from a2a_universal.adapters.beeai_agent import make_beeai_agent
    except Exception as e:
        return ("beeai", "SKIP", f"not installed or import failed: {e!r}")
    try:
        async def run_once():
            agent = make_beeai_agent(BASE)
            result = await agent.run(UserMessage("ping from BeeAI"))
            return str(result)

        txt = asyncio.run(run_once())
        return ("beeai", "PASS" if ok_text(txt) else "FAIL", txt)
    except Exception as e:
        return ("beeai", "FAIL", repr(e))

# --- Main driver ---
def main() -> int:
    print(f"== Universal A2A Integration Check ==\nBASE={BASE}\n")
    proc = start_server_if_needed()

    checks = [
        check_base_http,
        check_langchain,
        check_langgraph,
        check_crewai,
        check_autogen,
        check_beeai,
    ]
    results: List[Tuple[str, str, str]] = []
    try:
        for fn in checks:
            name = fn.__name__.replace("check_", "")
            print(f"-> {name} …", end=" ", flush=True)
            res = fn()
            results.append(res)
            print(f"{res[1]}")
    finally:
        stop_server(proc)

    print("\nSummary:")
    attempted = 0
    failed = 0
    for name, status, details in results:
        badge = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(status, "•")
        print(f"  {badge} {name:10s} {status:4s}  {details[:120]}")
        if status != "SKIP":
            attempted += 1
            if status == "FAIL":
                failed += 1

    if attempted == 0:
        print("\nNo frameworks installed (only base HTTP was attempted)." if results and results[0][0] == "base-http" else "\nNo checks attempted.")
    print(f"\nAttempted: {attempted}  Failed: {failed}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
