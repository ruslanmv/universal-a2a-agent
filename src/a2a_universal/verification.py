#Beta not integrated yet
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import httpx

from .validators import validate_agent_card, validate_message


DEFAULT_TIMEOUT_SEC = float(
    # Allow override via env; keep tight deadlines for safety
    # e.g., VERIFY_TIMEOUT=8.0
    __import__("os").getenv("VERIFY_TIMEOUT", "8.0")
)
DEFAULT_CONCURRENCY = int(__import__("os").getenv("VERIFY_CONCURRENCY", "50"))
VERIFY_TLS = __import__("os").getenv("VERIFY_TLS", "true").lower() != "false"


def _normalize_base(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    # strip trailing slashes
    while url.endswith("/"):
        url = url[:-1]
    return url


async def _fetch_json(client: httpx.AsyncClient, url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], float]:
    t0 = time.perf_counter()
    try:
        r = await client.get(url)
        r.raise_for_status()
        return r.json(), None, (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        return None, str(e), (time.perf_counter() - t0) * 1000.0


async def _post_json(client: httpx.AsyncClient, url: str, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str], float]:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json(), None, (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        return None, str(e), (time.perf_counter() - t0) * 1000.0


async def _discover_card(client: httpx.AsyncClient, base_url: str) -> Tuple[Optional[Dict[str, Any]], List[str], str, float]:
    """Try standard well-known locations; return first success."""
    base = _normalize_base(base_url)
    candidates = [
        urljoin(base + "/", ".well-known/agent-card.json"),
        urljoin(base + "/", ".well-known/agent.json"),  # alias some stacks expose
    ]
    errors: List[str] = []
    for u in candidates:
        data, err, elapsed = await _fetch_json(client, u)
        if data is not None:
            return data, [], u, elapsed
        errors.append(f"{u}: {err}")
    # give up
    return None, errors, candidates[-1], 0.0


async def _check_health(client: httpx.AsyncClient, base_url: str) -> Tuple[str, int, float]:
    """Probe common health endpoints. Returns (status, http_code, ms)."""
    base = _normalize_base(base_url)
    candidates = ["/healthz", "/health", "/readyz"]
    for path in candidates:
        data, err, elapsed = await _fetch_json(client, base + path)
        if data is not None:
            return "ok", 200, elapsed
    return "unknown", 0, 0.0


def _extract_rpc_url(card: Dict[str, Any], base_url: str) -> str:
    # Prefer 'url' in the Agent Card; if relative, join with base
    url = str(card.get("url") or "").strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin(_normalize_base(base_url) + "/", url.lstrip("/"))


def _supports_streaming(card: Dict[str, Any]) -> bool:
    caps = card.get("capabilities") or {}
    if not isinstance(caps, dict):
        return False
    return bool(caps.get("streaming") is True)


async def _send_probe_message(client: httpx.AsyncClient, rpc_url: str) -> Tuple[Optional[Dict[str, Any]], List[str], float]:
    """Send a minimal JSON-RPC 'message/send' and validate the response 'event'."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [{"text": "ping"}],  # accept modern TextPart
            },
            "configuration": {"accepted_output_modes": ["text/plain"]},
        },
    }
    data, err, elapsed = await _post_json(client, rpc_url, payload)
    if err:
        return None, [f"RPC error: {err}"], elapsed

    # JSON-RPC success should carry "result"
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, dict):
        return None, ["RPC response missing 'result' object."], elapsed

    # Validate the 'event' (kind: message / etc.)
    errors = validate_message(result)
    if errors:
        return result, errors, elapsed

    return result, [], elapsed


async def verify_a2a(url: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC, verify_tls: bool = VERIFY_TLS) -> Dict[str, Any]:
    """Verify a single A2A server: card, health, and a probe message."""
    base = _normalize_base(url)
    timeout = httpx.Timeout(timeout_sec, connect=timeout_sec)
    async with httpx.AsyncClient(timeout=timeout, verify=verify_tls) as client:
        report: Dict[str, Any] = {
            "target": base,
            "status": "fail",
            "card_url": None,
            "rpc_url": None,
            "timings_ms": {"card": 0.0, "health": 0.0, "rpc": 0.0},
            "card": None,
            "errors": {"card": [], "rpc": [], "health": []},
            "warnings": [],
        }

        # 1) Discover Agent Card
        card, card_fetch_errors, card_url, card_ms = await _discover_card(client, base)
        report["card_url"] = card_url
        report["timings_ms"]["card"] = round(card_ms, 2)

        if card is None:
            report["errors"]["card"].extend(card_fetch_errors or ["Agent Card not found."])
            return report

        # 2) Validate Agent Card structure
        report["card"] = {
            "name": card.get("name"),
            "version": card.get("version"),
            "description": card.get("description"),
            "capabilities": card.get("capabilities"),
            "defaultInputModes": card.get("defaultInputModes"),
            "defaultOutputModes": card.get("defaultOutputModes"),
            "skills": card.get("skills"),
            "url": card.get("url"),
        }
        card_errors = validate_agent_card(card)
        if card_errors:
            report["errors"]["card"].extend(card_errors)

        # 3) Health probe (best-effort; not strictly required by spec)
        health_status, _, health_ms = await _check_health(client, base)
        report["timings_ms"]["health"] = round(health_ms, 2)
        if health_status != "ok":
            report["errors"]["health"].append("No healthy /healthz|/health|/readyz endpoint detected.")

        # 4) RPC probe
        rpc_url = _extract_rpc_url(card, base)
        report["rpc_url"] = rpc_url
        event, rpc_errors, rpc_ms = await _send_probe_message(client, rpc_url)
        report["timings_ms"]["rpc"] = round(rpc_ms, 2)
        if rpc_errors:
            report["errors"]["rpc"].extend(rpc_errors)

        # 5) Determine overall status
        any_errors = report["errors"]["card"] or report["errors"]["rpc"]
        degraded = (not report["errors"]["card"]) and (not report["errors"]["rpc"]) and report["errors"]["health"]
        report["status"] = "ok" if not any_errors else ("degraded" if degraded else "fail")
        return report


async def verify_a2a_bulk(urls: List[str], concurrency: int = DEFAULT_CONCURRENCY, timeout_sec: float = DEFAULT_TIMEOUT_SEC, verify_tls: bool = VERIFY_TLS) -> List[Dict[str, Any]]:
    """Verify many A2A servers concurrently with a bounded semaphore."""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(u: str) -> Dict[str, Any]:
        async with sem:
            return await verify_a2a(u, timeout_sec=timeout_sec, verify_tls=verify_tls)

    tasks = [asyncio.create_task(_one(u)) for u in urls]
    return await asyncio.gather(*tasks)
