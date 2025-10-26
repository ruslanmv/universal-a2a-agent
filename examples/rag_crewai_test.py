#!/usr/bin/env python3
"""
Smoke test for Universal A2A Agent (/knowledge).

What this script does
---------------------
1) Verifies that /knowledge is enabled for a given (or derived) collection.
2) Ingests content from one of three sources into THAT collection:
     - source=github  : clones the repo and ingests docs/tutorial.md (mode=file) or the whole repo (mode=repo).
     - source=local   : ingests a local file or directory (default: this repo's docs/tutorial.md).
     - source=inline  : writes a tiny hardcoded .md into a temp file and ingests it.
3) Runs a sample query against the same collection and prints top results.
4) Shows post-ingest stats so you can confirm the collection actually grew.

Adaptations for production & the updated server code
----------------------------------------------------
- Explicit per-run collections (auto-derived when --collection is omitted):
    * inline  -> "inline"
    * local   -> "local"
    * github  -> "github-file" or "github-repo"
- Optional host:container bind mapping (--bind-map) to make paths visible to a Dockerized server.
- Optional staging into the bind mount (--stage-into-bind) so the server can read test files reliably.
- Pass include_ext/exclude_ext through to the server (defaults align with server-side).
- Compare pre/post collection counts; if not growing and bind-map is present, optionally retry by staging.
- Clear, actionable console messages.

Usage examples
--------------
# Default: clone from GitHub and ingest docs/tutorial.md into derived collection "github-file"
python examples/rag_crewai_test.py --base http://localhost:8000 --mode file

# Clone from GitHub and ingest entire repo into named collection
python examples/rag_crewai_test.py --base http://localhost:8000 --mode repo --collection a2a-repo

# Use local docs/tutorial.md into collection "local"
python examples/rag_crewai_test.py --base http://localhost:8000 --source local --collection local

# Use a specific local path (file or directory)
python examples/rag_crewai_test.py --base http://localhost:8000 --source local --local-path ./docs --collection docs

# Inline hardcoded test, into collection "inline" (auto)
python examples/rag_crewai_test.py --base http://localhost:8000 --source inline

# If your server runs in Docker and only sees /work, stage into that mount automatically:
python examples/rag_crewai_test.py \
  --base http://localhost:8000 \
  --source local \
  --local-path ./docs/tutorial.md \
  --bind-map "/Users/me/universal-a2a-agent:/work" \
  --stage-into-bind \
  --collection local
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_URL = "https://github.com/ruslanmv/universal-a2a-agent.git"
TUTORIAL_PATH = "docs/tutorial.md"  # relative to repo root

# Friendly defaults
DEFAULT_QUESTION_TUTORIAL = "From the Universal A2A tutorial, list the primary endpoints and how to call them."
DEFAULT_QUESTION = " what is A2A "
DEFAULT_THRESHOLD = 0.0  # permissive for smoke tests

# Keep include/exclude in sync with the server defaults
DEFAULT_INCLUDE_EXT = ".md,.mdx,.py,.ipynb,.txt"
DEFAULT_EXCLUDE_EXT = ".png,.jpg,.jpeg,.gif,.pdf"

INLINE_DOC = textwrap.dedent(
    """
    # A2A Test Doc (Inline)
    This is a tiny, hardcoded document to verify RAG end-to-end.

    Primary endpoints (for testing):
    - POST /a2a — Universal A2A envelope
    - POST /rpc — JSON-RPC 2.0 wrapper
    - POST /openai/v1/chat/completions — OpenAI-compatible chat
    - GET  /healthz — liveness probe
    - GET  /readyz  — readiness probe
    """
).strip()

INLINE_QUESTION = "List the primary endpoints mentioned in the A2A Test Doc."


# ------------------------------- HTTP helpers -------------------------------

def _get(base: str, path: str, timeout: int = 15, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    return requests.get(f"{base}{path}", timeout=timeout, params=params or {})


def _post(
    base: str,
    path: str,
    json_body: Dict[str, Any],
    timeout: int = 60,
    params: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    return requests.post(f"{base}{path}", json=json_body, timeout=timeout, params=params or {})


# ------------------------------- util helpers -------------------------------

def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def clone_repo(tmpdir: Path) -> Path:
    repo_dir = tmpdir / "universal-a2a-agent"
    run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
    return repo_dir


def _parse_csv_opt(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items or None


def _parse_bind_map(s: Optional[str]) -> Optional[Tuple[Path, Path]]:
    """Parse --bind-map 'HOST:CONTAINER' into (host_root, container_root)."""
    if not s:
        return None
    if ":" not in s:
        print("⚠️  --bind-map must be HOST:CONTAINER (e.g., /Users/me/project:/work)")
        return None
    host, container = s.split(":", 1)
    return Path(host).resolve(), Path(container)


def _rewrite_for_container(p: Path, bind: Optional[Tuple[Path, Path]]) -> Optional[Path]:
    """If p is under bind host_root, return its container path; else None."""
    if not bind:
        return None
    host_root, container_root = bind
    try:
        rel = p.resolve().relative_to(host_root)
    except Exception:
        return None
    return (container_root / rel).resolve()


def _stage_into_bind(src: Path, bind: Tuple[Path, Path]) -> Tuple[Path, Path]:
    """
    Copy a file/directory into a stable folder under the bind host root,
    and return (host_staged_path, container_staged_path).
    """
    host_root, container_root = bind
    stage_host_dir = host_root / ".a2a_ingest_tmp"
    stage_host_dir.mkdir(parents=True, exist_ok=True)

    dest_host = stage_host_dir / src.name
    if dest_host.exists():
        # Clean previous staging of the same name
        if dest_host.is_dir():
            shutil.rmtree(dest_host)
        else:
            dest_host.unlink()

    if src.is_dir():
        shutil.copytree(src, dest_host)
    else:
        shutil.copy2(src, dest_host)

    dest_container = (container_root / ".a2a_ingest_tmp" / src.name).resolve()
    return dest_host, dest_container


def derive_collection_name(source: str, mode: str, user_collection: Optional[str]) -> str:
    """Derive a sensible collection name if not provided by the user."""
    if user_collection:
        return user_collection
    if source == "inline":
        return "inline"
    if source == "local":
        return "local"
    if source == "github":
        return f"github-{mode}"
    return "default"


# --------------------------------- checks -----------------------------------

def check_knowledge_enabled(base: str, collection: Optional[str]) -> Dict[str, Any]:
    try:
        r = _get(base, "/knowledge/stats", timeout=15, params={"collection": collection} if collection else None)
    except requests.RequestException as e:
        print(f"❌ Cannot reach {base}/knowledge/stats: {e}")
        sys.exit(2)

    if r.status_code == 503:
        print("❌ /knowledge is disabled. Set A2A_ENABLE_KNOWLEDGE=1 on the server and restart.")
        sys.exit(3)
    if r.status_code != 200:
        print(f"❌ Unexpected status from /knowledge/stats: {r.status_code} {r.text}")
        sys.exit(4)

    stats = {}
    try:
        stats = r.json()
    except Exception:
        pass
    print(f"✅ /knowledge is enabled (collection={collection or 'default'}): {json.dumps(stats, indent=2)}")
    return stats


def reset_index(base: str, collection: Optional[str]) -> None:
    print(f"→ Resetting knowledge index (collection={collection or 'default'})...")
    try:
        r = _post(base, "/knowledge/reset", json_body={}, timeout=60, params={"collection": collection} if collection else None)
    except requests.RequestException as e:
        print(f"❌ Reset failed: {e}")
        sys.exit(5)
    if r.status_code != 200:
        print(f"❌ Reset failed: {r.status_code} {r.text}")
        sys.exit(5)
    print("✅ Reset ok:", r.text)


# -------------------------------- ingest ------------------------------------

def ingest_paths(
    base: str,
    paths: List[Path],
    chunk_size: int = 1400,
    chunk_overlap: int = 160,
    collection: Optional[str] = None,
    include_ext: Optional[List[str]] = None,
    exclude_ext: Optional[List[str]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "paths": [str(p) for p in paths],
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    if include_ext:
        payload["include_ext"] = include_ext
    if exclude_ext:
        payload["exclude_ext"] = exclude_ext
    if collection:
        payload["collection"] = collection

    print("→ Ingesting:", json.dumps(payload, indent=2))
    try:
        r = _post(base, "/knowledge/ingest", json_body=payload, timeout=600)
    except requests.RequestException as e:
        print(f"❌ Ingest failed: {e}")
        sys.exit(5)
    if r.status_code != 200:
        print(f"❌ Ingest failed: {r.status_code} {r.text}")
        sys.exit(5)
    print("✅ Ingest ok:", json.dumps(r.json(), indent=2))


def fetch_stats(base: str, label: str = "Stats", collection: Optional[str] = None) -> Dict[str, Any]:
    try:
        r = _get(base, "/knowledge/stats", timeout=15, params={"collection": collection} if collection else None)
        s = r.json() if r.status_code == 200 else {}
    except Exception:
        s = {}
    print(f"{label} (collection={collection or 'default'}):", json.dumps(s, indent=2))
    return s


# --------------------------------- query ------------------------------------

def query(
    base: str,
    question: str,
    k: int = 6,
    score_threshold: float = DEFAULT_THRESHOLD,
    collection: Optional[str] = None,
) -> List[Any]:
    payload: Dict[str, Any] = {"q": question, "k": k, "score_threshold": score_threshold}
    if collection:
        payload["collection"] = collection

    print("→ Querying:", json.dumps(payload, indent=2))
    try:
        r = _post(base, "/knowledge/query", json_body=payload, timeout=60)
    except requests.RequestException as e:
        print(f"❌ Query failed: {e}")
        sys.exit(6)
    if r.status_code != 200:
        print(f"❌ Query failed: {r.status_code} {r.text}")
        sys.exit(6)

    # Be tolerant to both {"results":[...]} and bare [...]
    try:
        data = r.json()
    except ValueError:
        data = {"results": [r.text]}

    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        results = [results]

    # Filter out crewai string like "Relevant Content: No relevant content found."
    def _is_empty_no_relevant(x: Any) -> bool:
        s: str
        if isinstance(x, str):
            s = x.strip().lower()
        elif isinstance(x, dict):
            txt = (
                x.get("text")
                or x.get("content")
                or x.get("page_content")
                or (x.get("document") or {}).get("page_content")
                or ""
            )
            s = str(txt).strip().lower()
        else:
            return False
        return "relevant content" in s and "no relevant content found" in s

    results = [r for r in results if not _is_empty_no_relevant(r)]

    print(f"✅ Got {len(results)} results\n")
    return results


def _extract_view(item: Any) -> Tuple[str, Optional[float], str]:
    """
    Extract (text, score, source) from heterogeneous result items:
    - dicts from server normalizer
    - strings (legacy)
    - LangChain Document-like objects
    """
    # 1) dict-like
    if isinstance(item, dict):
        meta = item.get("metadata") or item.get("meta") or {}
        text = (
            item.get("text")
            or item.get("content")
            or item.get("page_content")
            or (item.get("document") or {}).get("page_content")
            or (item.get("document") or {}).get("content")
            or item.get("answer")
            or item.get("chunk")
            or ""
        )
        score = item.get("score") or item.get("similarity") or meta.get("score")
        src = (
            meta.get("source")
            or meta.get("path")
            or meta.get("file")
            or meta.get("id")
            or "unknown"
        )
        return str(text), float(score) if score is not None else None, str(src)

    # 2) LangChain Document-like
    if hasattr(item, "page_content"):
        try:
            text = getattr(item, "page_content", "") or ""
            meta = getattr(item, "metadata", {}) or {}
            score = meta.get("score")
            src = meta.get("source") or meta.get("path") or meta.get("file") or "unknown"
            return str(text), float(score) if score is not None else None, str(src)
        except Exception:
            pass

    # 3) plain string or anything else
    return str(item), None, "unknown"


def pretty_print(results: List[Any], max_chars: int = 320) -> None:
    if not results:
        print("No results. If you just switched embeddings providers, reset + re-ingest your data.\n")
        return

    for i, res in enumerate(results, 1):
        text, score, src = _extract_view(res)
        snippet = (text[:max_chars] + "…") if len(text) > max_chars else text
        print(f"[{i}] score={score} source={src}\n{snippet}\n---\n")


# --------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Smoke test for /knowledge endpoints (production-ready).")

    ap.add_argument("--base", default=os.getenv("A2A_BASE", "http://localhost:8000"), help="Base URL of A2A server")
    ap.add_argument("--mode", choices=["file", "repo"], default="file",
                    help="(source=github only) Ingest single file (docs/tutorial.md) or entire repo")
    ap.add_argument("--source", choices=["github", "local", "inline"], default="github",
                    help="Where to read content from")
    ap.add_argument("--local-path", default=None,
                    help="When --source=local, path to file or directory (default: ../docs/tutorial.md)")
    ap.add_argument("--question", default=None, help="Query to run after ingestion")
    ap.add_argument("--k", type=int, default=6, help="Top-k results to fetch")
    ap.add_argument("--score_threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Retrieval score threshold (0.0 recommended for smoke tests)")
    ap.add_argument("--reset", action="store_true",
                    help="Reset the knowledge index before ingesting")
    ap.add_argument("--collection", default=None,
                    help="Optional explicit collection name for this test run")

    # NEW: Bind-map & staging so the server can see your paths inside Docker
    ap.add_argument("--bind-map", default=None,
                    help="HOST:CONTAINER path mapping for Docker (e.g., /Users/me/project:/work)")
    ap.add_argument("--stage-into-bind", action="store_true",
                    help="Copy content into the host side of the bind and ingest via the container path")

    # Pass-through include/exclude (to match server defaults if you customize)
    ap.add_argument("--include-ext", default=DEFAULT_INCLUDE_EXT,
                    help=f"Comma-separated include extensions (default: {DEFAULT_INCLUDE_EXT})")
    ap.add_argument("--exclude-ext", default=DEFAULT_EXCLUDE_EXT,
                    help=f"Comma-separated exclude extensions (default: {DEFAULT_EXCLUDE_EXT})")

    args = ap.parse_args()

    base = args.base.rstrip("/")
    collection = derive_collection_name(args.source, args.mode, args.collection)

    print(f"A2A base: {base}")
    print(f"Using collection: {collection}")

    bind = _parse_bind_map(args.bind_map)

    # 1) Knowledge check (pre-ingest stats)
    pre_stats = check_knowledge_enabled(base, collection)
    pre_count = ((pre_stats or {}).get("collection") or {}).get("count")

    # Optional reset (clean index if switching providers/creds)
    if args.reset:
        reset_index(base, collection)

    # 2) Prepare content source
    tmpdir = Path(tempfile.mkdtemp(prefix="a2a_patch_smoke_"))
    print(f"Working dir: {tmpdir}")

    try:
        target_paths: List[Path] = []
        question = args.question  # default None here; set per source below

        if args.source == "github":
            repo_dir = clone_repo(tmpdir)
            target = repo_dir / TUTORIAL_PATH if args.mode == "file" else repo_dir
            if not target.exists():
                print(f"❌ Target not found: {target}")
                sys.exit(7)
            print(f"Target to ingest (github/{args.mode}): {target}")

            # Staging or rewrite (so server can see it in container)
            if args.stage_into_bind and bind:
                host_staged, container_staged = _stage_into_bind(target, bind)
                print(f"↳ Staged into bind: host={host_staged} -> container={container_staged}")
                target_paths = [container_staged]
            else:
                rewritten = _rewrite_for_container(target, bind) if bind else None
                target_paths = [rewritten or target]

            question = question or (DEFAULT_QUESTION_TUTORIAL if args.mode == "file" else DEFAULT_QUESTION)

        elif args.source == "local":
            if args.local_path:
                target = Path(args.local_path).resolve()
            else:
                # examples/ -> project root -> docs/tutorial.md
                project_root = Path(__file__).resolve().parents[1]
                target = (project_root / TUTORIAL_PATH).resolve()

            if not target.exists():
                print(f"❌ Local target not found: {target}")
                sys.exit(7)
            print(f"Target to ingest (local): {target}")

            if args.stage_into_bind and bind:
                host_staged, container_staged = _stage_into_bind(target, bind)
                print(f"↳ Staged into bind: host={host_staged} -> container={container_staged}")
                target_paths = [container_staged]
            else:
                rewritten = _rewrite_for_container(target, bind) if bind else None
                if bind and rewritten is None:
                    print("⚠️  --bind-map provided but target is not under HOST prefix; using original path (may not be visible in container).")
                target_paths = [rewritten or target]

            question = question or DEFAULT_QUESTION_TUTORIAL

        else:  # inline
            inline_file = tmpdir / "inline_test_doc.md"
            inline_file.write_text(INLINE_DOC, encoding="utf-8")
            print(f"Target to ingest (inline): {inline_file}")

            if args.stage_into_bind and bind:
                host_staged, container_staged = _stage_into_bind(inline_file, bind)
                print(f"↳ Staged into bind: host={host_staged} -> container={container_staged}")
                target_paths = [container_staged]
            else:
                rewritten = _rewrite_for_container(inline_file, bind) if bind else None
                target_paths = [rewritten or inline_file]

            question = question or INLINE_QUESTION

        include_ext = _parse_csv_opt(args.include_ext)
        exclude_ext = _parse_csv_opt(args.exclude_ext)

        # 3) Ingest (into derived/explicit collection)
        ingest_paths(
            base,
            target_paths,
            collection=collection,
            include_ext=include_ext,
            exclude_ext=exclude_ext,
        )

        # 4) Show stats after ingest to confirm growth
        post_stats = fetch_stats(base, label="Post-ingest stats", collection=collection)
        coll = (post_stats or {}).get("collection") or {}
        vdb = ((post_stats or {}).get("vectordb") or {}).get("provider", "")
        post_count = coll.get("count") if isinstance(coll, dict) else None

        grew = (
            isinstance(post_count, int)
            and (
                (isinstance(pre_count, int) and post_count > pre_count)
                or (pre_count is None and post_count >= 1)
            )
        )

        if vdb == "chromadb" and not grew:
            print("⚠️  Warning: Chroma collection count did not grow after ingest.")
            print("   If the server runs in Docker, ensure the ingested path exists inside the container.")
            if bind:
                print("   Tip: use --stage-into-bind to copy the target into the bind mount automatically.")

        # 5) Query (from same collection)
        results = query(
            base,
            question or DEFAULT_QUESTION,
            k=args.k,
            score_threshold=args.score_threshold,
            collection=collection,
        )
        pretty_print(results)

        print("\n🎉 Smoke test looks healthy if results above reference the ingested content.")
        print("   If you see zero/irrelevant hits, verify embeddings, lower score_threshold, ensure path visibility or use --stage-into-bind.")

    finally:
        # Clean up the cloned repo/inline file (optional)
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
