#!/usr/bin/env python3
"""
embeddings_test_watsonx.py

Purpose
-------
Loads .env, validates IBM watsonx.ai credentials, and tests the Watsonx
embeddings pipeline via langchain-ibm. Optionally does a quick end-to-end
vector search with Chroma to ensure the embeddings are actually usable.

Environment variables (any of the aliases will be accepted)
-----------------------------------------------------------
URL:
  - EMBEDDINGS_WATSONX_URL  | WATSONX_URL  | IBM_WATSONX_URL
API KEY:
  - EMBEDDINGS_WATSONX_API_KEY | WATSONX_API_KEY | IBM_CLOUD_API_KEY
PROJECT ID:
  - EMBEDDINGS_WATSONX_PROJECT_ID | WATSONX_PROJECT_ID | PROJECT_ID | IBM_CLOUD_PROJECT_ID

Embedding model:
  - A2A_EMBEDDINGS_MODEL  (default: ibm/slate-125m-english-rtrvr)

Usage
-----
  python embeddings_test_watsonx.py                 # basic check
  python embeddings_test_watsonx.py --with-chroma   # also test Chroma E2E
  python embeddings_test_watsonx.py --dotenv .env.local --model ibm/slate-125m-english-rtrvr
  python embeddings_test_watsonx.py --query "hello embeddings"

Exit codes
----------
  0 : success
  2 : missing dependencies
  3 : missing credentials
  4 : authentication / API error
  5 : embeddings call returned no vectors
  6 : optional Chroma test failed
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# --- Load .env early (best-effort, with fallback to .env.example) ---
def _load_env(dotenv_path: Optional[str]) -> str:
    try:
        from dotenv import load_dotenv, find_dotenv
    except Exception:
        print("ERROR: python-dotenv is required: pip install python-dotenv", file=sys.stderr)
        sys.exit(2)

    loaded_from = ""
    # Explicit path first
    if dotenv_path:
        if Path(dotenv_path).exists():
            load_dotenv(dotenv_path, override=False)
            loaded_from = dotenv_path
    if not loaded_from:
        # Project .env
        p = find_dotenv(filename=".env", usecwd=True)
        if p:
            load_dotenv(p, override=False)
            loaded_from = p
    if not loaded_from:
        # Fall back to .env.example
        p = find_dotenv(filename=".env.example", usecwd=True)
        if p:
            load_dotenv(p, override=False)
            loaded_from = p
    return loaded_from or "(no .env file loaded; using OS env only)"

def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default

def _mask(s: Optional[str], keep: int = 4) -> str:
    if not s:
        return "(missing)"
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep)

def _die(code: int, msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def _cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb + 1e-12)

def main():
    ap = argparse.ArgumentParser(description="watsonx.ai embeddings env + connectivity checker")
    ap.add_argument("--dotenv", default=None, help="Path to .env (optional)")
    ap.add_argument("--model", default=os.getenv("A2A_EMBEDDINGS_MODEL", "ibm/slate-125m-english-rtrvr"),
                    help="Embedding model id (default: ibm/slate-125m-english-rtrvr)")
    ap.add_argument("--query", default="What did the president say about Ketanji Brown Jackson",
                    help="Sample query text for embed_query()")
    ap.add_argument("--with-chroma", action="store_true",
                    help="Also run a mini Chroma roundtrip (index/query) with embeddings")
    args = ap.parse_args()

    loaded_from = _load_env(args.dotenv)
    print(f"Loaded environment from: {loaded_from}")

    # --- Collect credentials (accept multiple common aliases) ---
    url = _env_first("EMBEDDINGS_WATSONX_URL", "WATSONX_URL", "IBM_WATSONX_URL", default="https://us-south.ml.cloud.ibm.com")
    apikey = _env_first("EMBEDDINGS_WATSONX_API_KEY", "WATSONX_API_KEY", "IBM_CLOUD_API_KEY")
    project_id = _env_first("EMBEDDINGS_WATSONX_PROJECT_ID", "WATSONX_PROJECT_ID", "PROJECT_ID", "IBM_CLOUD_PROJECT_ID")

    print("watsonx config:")
    print(f"  url        : {url}")
    print(f"  api_key    : {_mask(apikey)}")
    print(f"  project_id : {_mask(project_id)}")
    print(f"  model_id   : {args.model}")

    if not apikey or not project_id:
        _die(3, "Missing required credentials. Ensure API key and PROJECT ID are set in env. See header for accepted variable names.")

    # --- Import dependencies ---
    try:
        from langchain_ibm import WatsonxEmbeddings
    except Exception as e:
        _die(2, f"Missing or incompatible 'langchain_ibm' package: {e}\nInstall with: pip install langchain-ibm")

    # Optional Chroma import (only if --with-chroma)
    if args.with_chroma:
        try:
            import chromadb  # noqa: F401
            from chromadb import PersistentClient
        except Exception as e:
            _die(2, f"Chroma requested but not installed/usable: {e}\nInstall with: pip install 'chromadb==0.3.26'")

    # --- Initialize embeddings client ---
    print("\nCreating WatsonxEmbeddings client...")
    try:
        emb = WatsonxEmbeddings(
            model_id=args.model,
            url=url,
            apikey=apikey,
            project_id=project_id,
        )
    except Exception as e:
        _die(4, f"Failed to initialize WatsonxEmbeddings (auth/config error?): {e}")

    # --- Smoke test: embed a few strings ---
    docs = [
        "Ketanji Brown Jackson was nominated to the Supreme Court.",
        "The economy is growing and jobs are being created.",
        "Space exploration continues to inspire innovation.",
    ]
    print("Calling embed_documents(...) on 3 short texts...")
    try:
        vecs = emb.embed_documents(docs)
    except Exception as e:
        _die(4, f"Embedding request failed: {e}")

    if not vecs or not isinstance(vecs, list) or not vecs[0]:
        _die(5, "Embeddings returned empty vectors.")
    dim = len(vecs[0])
    print(f"OK ✓ Received {len(vecs)} document vectors of dimension {dim}")
    print(f"  First 5 dims of vec[0]: {[round(x, 6) for x in vecs[0][:5]]}")

    # --- Smoke test: embed a query and compare cosine similarities ---
    print("\nCalling embed_query(...) on the sample question...")
    try:
        qvec = emb.embed_query(args.query)
    except Exception as e:
        _die(4, f"embed_query failed: {e}")
    if not qvec or not isinstance(qvec, list):
        _die(5, "embed_query returned an empty vector.")

    cosines = [round(_cosine(qvec, v), 6) for v in vecs]
    best_i = max(range(len(cosines)), key=lambda i: cosines[i])
    print(f"Query cosine similarities vs docs: {cosines}")
    print(f"Best match index: {best_i}  (doc='{docs[best_i]}')")

    # --- Optional: Mini Chroma roundtrip ---
    if args.with_chroma:
        print("\nRunning minimal Chroma roundtrip with these embeddings...")
        try:
            tmpdir = Path(tempfile.mkdtemp(prefix="wxdemo_chroma_"))
            client = PersistentClient(path=str(tmpdir))
            coll = client.create_collection("wxdemo", metadata={"hnsw:space": "cosine"})
            ids = [f"d{i+1}" for i in range(len(docs))]
            coll.add(
                ids=ids,
                documents=docs,
                embeddings=vecs,
                metadatas=[{"source": "demo"} for _ in docs],
            )
            res = coll.query(
                query_embeddings=[qvec],
                n_results=3,
            )
            print("Chroma query results (ids->distance/order inferred by Chroma):")
            print(res)
        except Exception as e:
            _die(6, f"Chroma roundtrip failed: {e}")
        finally:
            try:
                # best-effort cleanup
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    print("\nAll checks passed ✅")
    sys.exit(0)

if __name__ == "__main__":
    main()
