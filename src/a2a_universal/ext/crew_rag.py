# src/a2a_universal/ext/crew_rag.py
# SPDX-License-Identifier: Apache-2.0
"""
Universal A2A Agent — RAG utilities (production-ready)

- Builds a CrewAI RagTool using env-driven configuration.
- Works with Chroma (default) or Qdrant vector DBs.
- Supports multiple embeddings providers; verified with IBM watsonx.ai.
- Adds a robust, optional fallback that performs ingestion/query directly via
  WatsonxEmbeddings + Chroma if the upstream tool path returns empty results.
- Avoids duplicate indexing by running the fallback only when the native path
  doesn't grow the collection.
- Uses stable, content-aware IDs and upsert behavior for Chroma fallback.
- Automatically shrinks chunk size in the fallback if the embeddings model
  rejects inputs that exceed its maximum sequence length.
- Truncates over-long queries in the fallback path defensively.

Key env vars
-----------
A2A_ENABLE_KNOWLEDGE=1
A2A_KNOWLEDGE_DIR=./.a2a_knowledge
A2A_VDB=chromadb | qdrant
A2A_CHROMA_COLLECTION=a2a-knowledge
A2A_QDRANT_COLLECTION=a2a-knowledge
A2A_QDRANT_URL=http://localhost:6333

A2A_EMBEDDINGS_PROVIDER=watsonx | openai | ...
A2A_EMBEDDINGS_MODEL=ibm/slate-125m-english-rtrvr (for watsonx)

# Watsonx credentials (any alias works)
EMBEDDINGS_WATSONX_URL | WATSONX_URL
EMBEDDINGS_WATSONX_API_KEY | WATSONX_API_KEY | IBM_CLOUD_API_KEY
EMBEDDINGS_WATSONX_PROJECT_ID | WATSONX_PROJECT_ID | PROJECT_ID | IBM_CLOUD_PROJECT_ID

Fallback controls
-----------------
A2A_RAG_FALLBACK_LC=1  (default)  -> enable manual Watsonx+Chroma fallback
A2A_RAG_FALLBACK_LC=0             -> disable fallback

Security (optional)
-------------------
A2A_STRICT_INGEST_ROOT=1 -> refuse ingest outside A2A_KNOWLEDGE_DIR

Embedding guardrails (optional)
-------------------------------
A2A_MAX_EMBED_TOKENS=512          # model's max tokens (approx)
A2A_CHARS_PER_TOKEN=2.0           # rough chars/token for code/md
A2A_MIN_CHUNK_CHARS=256           # minimum chunk size used by auto-shrinker
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crewai_tools.tools.rag.rag_tool import RagTool

# Logger for all RAG internals
log = logging.getLogger("a2a.knowledge")

# Cache RagTool per collection
_RAG_CACHE: Dict[str, RagTool] = {}


# ---------- helpers ----------

def _env_list(name: str, default_csv: str) -> List[str]:
    return [x.strip() for x in os.getenv(name, default_csv).split(",") if x.strip()]


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty env var among names (or default)."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


def _norm_vdb(name: str | None) -> str:
    """Normalize known aliases to values RagTool supports."""
    n = (name or "chromadb").lower()
    if n in {"chroma", "chromadb", "chroma-db"}:
        return "chromadb"
    if n in {"qdrant"}:
        return "qdrant"
    return "chromadb"


def _default_collection() -> str:
    return (
        os.getenv("A2A_CHROMA_COLLECTION")
        or os.getenv("A2A_QDRANT_COLLECTION")
        or "a2a-knowledge"
    )


def _redact(obj: Any) -> Any:
    """Redact secrets from nested dicts for safe logging."""
    if isinstance(obj, dict):
        S = {"api_key", "apikey", "key", "token", "password", "secret"}
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                out[k] = _redact(v)
            elif k.lower() in S:
                out[k] = "****"
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


def _call_query_like(fn, question: str, k: int, score: float):
    """
    Try a series of argument name variants that different RagTool/adapters use.
    Prefer calls WITHOUT thresholds first (some adapters misinterpret them).
    Returns the first non-empty list; otherwise returns the first successful result.
    """
    candidates: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [
        # threshold-free variants first
        ((), {"question": question, "k": k}),
        ((), {"query": question, "k": k}),
        ((), {"query": question, "top_k": k}),
        ((), {"question": question}),
        ((), {"query": question}),
        ((question,), {}),  # positional only
        # then variants with explicit thresholds
        ((), {"question": question, "limit": k, "similarity_threshold": score}),
        ((), {"query": question, "limit": k, "similarity_threshold": score}),
        ((), {"question": question, "k": k, "score_threshold": score}),
        ((), {"query": question, "k": k, "score_threshold": score}),
        ((), {"query": question, "top_k": k, "score_threshold": score}),
    ]
    last_exc: Optional[Exception] = None
    best_result: Any = None

    for args, kwargs in candidates:
        try:
            log.debug("RAG query dispatch try", extra={"args": list(args), "kwargs": _redact(kwargs)})
            res = fn(*args, **kwargs)
            if isinstance(res, list) and len(res) > 0:
                return res
            if best_result is None:
                best_result = res
        except TypeError as e:
            last_exc = e
        except Exception as e:
            last_exc = e

    if best_result is not None:
        return best_result
    if last_exc:
        raise last_exc
    raise AttributeError("No usable query-like callable found")


def _normalize_results(raw: Any) -> List[Dict[str, Any]]:
    """
    Normalize different return shapes into a common list[{text, score?, metadata?}].
    Be permissive to handle various adapters/libs.
    """
    # Already list of items
    if isinstance(raw, list):
        out: List[Dict[str, Any]] = []
        for item in raw:
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
                out.append({"text": text, "score": score, "metadata": meta})
            else:
                # Duck-typing for LangChain-style Document objects
                if hasattr(item, "page_content"):
                    try:
                        text = getattr(item, "page_content", "") or ""
                        meta = getattr(item, "metadata", {}) or {}
                        out.append({"text": text, "metadata": meta})
                        continue
                    except Exception:
                        pass
                out.append({"text": str(item), "metadata": {}})
        return out

    # Dict with nested arrays of results/documents/etc.
    if isinstance(raw, dict):
        # Chroma-style dict: lists-of-lists
        if all(k in raw for k in ("documents", "distances", "metadatas")):
            docs = (raw.get("documents") or [[]])[0]
            dists = (raw.get("distances") or [[]])[0]
            metas = (raw.get("metadatas") or [[]])[0]
            out: List[Dict[str, Any]] = []
            for doc, dist, meta in zip(docs, dists, metas):
                sim = 1.0 - float(dist) if dist is not None else None
                out.append({"text": doc, "score": sim, "metadata": meta or {}})
            return out

        for key in ("results", "documents", "chunks", "sources", "data"):
            if key in raw and isinstance(raw[key], list):
                return _normalize_results(raw[key])

        # plain answer string wrapped in dict
        if isinstance(raw.get("answer"), str):
            return [{"text": raw["answer"], "metadata": {}}]

        # Sometimes a single doc-like dict
        if "page_content" in raw or "content" in raw or "text" in raw:
            return _normalize_results([raw])

    # String answer
    if isinstance(raw, str):
        lower = raw.strip().lower()
        if "no relevant content found" in lower and "relevant content" in lower:
            return []
        return [{"text": raw, "metadata": {}}]

    # Fallback
    return [{"text": repr(raw), "metadata": {}}]


# ---------- low-dependency chunking (fallback path) ----------

def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # Fallback: binary read then decode best-effort
        try:
            return path.read_bytes().decode("utf-8", errors="ignore")
        except Exception:
            return ""


def _chunk_text(txt: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [txt]
    chunks: List[str] = []
    i = 0
    n = len(txt)
    safe_overlap = max(0, min(chunk_overlap, max(0, chunk_size - 1)))
    step = max(1, chunk_size - safe_overlap)
    while i < n:
        chunks.append(txt[i: i + chunk_size])
        i += step
    return chunks or [""]


def _hash_id(s: str) -> str:
    import hashlib as _hashlib
    return _hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _looks_like_maxlen_error(err: Exception) -> bool:
    """Detect Watsonx 'maximum sequence length' errors from the SDK/server."""
    s = (str(err) or "").lower()
    return (
        "maximum sequence length" in s
        or "exceeds the maximum sequence length" in s
        or "token sequence length" in s
    )


def _char_limit_for_model() -> int:
    """
    Very rough char limit guard to avoid 512-token overflows.
    Defaults are conservative for code/markdown (≈2 chars/token).
    Tunable via env if you know your model/tokenizer better.
    """
    max_toks = int(os.getenv("A2A_MAX_EMBED_TOKENS", "512"))
    chars_per_tok = float(os.getenv("A2A_CHARS_PER_TOKEN", "2.0"))
    return max(256, int(max_toks * chars_per_tok))


# ---------- helpers bound to a specific RagTool ----------

def _get_collection_from_rag(rag: RagTool) -> str:
    """Read effective collection name from rag.config; fall back to defaults."""
    try:
        cfg = getattr(rag, "config", {}) or {}
        vcfg = (cfg.get("vectordb") or {}).get("config") or {}
        name = vcfg.get("collection_name")
        if name:
            return str(name)
    except Exception:
        pass
    return _default_collection()


# ---------- Chroma + Watsonx fallback helpers ----------

def _effective_root() -> Path:
    return Path(os.getenv("A2A_KNOWLEDGE_DIR", "./.a2a_knowledge")).resolve()


def _vdb_conf(collection_name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    vdb = _norm_vdb(os.getenv("A2A_VDB"))
    conf: Dict[str, Any] = {}
    if vdb == "chromadb":
        root = _effective_root()
        conf = {
            "collection_name": collection_name or os.getenv("A2A_CHROMA_COLLECTION", "a2a-knowledge"),
            "path": str(root),
            "persist_directory": str(root),
        }
    elif vdb == "qdrant":
        conf = {
            "collection_name": collection_name or os.getenv("A2A_QDRANT_COLLECTION", "a2a-knowledge"),
            "url": os.getenv("A2A_QDRANT_URL", "http://localhost:6333"),
        }
    return vdb, conf


def _build_watsonx_embeddings():
    """Build a langchain-ibm WatsonxEmbeddings from env. Return None on failure."""
    try:
        from langchain_ibm import WatsonxEmbeddings
    except Exception as e:
        log.warning("Fallback disabled: langchain-ibm missing", extra={"error": str(e)})
        return None

    model_id = os.getenv("A2A_EMBEDDINGS_MODEL", "ibm/slate-125m-english-rtrvr")
    url = _env_first("EMBEDDINGS_WATSONX_URL", "WATSONX_URL", default="https://us-south.ml.cloud.ibm.com")
    apikey = _env_first("EMBEDDINGS_WATSONX_API_KEY", "WATSONX_API_KEY", "IBM_CLOUD_API_KEY")
    project_id = _env_first("EMBEDDINGS_WATSONX_PROJECT_ID", "WATSONX_PROJECT_ID", "PROJECT_ID", "IBM_CLOUD_PROJECT_ID")

    if not (apikey and project_id):
        log.warning("Fallback disabled: watsonx credentials not set")
        return None

    try:
        emb = WatsonxEmbeddings(model_id=model_id, url=url, apikey=apikey, project_id=project_id)
        return emb
    except Exception as e:
        log.warning("Fallback disabled: cannot init WatsonxEmbeddings", extra={"error": str(e)})
        return None


def _get_chroma_collection(collection_name: str, path: str):
    """Return (client, collection) or (None, None) if chroma not installed/unavailable."""
    try:
        import chromadb  # type: ignore
    except Exception as e:
        log.warning("Fallback disabled: chromadb not installed", extra={"error": str(e)})
        return None, None
    try:
        client = chromadb.PersistentClient(path=path)
        try:
            coll = client.get_collection(collection_name)
        except Exception:
            # Create if missing; set cosine space
            coll = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        return client, coll
    except Exception as e:
        log.warning("Fallback disabled: cannot open chroma collection", extra={"error": str(e)})
        return None, None


def _manual_ingest_with_watsonx(
    paths: List[Path],
    chunk_size: int,
    chunk_overlap: int,
    collection_name: Optional[str] = None,
    include_ext: Optional[List[str]] = None,
    exclude_ext: Optional[List[str]] = None,
) -> int:
    """
    Minimal ingest pipeline using WatsonxEmbeddings + Chroma (cosine).
    Returns number of chunks added (best effort).
    """
    vdb, conf = _vdb_conf(collection_name)
    if vdb != "chromadb":
        log.info("Manual ingest fallback supports Chroma only (skipping for vdb=%s)", vdb)
        return 0

    emb = _build_watsonx_embeddings()
    if emb is None:
        return 0

    _, coll = _get_chroma_collection(conf["collection_name"], conf["path"])
    if coll is None:
        return 0

    include = set(include_ext or _env_list("A2A_INCLUDE_EXT", ".md,.mdx,.py,.ipynb,.txt"))
    exclude = set(exclude_ext or _env_list("A2A_EXCLUDE_EXT", ".png,.jpg,.jpeg,.gif,.pdf"))

    def _allowed(p: Path) -> bool:
        ext = p.suffix.lower()
        return (ext in include) and (ext not in exclude)

    # Collect files
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            for q in p.rglob("*"):
                if q.is_file() and _allowed(q):
                    files.append(q)
        elif p.is_file():
            if _allowed(p):
                files.append(p)

    total_added = 0
    min_chunk_chars = int(os.getenv("A2A_MIN_CHUNK_CHARS", "256"))
    safe_char_limit = _char_limit_for_model()

    for f in files:
        txt = _read_text_file(f)
        if not txt:
            continue

        local_chunk_size = max(min_chunk_chars, min(chunk_size, safe_char_limit))
        attempts = 0

        while True:
            chunks = _chunk_text(txt, local_chunk_size, chunk_overlap)
            if not chunks:
                break
            try:
                vectors = emb.embed_documents(chunks)
            except Exception as e:
                # If tokens too long, shrink and retry; otherwise give up on this file
                if _looks_like_maxlen_error(e) and local_chunk_size > min_chunk_chars:
                    prev = local_chunk_size
                    # shrink aggressively but keep reasonable minimum
                    local_chunk_size = max(min_chunk_chars, int(local_chunk_size * 0.6))
                    attempts += 1
                    log.warning(
                        "Manual ingest: chunk too long for embeddings, shrinking",
                        extra={
                            "file": str(f),
                            "prev_chunk_size": prev,
                            "new_chunk_size": local_chunk_size,
                            "attempt": attempts,
                        },
                    )
                    continue
                log.warning("Manual ingest: embeddings failed for file", extra={"file": str(f), "error": str(e)})
                vectors = None
            # success path
            if vectors is None:
                break

            # Stable, content-aware IDs
            try:
                st = f.stat()
                file_sig = f"{str(f.resolve())}::{st.st_size}::{st.st_mtime_ns}"
            except Exception:
                file_sig = str(f.resolve())
            fid = _hash_id(file_sig)
            ids = [f"{fid}:{i}:{_hash_id(chunks[i])}" for i in range(len(chunks))]
            metadatas = [
                {"source": str(f), "chunk_index": i, "ingest": "fallback-watsonx"}
                for i in range(len(chunks))
            ]
            try:
                if hasattr(coll, "upsert"):
                    coll.upsert(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)  # type: ignore[attr-defined]
                else:
                    try:
                        coll.delete(ids=ids)
                    except Exception:
                        pass
                    coll.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)
                total_added += len(chunks)
            except Exception as e:
                log.warning("Manual ingest: chroma add failed", extra={"file": str(f), "error": str(e)})
            break  # done with this file (we successfully embedded at this size)

    log.info("Manual ingest added chunks", extra={"count": total_added})
    return total_added


def _manual_query_with_watsonx(
    q: str,
    k: int,
    score_threshold: float,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Minimal query using WatsonxEmbeddings + Chroma (cosine).
    Returns normalized list of results.
    """
    vdb, conf = _vdb_conf(collection_name)
    if vdb != "chromadb":
        return []

    emb = _build_watsonx_embeddings()
    if emb is None:
        return []

    _, coll = _get_chroma_collection(conf["collection_name"], conf["path"])
    if coll is None:
        return []

    # Guard against overly long queries too
    q_safe = q
    char_limit = _char_limit_for_model()
    if len(q_safe) > char_limit:
        log.debug("Query truncated for embeddings limit", extra={"orig_len": len(q), "new_len": char_limit})
        q_safe = q_safe[:char_limit]

    try:
        qvec = emb.embed_query(q_safe)
    except Exception as e:
        log.warning("Manual query: embed_query failed", extra={"error": str(e)})
        return []

    try:
        res = coll.query(query_embeddings=[qvec], n_results=max(1, k))
    except Exception as e:
        log.warning("Manual query: chroma query failed", extra={"error": str(e)})
        return []

    # Chroma returns dict with keys: ids, distances, documents, metadatas
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for doc, dist, meta in zip(docs, dists, metas):
        # cosine distance -> similarity
        sim = 1.0 - float(dist)
        if score_threshold is not None and sim < score_threshold:
            continue
        out.append({"text": doc, "score": sim, "metadata": meta or {}})

    log.info("Manual query results", extra={"n": len(out)})
    return out


# ---------- factory & accessor ----------

def make_rag(collection_name: Optional[str] = None) -> RagTool:
    """
    Create a RagTool configured by env (vector DB + embeddings).
    Compatible with crewai-tools variants that expect either:
      - "embedding_model": {...}
      - "embeddings": {...}
    """
    coll = collection_name or _default_collection()

    root = Path(os.getenv("A2A_KNOWLEDGE_DIR", "./.a2a_knowledge")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    vdb = _norm_vdb(os.getenv("A2A_VDB"))
    cfg: Dict[str, Any] = {"vectordb": {"provider": vdb, "config": {}}}

    if vdb == "chromadb":
        # Support both keys: some adapters expect "path", others "persist_directory"
        cfg["vectordb"]["config"] = {
            "path": str(root),
            "persist_directory": str(root),
            "collection_name": coll,
        }
    elif vdb == "qdrant":
        cfg["vectordb"]["config"] = {
            "collection_name": coll,
            "url": os.getenv("A2A_QDRANT_URL", "http://localhost:6333"),
        }

    # Embedding model config (optional but strongly recommended)
    emb_provider = os.getenv("A2A_EMBEDDINGS_PROVIDER")
    emb_model = os.getenv("A2A_EMBEDDINGS_MODEL")
    if emb_provider:
        prov = emb_provider.lower()
        emb_cfg: Dict[str, Any] = {}
        if emb_model:
            emb_cfg["model_id"] = emb_model

        if prov == "watsonx":
            # Preflight check for IBM SDK presence
            try:
                import ibm_watsonx_ai  # noqa: F401
            except Exception as e:
                log.error("watsonx embeddings requested but ibm-watsonx-ai is missing")
                raise RuntimeError(
                    "A2A_EMBEDDINGS_PROVIDER=watsonx requires 'ibm-watsonx-ai'. Install it, then restart."
                ) from e
            emb_cfg.update(
                {
                    "url": _env_first("EMBEDDINGS_WATSONX_URL", "WATSONX_URL"),
                    # IMPORTANT: 'apikey' (not 'api_key') to match WatsonxEmbeddings
                    "apikey": _env_first(
                        "EMBEDDINGS_WATSONX_API_KEY",
                        "WATSONX_API_KEY",
                        "IBM_CLOUD_API_KEY",
                    ),
                    "project_id": _env_first(
                        "EMBEDDINGS_WATSONX_PROJECT_ID",
                        "WATSONX_PROJECT_ID",
                        "PROJECT_ID",
                        "IBM_CLOUD_PROJECT_ID",
                    ),
                }
            )
        elif prov == "openai":
            if emb_model and "model" not in emb_cfg:
                emb_cfg["model"] = emb_model
            emb_cfg["api_key"] = os.getenv("OPENAI_API_KEY")

        # Supply BOTH keys for compatibility with different RagTool versions
        cfg["embedding_model"] = {"provider": prov, "config": emb_cfg}
        cfg["embeddings"] = {"provider": prov, "config": emb_cfg}

    log.info("RAG init", extra={"index_dir": str(root), "config": _redact(cfg)})

    rag = RagTool(
        config=cfg or None,
        name=f"A2A Knowledge ({coll})",
        description="Repo-scale retrieval for large Markdown and code.",
    )
    # NOTE: Do NOT setattr unknown fields on RagTool (Pydantic forbids it).
    # We derive collection/path later from rag.config or env.
    return rag


def get_rag(collection_name: Optional[str] = None) -> RagTool:
    """
    Return a cached RagTool for the given collection (or default collection).
    """
    coll = collection_name or _default_collection()
    if coll not in _RAG_CACHE:
        _RAG_CACHE[coll] = make_rag(coll)
    return _RAG_CACHE[coll]


# ---------- operations used by the FastAPI router ----------

def ingest_paths(
    rag: RagTool,
    paths: List[str],
    include_ext: Optional[List[str]] = None,
    exclude_ext: Optional[List[str]] = None,
    chunk_size: int = 1400,
    chunk_overlap: int = 160,
):
    include = include_ext or _env_list("A2A_INCLUDE_EXT", ".md,.mdx,.py,.ipynb,.txt")
    exclude = exclude_ext or _env_list("A2A_EXCLUDE_EXT", ".png,.jpg,.jpeg,.gif,.pdf")

    log.info(
        "Ingest start",
        extra={
            "paths": paths,
            "include": include,
            "exclude": exclude,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "collection": _get_collection_from_rag(rag),
        },
    )

    def _do_add(p: Path):
        add_fn = getattr(rag, "add", None) or getattr(getattr(rag, "app", None), "add", None)
        if not callable(add_fn):
            log.error("RagTool has no usable add()")
            raise AttributeError("RagTool has no 'add' method (nor app.add).")
        try:
            exists = p.exists()
            log.debug(
                "Ingest path",
                extra={
                    "path": str(p),
                    "exists": exists,
                    "is_dir": p.is_dir() if exists else None,
                    "size": (p.stat().st_size if exists and p.is_file() else None),
                },
            )
            # Fail fast if the server cannot see the path (common Docker/host mismatch)
            if not exists:
                raise FileNotFoundError(f"Ingest path not found on server filesystem: {p}")

            # Optional: enforce ingests under knowledge dir
            if os.getenv("A2A_STRICT_INGEST_ROOT", "0") == "1":
                base = _effective_root()
                pres = p.resolve()
                if base not in pres.parents and pres != base:
                    raise PermissionError(f"Refusing to ingest outside knowledge dir: {p}")

            return add_fn(
                data_type="directory" if p.is_dir() else "file",
                path=str(p),
                include_extensions=include,
                exclude_extensions=exclude,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception:
            log.exception("Ingest add failed", extra={"path": str(p)})
            raise

    # Pre-measure collection count (Chroma) to detect no-op ingests
    coll_name = _get_collection_from_rag(rag)
    vdb, vconf = _vdb_conf(coll_name)
    before_count: Optional[int] = None
    coll_obj = None
    if vdb == "chromadb":
        _, coll_obj = _get_chroma_collection(vconf["collection_name"], vconf["path"])
        if coll_obj:
            try:
                before_count = coll_obj.count()
            except Exception:
                before_count = None

    # 1) Try the tool's native add()
    for raw in paths:
        _do_add(Path(raw))

    # 2) Only run fallback if enabled AND the collection didn't grow (or we couldn't measure)
    run_fallback = False
    after_count: Optional[int] = None
    if vdb == "chromadb" and coll_obj:
        try:
            after_count = coll_obj.count()
        except Exception:
            after_count = None

    if os.getenv("A2A_RAG_FALLBACK_LC", "1") == "1":
        if before_count is None or after_count is None or after_count == before_count:
            run_fallback = True

    if run_fallback and vdb == "chromadb":
        added = _manual_ingest_with_watsonx(
            [Path(p) for p in paths],
            chunk_size,
            chunk_overlap,
            coll_name,
            include_ext=include,
            exclude_ext=exclude,
        )
        if added > 0:
            log.info("Fallback ingest succeeded", extra={"chunks_added": added, "collection": coll_name})
        else:
            log.warning("Fallback ingest added 0 chunks", extra={"collection": coll_name})

    log.info("Ingest done", extra={"count": len(paths)})
    return {
        "indexed": paths,
        "include": include,
        "exclude": exclude,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def query(
    rag: RagTool,
    q: str,
    k: int = 8,
    score_threshold: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Query the index using whatever interface this RagTool version provides.
    Returns a normalized list[{text, score?, metadata?}].
    """
    coll = _get_collection_from_rag(rag)
    log.info(
        "Query start",
        extra={"q_preview": q[:160], "k": k, "score_threshold": score_threshold, "q_len": len(q), "collection": coll},
    )
    raw = None
    results: List[Dict[str, Any]] = []
    try:
        if callable(getattr(rag, "query", None)):
            raw = _call_query_like(rag.query, q, k, score_threshold)
        elif callable(getattr(rag, "run", None)):
            raw = _call_query_like(rag.run, q, k, score_threshold)
        elif callable(getattr(rag, "_run", None)):
            raw = _call_query_like(rag._run, q, k, score_threshold)
        else:
            adapter = getattr(rag, "adapter", None)
            app = getattr(rag, "app", None)
            if adapter and callable(getattr(adapter, "query", None)):
                raw = _call_query_like(adapter.query, q, k, score_threshold)
            elif app and callable(getattr(app, "query", None)):
                raw = _call_query_like(app.query, q, k, score_threshold)
            else:
                log.error("No query-capable interface on RagTool")
                raise AttributeError("No query-capable interface found on RagTool.")
        results = _normalize_results(raw)
    except Exception:
        log.exception("Query execution failed (tool path)")
        results = []

    # If tool path returned nothing, try the manual Watsonx+Chroma query
    if not results and os.getenv("A2A_RAG_FALLBACK_LC", "1") == "1":
        log.info("Attempting manual query fallback (watsonx+chroma)", extra={"collection": coll})
        try:
            results = _manual_query_with_watsonx(q, k, score_threshold, coll)
        except Exception:
            log.exception("Manual query fallback failed")
            results = []

    try:
        log.info("Query done", extra={"raw_type": str(type(raw)), "results": len(results), "collection": coll})
        if results:
            top = results[0]
            preview = (top.get("text") or "")[:200]
            meta = _redact(top.get("metadata") or {})
            log.debug("Top result", extra={"text_preview": preview, "metadata": meta})
        else:
            log.warning(
                "Query returned zero results",
                extra={"hint": "ensure embeddings; verify path visibility; fallback enabled; re-ingest", "collection": coll},
            )
        return results
    except Exception:
        log.exception("Result normalization failed")
        raise


def reset(rag: RagTool):
    app = getattr(rag, "app", None)
    adapter = getattr(rag, "adapter", None)
    if callable(getattr(rag, "reset", None)):
        rag.reset()
    elif app and callable(getattr(app, "reset", None)):
        app.reset()  # type: ignore[attr-defined]
    elif adapter and callable(getattr(adapter, "reset", None)):
        adapter.reset()  # type: ignore[attr-defined]
    else:
        log.warning("No reset() available on RagTool / app / adapter")
    log.info("Index reset requested")
    return {"reset": True}


def stats(rag: RagTool) -> Dict[str, Any]:
    """
    Lightweight stats + config summary (redacted).
    Includes Chroma collection count if available.
    """
    index_dir = os.getenv("A2A_KNOWLEDGE_DIR", "./.a2a_knowledge")
    root = Path(index_dir).resolve()

    cfg = getattr(rag, "config", None)
    vdb_provider, vdb_conf = "chromadb", {}
    emb_provider, emb_model = None, None

    if isinstance(cfg, dict):
        vdb_cfg = cfg.get("vectordb", {}) if isinstance(cfg.get("vectordb"), dict) else {}
        vdb_provider = vdb_cfg.get("provider", "chromadb")
        vdb_conf = vdb_cfg.get("config", {}) if isinstance(vdb_cfg.get("config"), dict) else {}

        # Accept both "embedding_model" and "embeddings"
        emb_section = None
        if isinstance(cfg.get("embedding_model"), dict):
            emb_section = cfg.get("embedding_model")
        elif isinstance(cfg.get("embeddings"), dict):
            emb_section = cfg.get("embeddings")

        if isinstance(emb_section, dict):
            emb_provider = emb_section.get("provider")
            emb_conf = emb_section.get("config", {}) if isinstance(emb_section.get("config"), dict) else {}
            emb_model = emb_conf.get("model_id") or emb_conf.get("model")

    # basic on-disk info (for Chroma)
    file_count = 0
    byte_size = 0
    if vdb_provider == "chromadb" and root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                file_count += 1
                try:
                    byte_size += p.stat().st_size
                except OSError:
                    pass

    # Compute collection name from rag to ensure stats align with active collection
    collection_name = _get_collection_from_rag(rag)

    # Try to expose Chroma collection count (very helpful after ingest)
    collection_count: Optional[int] = None
    # Choose effective path (either 'path' or 'persist_directory')
    chroma_path = (
        (vdb_conf.get("path") if isinstance(vdb_conf, dict) else None)
        or (vdb_conf.get("persist_directory") if isinstance(vdb_conf, dict) else None)
        or str(root)
    )
    if vdb_provider == "chromadb":
        try:
            import chromadb  # type: ignore
            client = chromadb.PersistentClient(path=chroma_path)
            try:
                coll = client.get_collection(collection_name)
                collection_count = coll.count()
            except Exception:
                collection_count = 0
        except Exception:
            collection_count = None

    summary = {
        "index_dir": str(root),
        "vectordb": {
            "provider": vdb_provider,
            "collection_name": collection_name,
        },
        "embeddings": {
            "provider": emb_provider,
            "model": emb_model,
            "configured": bool(emb_provider),
        },
        "storage": {"file_count": file_count, "bytes": byte_size},
        "collection": {"name": collection_name, "count": collection_count},
    }
    log.debug("Stats", extra=_redact(summary))
    return summary
