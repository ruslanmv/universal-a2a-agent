# src/a2a_universal/routers/knowledge.py
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..ext.crew_rag import (
    get_rag,  # per-collection RagTool (cached)
    ingest_paths,
    query as rag_query,
    reset as rag_reset,
    stats as rag_stats,
)

log = logging.getLogger("a2a.knowledge.api")
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _ensure_enabled() -> None:
    if os.getenv("A2A_ENABLE_KNOWLEDGE", "0") != "1":
        raise HTTPException(
            status_code=503, detail="Knowledge disabled (set A2A_ENABLE_KNOWLEDGE=1)."
        )


class IngestReq(BaseModel):
    paths: List[str]
    include_ext: Optional[List[str]] = None
    exclude_ext: Optional[List[str]] = None
    chunk_size: int = Field(default=int(os.getenv("A2A_CHUNK_SIZE", "1400")))
    chunk_overlap: int = Field(default=int(os.getenv("A2A_CHUNK_OVERLAP", "160")))
    # NEW: target a specific vector collection (otherwise defaults are used)
    collection: Optional[str] = None


class QueryReq(BaseModel):
    q: str
    k: int = 8
    score_threshold: float = 0.65
    # NEW: target a specific vector collection for this query
    collection: Optional[str] = None


@router.post("/ingest")
def ingest(req: IngestReq):
    """
    Ingest one or more filesystem paths into the knowledge index.
    Optional: req.collection chooses a specific vector collection.
    """
    _ensure_enabled()
    rag = get_rag(req.collection)
    log.info(
        "HTTP ingest",
        extra={
            "n_paths": len(req.paths),
            "chunk_size": req.chunk_size,
            "chunk_overlap": req.chunk_overlap,
            "collection": getattr(rag, "_collection_name", None) or req.collection,
        },
    )
    try:
        resp = ingest_paths(
            rag,
            req.paths,
            req.include_ext,
            req.exclude_ext,
            req.chunk_size,
            req.chunk_overlap,
        )
        log.info(
            "HTTP ingest done",
            extra={
                "indexed": len(resp.get("indexed", [])),
                "collection": getattr(rag, "_collection_name", None),
            },
        )
        return resp
    except Exception as e:
        log.exception("HTTP ingest error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
def query(req: QueryReq):
    """
    Query the knowledge index.
    Optional: req.collection chooses which collection to query.
    """
    _ensure_enabled()
    rag = get_rag(req.collection)
    log.info(
        "HTTP query",
        extra={
            "q_preview": req.q[:160],
            "k": req.k,
            "score_threshold": req.score_threshold,
            "collection": getattr(rag, "_collection_name", None) or req.collection,
        },
    )
    try:
        results = rag_query(rag, req.q, req.k, req.score_threshold)
        log.info("HTTP query done", extra={"results": len(results)})
        if not results:
            log.warning("HTTP query returned zero results")
        return {"results": results}
    except Exception as e:
        log.exception("HTTP query error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
def reset(
    collection: Optional[str] = Query(
        default=None, description="Optional collection to reset"
    )
):
    """
    Reset the knowledge index for the specified collection (or default).
    """
    _ensure_enabled()
    rag = get_rag(collection)
    log.info(
        "HTTP reset",
        extra={"collection": getattr(rag, "_collection_name", None) or collection},
    )
    try:
        return rag_reset(rag)
    except Exception as e:
        log.exception("HTTP reset error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
def stats(
    collection: Optional[str] = Query(
        default=None, description="Optional collection to inspect"
    )
):
    """
    Get knowledge index stats for the specified collection (or default).
    """
    _ensure_enabled()
    rag = get_rag(collection)
    log.debug(
        "HTTP stats",
        extra={"collection": getattr(rag, "_collection_name", None) or collection},
    )
    try:
        out = rag_stats(rag)
        # If a specific collection was requested, reflect it explicitly
        try:
            if collection:
                out.setdefault("collection", {})
                out["collection"]["name"] = collection
        except Exception:
            pass
        return out
    except Exception as e:
        log.exception("HTTP stats error")
        raise HTTPException(status_code=500, detail=str(e))
