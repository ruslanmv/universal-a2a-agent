# syntax=docker/dockerfile:1.7
################################################
#  Universal A2A Agent — Production Dockerfile #
#  Multi-stage, non-root, healthcheck, Gunicorn #
################################################

##########
#  STAGE 1: Builder — install deps & package
##########
FROM python:3.11-slim AS builder

# Avoid prompts & keep images small
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# System deps (uncomment if you need compilation)
# RUN apt-get update -y && apt-get install -y --no-install-recommends \
#     build-essential ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# Copy project metadata first to leverage Docker layer caching
COPY pyproject.toml README.md ./
COPY src ./src

# Optional build arg to choose extras (e.g., "all", "langgraph", "langchain")
ARG EXTRAS=all

# Install project + runtime server.
# The problematic 'compileall' command has been removed.
RUN python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install --no-cache-dir "gunicorn>=22.0" && \
    if [ -n "$EXTRAS" ]; then \
        python -m pip install --no-cache-dir ".[$EXTRAS]"; \
    else \
        python -m pip install --no-cache-dir .; \
    fi

##########
#  STAGE 2: Runtime — minimal, secure, non-root
##########
FROM python:3.11-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    \
    # App tuning knobs (override at runtime)
    PORT=8000 \
    WORKERS=2 \
    TIMEOUT=60 \
    KEEP_ALIVE=5 \
    LOG_LEVEL=info \
    \
    # App env defaults (override in Helm/compose)
    A2A_HOST=0.0.0.0 \
    A2A_PORT=8000 \
    PUBLIC_URL=http://localhost:8000 \
    \
    # Provider/Framework selection
    LLM_PROVIDER=echo \
    AGENT_FRAMEWORK=langgraph

# Create non-root user and app directory
RUN useradd -r -u 10001 -s /usr/sbin/nologin appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy installed site-packages & binaries from builder
COPY --from=builder /usr/local /usr/local

# --- HEALTHCHECK ---
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os, sys, urllib.request; \
  u=f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz'; \
  try: \
      with urllib.request.urlopen(u, timeout=2) as r: \
          ok = (r.status == 200); \
          sys.exit(0 if ok else 1); \
  except Exception: \
      sys.exit(1)"

# Drop privileges
USER appuser

EXPOSE 8000

# Optional: label metadata
LABEL org.opencontainers.image.title="Universal A2A Agent" \
      org.opencontainers.image.description="Framework-agnostic A2A service (FastAPI) with adapters for LangGraph, LangChain, CrewAI, AutoGen, Bee/BeeAI, MCP, and OpenAI-compatible endpoint." \
      org.opencontainers.image.licenses="Apache-2.0"

# Entrypoint script for flexible runtime tuning via env vars
COPY --chown=appuser:appuser <<'EOF' /app/entrypoint.sh
#!/bin/sh
set -e

# Use environment variables with sensible, POSIX-compliant defaults
PORT=${PORT:-8000}
WORKERS=${WORKERS:-2}
TIMEOUT=${TIMEOUT:-60}
KEEP_ALIVE=${KEEP_ALIVE:-5}
LOG_LEVEL=${LOG_LEVEL:-info}

# Show startup context (useful in k8s logs)
echo "[entrypoint] Starting Universal A2A on :$PORT (workers=$WORKERS, timeout=$TIMEOUT, keep-alive=$KEEP_ALIVE, log=$LOG_LEVEL)"
echo "[entrypoint] Provider=$LLM_PROVIDER Framework=$AGENT_FRAMEWORK PublicURL=$PUBLIC_URL"

# Execute Gunicorn with the configured parameters
exec gunicorn \
    -k uvicorn.workers.UvicornWorker \
    -w "${WORKERS}" \
    -b "0.0.0.0:${PORT}" \
    --timeout "${TIMEOUT}" \
    --keep-alive "${KEEP_ALIVE}" \
    --log-level "${LOG_LEVEL}" \
    --access-logfile - \
    --error-logfile - \
    a2a_universal.server:app
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]