#!/usr/bin/env bash
# scripts/dev-up.sh
# Spin up the Universal A2A Agent locally with Docker Compose (dev-friendly).
# You can override any env inline, e.g.:
#   HOST_PORT=8080 LLM_PROVIDER=openai OPENAI_API_KEY=sk-... ./scripts/dev-up.sh
set -euo pipefail

# -----------------------------
# Helpers
# -----------------------------
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "FATAL: '$1' is required but not installed."; exit 1; }
}

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
  elif docker-compose version >/dev/null 2>&1; then
    echo "docker-compose"
  else
    echo "FATAL: Docker Compose not found. Install Docker Desktop or docker-compose." >&2
    exit 1
  fi
}

wait_for_health() {
  local url=$1
  local attempts=${2:-50}
  local sleep_sec=${3:-0.2}
  echo "Waiting for health: $url"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "Service is healthy ✅"
      return 0
    fi
    sleep "$sleep_sec"
  done
  echo "Service failed to become healthy ❌"
  return 1
}

# -----------------------------
# Defaults (override inline)
# -----------------------------
export IMAGE_REPO=${IMAGE_REPO:-yourrepo/universal-a2a-agent}
export IMAGE_TAG=${IMAGE_TAG:-1.2.0}
export EXTRAS=${EXTRAS:-all}

# Gunicorn/Uvicorn & ports
export HOST_PORT=${HOST_PORT:-8000}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-2}
export TIMEOUT=${TIMEOUT:-60}
export KEEP_ALIVE=${KEEP_ALIVE:-5}
export LOG_LEVEL=${LOG_LEVEL:-info}

# Public URL (used by agent card)
export PUBLIC_URL=${PUBLIC_URL:-http://localhost:${HOST_PORT}}

# Provider/Framework selection (swap freely)
export LLM_PROVIDER=${LLM_PROVIDER:-echo}          # echo|openai|watsonx|ollama|anthropic|gemini|azure_openai|bedrock
export AGENT_FRAMEWORK=${AGENT_FRAMEWORK:-langgraph} # langgraph|crewai|autogen|beeai|langchain

# Provider credentials (examples)
export OPENAI_API_KEY=${OPENAI_API_KEY:-}
export WATSONX_API_KEY=${WATSONX_API_KEY:-}
export WATSONX_URL=${WATSONX_URL:-}
export WATSONX_PROJECT_ID=${WATSONX_PROJECT_ID:-}
export OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://host.docker.internal:11434}
export OLLAMA_MODEL=${OLLAMA_MODEL:-llama3}

# Enterprise adapter (off by default)
export PRIVATE_ADAPTER_ENABLED=${PRIVATE_ADAPTER_ENABLED:-false}
export PRIVATE_ADAPTER_AUTH_SCHEME=${PRIVATE_ADAPTER_AUTH_SCHEME:-NONE}
export PRIVATE_ADAPTER_AUTH_TOKEN=${PRIVATE_ADAPTER_AUTH_TOKEN:-}
export PRIVATE_ADAPTER_INPUT_KEY=${PRIVATE_ADAPTER_INPUT_KEY:-input}
export PRIVATE_ADAPTER_OUTPUT_KEY=${PRIVATE_ADAPTER_OUTPUT_KEY:-output}
export PRIVATE_ADAPTER_TRACE_KEY=${PRIVATE_ADAPTER_TRACE_KEY:-traceId}
export PRIVATE_ADAPTER_PATH=${PRIVATE_ADAPTER_PATH:-/enterprise/v1/agent}

# -----------------------------
# Checks
# -----------------------------
require_cmd docker
COMPOSE_BIN=$(detect_compose)

echo "== Universal A2A Agent :: dev up =="
echo "Image:   ${IMAGE_REPO}:${IMAGE_TAG}"
echo "Extras:  ${EXTRAS}"
echo "Ports:   host ${HOST_PORT} -> container ${PORT}"
echo "Runtime: workers=${WORKERS} timeout=${TIMEOUT}s keep-alive=${KEEP_ALIVE}s log=${LOG_LEVEL}"
echo "Public:  ${PUBLIC_URL}"
echo "Choice:  provider=${LLM_PROVIDER} framework=${AGENT_FRAMEWORK}"

# -----------------------------
# Up
# -----------------------------
${COMPOSE_BIN} up -d --build

# -----------------------------
# Health
# -----------------------------
HEALTH_URL="http://localhost:${HOST_PORT}/healthz"
wait_for_health "${HEALTH_URL}" 60 0.2

echo ""
echo "🌐 Service endpoints"
echo "  Health:      ${HEALTH_URL}"
echo "  Readiness:   http://localhost:${HOST_PORT}/readyz"
echo "  Agent Card:  http://localhost:${HOST_PORT}/.well-known/agent-card.json"
echo "  OpenAPI:     http://localhost:${HOST_PORT}/openapi.json"
echo "  Swagger UI:  http://localhost:${HOST_PORT}/docs"
echo ""
echo "🔎 Quick checks"
echo "  curl -s http://localhost:${HOST_PORT}/readyz | jq"
echo "  curl -s http://localhost:${HOST_PORT}/a2a -H 'Content-Type: application/json' -d '{\"method\":\"message/send\",\"params\":{\"message\":{\"role\":\"user\",\"messageId\":\"m1\",\"parts\":[{\"type\":\"text\",\"text\":\"ping\"}]}}}' | jq"
echo "  curl -s http://localhost:${HOST_PORT}/openai/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"universal-a2a-hello\",\"messages\":[{\"role\":\"user\",\"content\":\"ping from compose\"}]}' | jq"
echo ""
echo "✅ Service running at: http://localhost:${HOST_PORT}"
