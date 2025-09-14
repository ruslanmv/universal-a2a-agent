#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------
# Universal A2A Agent — multi-arch builder
# Usage:
#   ./scripts/build-containers.sh                 # build local (load into Docker)
#   PUSH=true ./scripts/build-containers.sh       # buildx + push to registry
#   IMAGE_REPO=ghcr.io/you/a2a IMAGE_TAG=1.2.0 ./scripts/build-containers.sh
#   EXTRAS=all PLATFORMS=linux/amd64,linux/arm64 PUSH=true ./scripts/build-containers.sh
# -------------------------------------------

IMAGE_REPO=${IMAGE_REPO:-yourrepo/universal-a2a-agent}
IMAGE_TAG=${IMAGE_TAG:-1.2.0}
EXTRAS=${EXTRAS:-all}
PLATFORMS=${PLATFORMS:-linux/amd64}
PUSH=${PUSH:-false}
BUILDER=${BUILDER:-a2a_builder}

# Ensure buildx builder exists
if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  docker buildx create --name "$BUILDER" --use >/dev/null
else
  docker buildx use "$BUILDER" >/dev/null
fi

echo ":: Building ${IMAGE_REPO}:${IMAGE_TAG} (extras=${EXTRAS}; platforms=${PLATFORMS}; push=${PUSH})"

# Build flags
if [[ "$PUSH" == "true" ]]; then
  PUBLISH_FLAG=(--push)
else
  PUBLISH_FLAG=(--load)
  # For --load, Docker only supports a single platform; ensure that's set
  if [[ "$PLATFORMS" != "linux/amd64" && "$PLATFORMS" != "linux/arm64" ]]; then
    echo "[warn] --load supports one platform; overriding PLATFORMS=linux/amd64"
    PLATFORMS=linux/amd64
  fi
fi

DOCKER_BUILDKIT=1 docker buildx build \
  --platform "$PLATFORMS" \
  -t "${IMAGE_REPO}:${IMAGE_TAG}" \
  --build-arg EXTRAS="$EXTRAS" \
  "${PUBLISH_FLAG[@]}" \
  .

echo ":: Done. Image: ${IMAGE_REPO}:${IMAGE_TAG}"

# Optional convenience tags
if [[ "${LATEST:-false}" == "true" ]]; then
  if [[ "$PUSH" == "true" ]]; then
    docker buildx imagetools create "${IMAGE_REPO}:${IMAGE_TAG}" -t "${IMAGE_REPO}:latest"
  else
    docker tag "${IMAGE_REPO}:${IMAGE_TAG}" "${IMAGE_REPO}:latest"
  fi
  echo ":: Tagged latest"
fi