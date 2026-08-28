#!/usr/bin/env bash
# =============================================================================
# Build and push MRI Preprocessing image to self-hosted registry
# =============================================================================
#
# Usage:
#   bash scripts/build_and_push.sh [tag]
#
# Examples:
#   bash scripts/build_and_push.sh              # tags as latest + git short hash
#   bash scripts/build_and_push.sh v1.0.0        # exact tag
#
# Requires:
#   - Docker installed and running
#   - Registry credentials in ~/.docker/config.json (run `docker login <registry>`)
#     OR set REGISTRY_USER / REGISTRY_PASS env vars before running
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")

# ── Configuration (override in .env or via env vars) ─────────────
ENV_FILE="${PROJECT_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
  while IFS='=' read -r key value || [ -n "$key" ]; do
    key="${key//[[:space:]]/}"
    case "$key" in ''|\#*) continue ;; esac
    export "${key}=${value:-}"
  done < "$ENV_FILE"
fi
REGISTRY="${REGISTRY_URL:-registry.forgejo.local:5000}"
REPO="${IMAGE_REPOSITORY:-mri_preprocessing}"
TAG="${1:-latest}"
FULL_NAME="${REGISTRY}/${REPO}:${TAG}"

DOCKERFILE="${PROJECT_ROOT}/control_system/dockerfile"

echo "Registry : ${REGISTRY}"
echo "Repo     : ${REPO}"
echo "Tag      : ${TAG}"
echo "Image    : ${FULL_NAME}"
echo ""

# ── Authenticate if credentials provided ───────────────────────────
if [ -n "${REGISTRY_USER:-}" ] && [ -n "${REGISTRY_PASS:-}" ]; then
  echo "Authenticating with registry..."
  echo "${REGISTRY_PASS}" | docker login -u "${REGISTRY_USER}" --password-stdin "${REGISTRY}"
fi

# ── Build ──────────────────────────────────────────────────────────
echo ""
echo "Building image..."
docker build \
  --tag "${FULL_NAME}" \
  --file "${DOCKERFILE}" \
  --progress=plain \
  "${PROJECT_ROOT}"

# ── Tag with git commit hash for traceability ──────────────────────
GIT_HASH=$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
if [ "$TAG" != "latest" ]; then
  docker tag "${FULL_NAME}" "${REGISTRY}/${REPO}:${GIT_HASH}"
fi

# ── Push ───────────────────────────────────────────────────────────
echo ""
echo "Pushing image..."
docker push "${FULL_NAME}"

if [ "$TAG" != "latest" ]; then
  docker push "${REGISTRY}/${REPO}:${GIT_HASH}"
  echo "  Pushed: ${REGISTRY}/${REPO}:${GIT_HASH}"
fi

# ── Verify on registry (optional dry-run check) ───────────────────
echo ""
echo "Build and push complete."
echo "On your HPC, pull the image with:"
echo "  singularity pull mri_preprocessing.sif docker://${FULL_NAME}"
echo ""
echo "Or use start_control.sh with REGISTRY_URL=${REGISTRY}"
