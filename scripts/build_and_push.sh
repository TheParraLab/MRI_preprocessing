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
# Two variants per release:
#   <tag>      → CPU-only reg_f3d (default, runs on any node)
#   <tag>-gpu  → CUDA-linked reg_f3d (GPU nodes only)
CPU_NAME="${REGISTRY}/${REPO}:${TAG}"
GPU_NAME="${REGISTRY}/${REPO}:${TAG}-gpu"

DOCKERFILE="${PROJECT_ROOT}/control_system/dockerfile"

echo "Registry : ${REGISTRY}"
echo "Repo     : ${REPO}"
echo "Tag      : ${TAG}"
echo "Images   : ${CPU_NAME} (CPU)"
echo "           ${GPU_NAME} (GPU)"
echo ""

# ── Authenticate if credentials provided ───────────────────────────
if [ -n "${REGISTRY_USER:-}" ] && [ -n "${REGISTRY_PASS:-}" ]; then
  echo "Authenticating with registry..."
  echo "${REGISTRY_PASS}" | docker login -u "${REGISTRY_USER}" --password-stdin "${REGISTRY}"
fi

# ── Build + push one variant ───────────────────────────────────────
# $1 = USE_CUDA (OFF|ON), $2 = tag suffix ("", "-gpu")
build_and_push_variant() {
  local use_cuda="$1" tag_suffix="$2"
  local tag="${TAG}${tag_suffix}"
  local full_name="${REGISTRY}/${REPO}:${tag}"

  echo ""
  echo "Building image: ${full_name} (USE_CUDA=${use_cuda})..."
  docker build \
    --tag "${full_name}" \
    --file "${DOCKERFILE}" \
    --progress=plain \
    --build-arg USE_CUDA="${use_cuda}" \
    --build-arg VCS_REVISION="${GIT_HASH}" \
    "${PROJECT_ROOT}"

  echo ""
  echo "Pushing ${full_name}..."
  docker push "${full_name}"

  # Tag with git commit hash for traceability (release tags only)
  if [ "$TAG" != "latest" ]; then
    local hash_name="${REGISTRY}/${REPO}:${GIT_HASH}${tag_suffix}"
    docker tag "${full_name}" "${hash_name}"
    docker push "${hash_name}"
    echo "  Pushed: ${hash_name}"
  fi
}

GIT_HASH=$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")

build_and_push_variant OFF ""
build_and_push_variant ON "-gpu"

# ── Verify on registry (optional dry-run check) ───────────────────
echo ""
echo "Build and push complete."
echo "On your HPC, pull the image with:"
echo "  singularity pull mri_preprocessing.sif     docker://${CPU_NAME}"
echo "  singularity pull mri_preprocessing-gpu.sif docker://${GPU_NAME}"
echo ""
echo "Or use start_control.sh with REGISTRY_URL=${REGISTRY}"
