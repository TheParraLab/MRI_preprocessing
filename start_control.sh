#!/usr/bin/env bash
# =============================================================================
# MRI Preprocessing — Unified entry point
# =============================================================================
#
# Auto-detects container runtime and deploys accordingly:
#   1. Docker (local/WSL) → docker-compose with --gpus all
#   2. Singularity/Apptainer (HPC) → singularity exec --bind ...
#   3. Conda/Mamba (bare HPC, no containers) → run natively
#
# Requires a .env file at the project root with deployment paths.
#   See .env.example for reference and required variables.
# =============================================================================

set -euo pipefail

# ── Determine the script's directory ─────────────────────────────
script_directory=$(dirname "$(readlink -f "$0")")
project_directory_path=$(realpath "$script_directory")

# ── Load or create .env ─────────────────────────────────────────
ENV_FILE="${project_directory_path}/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "No .env file found at ${ENV_FILE}."
  if [ -f "${project_directory_path}/.env.example" ]; then
    echo "Copying .env.example → .env — please review paths before re-running."
    cp "${project_directory_path}/.env.example" "$ENV_FILE"
  else
    echo "ERROR: Neither .env nor .env.example found."
    exit 1
  fi
  echo ""
  echo "Edit ${ENV_FILE} with your deployment paths, then run:"
  echo "  bash start_control.sh"
  exit 0
fi

# Source variables (each line must be KEY=VALUE with no surrounding whitespace)
while IFS='=' read -r key value || [ -n "$key" ]; do
  # Skip comments and blank lines
  [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
  export "${key}=${value}"
done < "$ENV_FILE"

# ── Validate required paths ─────────────────────────────────────
required_vars=(
  COMPOSE_PROJECT_NAME
  DATA_DIRECTORY_PATH
  NIFTI_DIRECTORY_PATH
  RAS_DIRECTORY_PATH
  COREG_DIRECTORY_PATH
  INPUTS_DIRECTORY_PATH
)

optional_reg_vars=(REGISTRY_URL IMAGE_REPOSITORY IMAGE_TAG)
for var in "${optional_reg_vars[@]}"; do
  export "${var:-}"
done

missing_deps=()
for var in "${required_vars[@]}"; do
  if [ -z "${!var:-}" ]; then
    missing_deps+=("$var")
  fi
done

if [ ${#missing_deps[@]} -gt 0 ]; then
  echo "ERROR: Missing required variables in .env:"
  for var in "${missing_deps[@]}"; do
    echo "  - ${var}"
  done
  exit 1
fi

# ── Build the intended image reference ───────────────────────────
# Registry URL is optional (e.g. local Docker build-and-run); only prefix it
# when set so the result is always a well-formed reference.
image_ref() {
  local repo="${IMAGE_REPOSITORY:-mri_preprocessing}"
  local tag="${IMAGE_TAG:-latest}"
  if [ -n "${REGISTRY_URL:-}" ]; then
    echo "${REGISTRY_URL%/}/${repo}:${tag}"
  else
    echo "${repo}:${tag}"
  fi
}

# ── Deployment finalizer (runs on exit / Ctrl-C) ────────────────
# All supported runtimes spawn a long-lived foreground process, so sessions
# normally end with a signal; the EXIT trap is the one place we can reliably
# record what actually ran. Appends a "runtime" section to the manifest with
# the runtime-resolved identity (image ID / sif / env), complementing the
# *intended* reference written at start.
deployment_finalize() {
  [ "${FINALIZED:-false}" = true ] && return 0
  FINALIZED=true

  local manifest="${DEPLOY_LOG_DIR:-}/manifest.json"
  [ -n "${DEPLOY_LOG_DIR:-}" ] || return 0
  [ -f "$manifest" ] || return 0
  command -v python3 &>/dev/null || return 0
  [ -n "${RUNTIME:-}" ] || return 0

  local image_id="unresolved" sif_bytes=0 env_sha="none"
  case "$RUNTIME" in
    docker|docker-compose)
      if [ -n "${CONTAINER_NAME:-}" ] && docker inspect "${CONTAINER_NAME}" &>/dev/null; then
        image_id=$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}" 2>/dev/null || echo unresolved)
      fi
      python3 - "$manifest" "docker" "${CONTAINER_NAME:-}" "${COMPOSE_PROJECT_NAME:-}" "$image_id" <<'PYEOF'
import json, sys
path, rtype, container, project, image_id = sys.argv[1:6]
m = json.load(open(path))
m["runtime"] = {
    "type": rtype,
    "container_name": container or None,
    "compose_project": project or None,
    "resolved_image_id": image_id,
}
with open(path, "w") as f:
    json.dump(m, f, indent=2)
    f.write("\n")
PYEOF
      ;;
    singularity|apptainer)
      [ -f "${SIF_IMAGE:-}" ] && sif_bytes=$(stat -c %s "${SIF_IMAGE}" 2>/dev/null || echo 0)
      python3 - "$manifest" "$RUNTIME" "${SIF_IMAGE:-}" "$sif_bytes" <<'PYEOF'
import json, sys
path, rtype, sif, size = sys.argv[1:5]
m = json.load(open(path))
m["runtime"] = {
    "type": rtype,
    "sif_path": sif or None,
    "sif_bytes": int(size),
}
with open(path, "w") as f:
    json.dump(m, f, indent=2)
    f.write("\n")
PYEOF
      ;;
    conda|mamba)
      if [ -f "${project_directory_path}/environment.yml" ]; then
        env_sha=$(sha256sum "${project_directory_path}/environment.yml" | awk '{print $1}')
      fi
      python3 - "$manifest" "$RUNTIME" "${CONDA_ENV_NAME:-mri_preproc}" "$env_sha" <<'PYEOF'
import json, sys
path, rtype, env_name, env_sha = sys.argv[1:5]
m = json.load(open(path))
m["runtime"] = {
    "type": rtype,
    "conda_env": env_name,
    "environment_yml_sha256": env_sha,
}
with open(path, "w") as f:
    json.dump(m, f, indent=2)
    f.write("\n")
PYEOF
      ;;
  esac
}

# ── Create timestamped deployment log directory ─────────────────
DEPLOYMENT_ID=$(date +%Y%m%d_%H%M%S)
DEPLOY_LOG_DIR="${project_directory_path}/deployments/${DEPLOYMENT_ID}"
mkdir -p "$DEPLOY_LOG_DIR"

# Record which source tree this deployment started from.
# (The runtime-detected image/container identity is appended later at exit,
# see deployment_finalize().)
GIT_COMMIT="unknown"
GIT_DIRTY=true
if git -C "$project_directory_path" rev-parse --git-dir &>/dev/null; then
  GIT_COMMIT=$(git -C "$project_directory_path" rev-parse HEAD)
  if [ -z "$(git -C "$project_directory_path" status --porcelain)" ]; then
    GIT_DIRTY=false
  fi
fi

# Write a minimal manifest so deployments can be audited later
cat > "${DEPLOY_LOG_DIR}/manifest.json" <<MANIFEST
{
  "deployment_id": "${DEPLOYMENT_ID}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "runtime_env_file": "${ENV_FILE}",
  "image": "$(image_ref)",
  "source": {
    "git_commit": "${GIT_COMMIT}",
    "git_dirty": ${GIT_DIRTY}
  },
  "paths": {
    "raw_data": "${DATA_DIRECTORY_PATH}",
    "nifti": "${NIFTI_DIRECTORY_PATH}",
    "ras": "${RAS_DIRECTORY_PATH}",
    "coreg": "${COREG_DIRECTORY_PATH}",
    "inputs": "${INPUTS_DIRECTORY_PATH}"
  }
}
MANIFEST

# Snapshot the .env used for this deployment (so deployments are self-contained)
cp "$ENV_FILE" "${DEPLOY_LOG_DIR}/.env.snapshot"

export DEPLOY_LOG_DIR
# Record the runtime-resolved identity of what actually ran (image ID, sif,
# conda env) in the manifest when this session ends — normally via Ctrl-C.
trap 'deployment_finalize' EXIT INT TERM
echo "Deployment log: ${DEPLOY_LOG_DIR}"
echo ""

# ── Detect WSL platform ────────────────────────────────────────
WSL=false
if grep -qi Microsoft /proc/version; then
  echo "Running on WSL"
  WSL=true
elif grep -qi WSL /proc/version; then
  echo "Running on WSL 2"
  WSL=true
else
  echo "Running on pure Linux"
fi

# ── Auto-detect container runtime ──────────────────────────────
# Priority: Docker → Singularity/Apptainer → Conda/Mamba → error

detect_runtime() {
  if command -v docker &>/dev/null && docker info &>/dev/null; then
    if command -v docker compose &>/dev/null; then
      echo "docker"
      return 0
    elif command -v docker-compose &>/dev/null; then
      echo "docker-compose"
      return 0
    fi
  fi

  if command -v singularity &>/dev/null; then
    echo "singularity"
    return 0
  elif command -v apptainer &>/dev/null; then
    echo "apptainer"
    return 0
  fi

  # Fallback: conda/mamba (native HPC, no containers)
  if command -v mamba &>/dev/null; then
    echo "mamba"
    return 0
  elif command -v conda &>/dev/null; then
    echo "conda"
    return 0
  fi

  return 1
}

RUNTIME=$(detect_runtime) || {
  echo ""
  echo "ERROR: No container runtime or conda found. Install one of:"
  echo ""
  echo "DOCKER (recommended for development):"
  echo "  https://docs.docker.com/get-docker/"
  echo ""
  echo "SINGULARITY/APPTAINER (for HPC clusters, no root required):"
  echo "  https://apptainer.org/docs/user/latest/quick_start.html#installation"
  echo ""
  echo "CONDA/MAMBA (native HPC, fully local):"
  echo "  https://docs.conda.io/en/latest/miniconda.html"
  echo "  https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html"
  echo ""
  echo "Then run: conda env create -f environment.yml"
  echo "         conda activate ${CONDA_ENV_NAME:-mri_preproc}"
  echo "         ./run_pipeline_conda.sh"
  exit 1
}

echo "Detected runtime: ${RUNTIME}"

# ── Start the container / pipeline ─────────────────────────────
case "$RUNTIME" in
  docker|docker-compose)
    COMPOSE_CMD=$(command -v docker compose &>/dev/null && echo "docker compose" || echo "docker-compose")

    if [ "$WSL" = true ]; then
      COMPOSE_FILE="./control_system/docker-compose-wsl.yml"
      echo "Using Docker (WSL): ${COMPOSE_FILE}"
    else
      COMPOSE_FILE="./control_system/docker-compose.yml"
      echo "Using Docker: ${COMPOSE_FILE}"
    fi

    COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME}-${DEPLOYMENT_ID}"
    CONTAINER_NAME="control-${DEPLOYMENT_ID}"
    # Inside the container the deployment dir is mounted at /deployment.
    export LOG_DIR="/deployment/logs"
    export COMPOSE_PROJECT_NAME CONTAINER_NAME DEPLOY_LOG_DIR
    ${COMPOSE_CMD} -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" up --build
    ;;

  singularity|apptainer)
    SIF_IMAGE="${SIF_PATH:-./control_system/mri_preprocessing.sif}"
    REGISTRY_REF="$(image_ref)"

    if [ ! -f "$SIF_IMAGE" ]; then
      echo "No local .sif found. Pulling from registry: ${REGISTRY_REF}"
      "${RUNTIME}" pull "$SIF_IMAGE" "docker://${REGISTRY_REF}" || {
        echo ""
        echo "ERROR: Failed to pull image from ${REGISTRY_REF}"
        echo "Check that the registry URL is correct and accessible."
        exit 1
      }
      echo "Image cached at ${SIF_IMAGE}"
    fi

    Binds=(
      "${DATA_DIRECTORY_PATH}:/FL_system/data/raw"
      "${NIFTI_DIRECTORY_PATH}:/FL_system/data/nifti"
      "${RAS_DIRECTORY_PATH}:/FL_system/data/RAS"
      "${COREG_DIRECTORY_PATH}:/FL_system/data/coreg"
      "${INPUTS_DIRECTORY_PATH}:/FL_system/data/inputs"
      "${DEPLOY_LOG_DIR}:/deployment/"
    )

    # Pass GPUs into the container only when the host node actually has a
    # GPU (nvidia-smi on PATH). On a CPU-only HPC partition, --nv causes
    # Singularity/Apptainer to fail or hang; reg_f3d falls back to CPU either
    # way, so omitting --nv is safe and matches the container's own GPU-less
    # path (see control_system/scripts/install_niftyreg_runtime.sh).
    Additional=( )
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
      Additional+=( --nv )
      echo "GPU detected on host — passing --nv into the container."
    else
      echo "No GPU detected on this host — launching without --nv (CPU-only)."
      echo "Coregistration (step 05) will fall back to CPU if niftyreg/CUDA is not loaded."
    fi

    bind_str=$(IFS=','; echo "${Binds[*]}")

    echo "Using ${RUNTIME} with image: $SIF_IMAGE"
    echo "Registry reference: ${REGISTRY_REF}"
    echo "Bindings:"
    for b in "${Binds[@]}"; do echo "  ${b}"; done
    echo ""
    echo "Pipeline scripts are baked into the image."
    echo "Once the prompt appears, run:"
    echo "  python code/preprocessing/01_scanDicom.py --scan-dir /FL_system/data/raw --save-dir /FL_system/data"
    echo "  bash code/preprocessing/00_preprocess.sh              (runs all steps)"
    echo ""

    ${RUNTIME} exec \
      "${Additional[@]}" \
      --bind "$bind_str" \
      --pwd /FL_system \
      -e DATA_DIRECTORY_PATH="$DATA_DIRECTORY_PATH" \
      -e NIFTI_DIRECTORY_PATH="$NIFTI_DIRECTORY_PATH" \
      -e LOG_DIR="/deployment/logs" \
      "$SIF_IMAGE" bash
    ;;

  conda|mamba)
    ENV_YML="${script_directory}/environment.yml"
    ENV_NAME="${CONDA_ENV_NAME:-mri_preproc}"

    # Host-local deployment log dir (no container mount for bare conda runs).
    export LOG_DIR="${DEPLOY_LOG_DIR}/logs"

    if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" == "${ENV_NAME}" ]]; then
      echo "Conda env ${ENV_NAME} already active."
    else
      echo ""
      echo "Installing/activating conda environment ${ENV_NAME}..."
      if ${RUNTIME} env create -f "${ENV_YML}" --yes 2>/dev/null; then
        echo "→ Environment installed."
      fi

      eval "$(${RUNTIME} shell.bash hook)"
      ${RUNTIME} activate ${ENV_NAME}
      echo "→ ${ENV_NAME} activated."
    fi

    # Check for niftyreg availability
    if module load niftyreg 2>/dev/null; then
      echo "→ Found niftyreg via system module."
    elif command -v reg_f3d &>/dev/null; then
      echo "→ Found niftyreg in PATH."
    else
      echo ""
      echo "WARNING: reg_f3d (niftyreg) not found in PATH."
      echo "Install options:"
      echo "  1) module load niftyreg                        ← if available as HPC module"
      echo "  2) ${script_directory}/code/scripts/install_niftyreg.sh  ← build from source"
      echo ""
      echo "After installing, re-run this script."
      exit 1
    fi

    echo ""
    echo "✓ dcm2niix: $(dcm2niix -version 2>&1 | head -1)"
    echo "✓ reg_f3d:  $(reg_f3d -version 2>&1 | head -1 || echo 'available')"
    echo "✓ Python:   $(python --version 2>&1)"
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "Pipeline ready. Running 00_preprocess.sh..."
    echo "──────────────────────────────────────────────────────────"
    echo ""

    cd "${project_directory_path}"
    bash code/preprocessing/00_preprocess.sh \
      --scan-dir "${DATA_DIRECTORY_PATH}" \
      --save-dir "${DATA_DIRECTORY_PATH}"
    ;;

  *)
    echo "ERROR: Unknown runtime: ${RUNTIME}"
    exit 1
    ;;
esac