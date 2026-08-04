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
  PROJECT_DIRECTORY_PATH
  DATA_DIRECTORY_PATH
  NIFTI_DIRECTORY_PATH
  RAS_DIRECTORY_PATH
  COREG_DIRECTORY_PATH
  INPUTS_DIRECTORY_PATH
)

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

# ── Create timestamped deployment log directory ─────────────────
DEPLOYMENT_ID=$(date +%Y%m%d_%H%M%S)
DEPLOY_LOG_DIR="${project_directory_path}/deployments/${DEPLOYMENT_ID}"
mkdir -p "$DEPLOY_LOG_DIR"

# Write a minimal manifest so deployments can be audited later
cat > "${DEPLOY_LOG_DIR}/manifest.json" <<MANIFEST
{
  "deployment_id": "${DEPLOYMENT_ID}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "runtime_env_file": "${ENV_FILE}",
  "paths": {
    "project": "${PROJECT_DIRECTORY_PATH}",
    "raw_data": "${DATA_DIRECTORY_PATH}",
    "nifti": "${NIFTI_DIRECTORY_PATH}",
    "ras": "${RAS_DIRECTORY_PATH}",
    "coreg": "${COREG_DIRECTORY_PATH}",
    "inputs": "${INPUTS_DIRECTORY_PATH}"
  }
}
MANIFEST
export DEPLOY_LOG_DIR

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
  echo "         conda activate mri_preproc"
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

    # Add deployment log volume to pass through compose env
    export DEPLOY_LOG_DIR
    ${COMPOSE_CMD} -f "${COMPOSE_FILE}" up --build
    ;;

  singularity|apptainer)
    SIF_IMAGE="./control_system/mri_preprocessing.sif"

    if [ ! -f "$SIF_IMAGE" ]; then
      echo ""
      echo "ERROR: Singularity image not found at $SIF_IMAGE"
      echo ""
      echo "To deploy on HPC sites, pull from an existing Docker/OCI image:"
      echo ""
      echo "  ${RUNTIME} pull mri_preprocessing.sif docker://<your-image>:tag"
      echo ""
      echo "Then copy the .sif file to control_system/ on your HPC site."
      echo ""
      exit 1
    fi

    Binds=(
      "${PROJECT_DIRECTORY_PATH}:/FL_system"
      "${DATA_DIRECTORY_PATH}:/FL_system/data/raw"
      "${NIFTI_DIRECTORY_PATH}:/FL_system/data/nifti"
      "${RAS_DIRECTORY_PATH}:/FL_system/data/RAS"
      "${COREG_DIRECTORY_PATH}:/FL_system/data/coreg"
      "${INPUTS_DIRECTORY_PATH}:/FL_system/data/inputs"
      "${DEPLOY_LOG_DIR}:/FL_system/logs/deployments/${DEPLOYMENT_ID}"
    )

    bind_str=$(IFS=','; echo "${Binds[*]}")

    echo "Using ${RUNTIME} with image: $SIF_IMAGE"
    echo "Binding:"
    for b in "${Binds[@]}"; do echo "  ${b}"; done
    echo ""
    echo "Once the prompt appears, run your pipeline scripts inside the container:"
    echo "  python code/preprocessing/01_scanDicom.py --scan-dir /FL_system/data/raw --save-dir /FL_system/data"
    echo "  bash code/preprocessing/00_preprocess.sh              (runs all steps)"
    echo ""

    ${RUNTIME} exec \
      --bind "$bind_str" \
      -e DATA_DIRECTORY_PATH="$DATA_DIRECTORY_PATH" \
      -e NIFTI_DIRECTORY_PATH="$NIFTI_DIRECTORY_PATH" \
      -e PROJECT_DIRECTORY_PATH="$PROJECT_DIRECTORY_PATH" \
      "$SIF_IMAGE" bash
    ;;

  conda|mamba)
    ENV_YML="${script_directory}/environment.yml"
    ENV_NAME="mri_preproc"

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