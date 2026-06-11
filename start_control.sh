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
# =============================================================================

set -euo pipefail

# ── Prompt the user for paths ─────────────────────────────────────
echo "Please enter the raw data path:"
read -r data_directory_path

echo "Please enter the NIfTI output path:"
read -r nifti_directory_path

# ── Determine the script's directory ─────────────────────────────
script_directory=$(dirname "$(readlink -f "$0")")
project_directory_path=$(realpath "$script_directory")

# ── Export environment variables ────────────────────────────────
export PROJECT_DIRECTORY_PATH="${project_directory_path}"
export DATA_DIRECTORY_PATH="${data_directory_path}"
export NIFTI_DIRECTORY_PATH="${nifti_directory_path}"

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
      echo "Using Docker (WSL): docker-compose-wsl.yml"
      ${COMPOSE_CMD} -f ./control_system/docker-compose-wsl.yml up --build
    else
      echo "Using Docker: docker-compose.yml"
      ${COMPOSE_CMD} -f ./control_system/docker-compose.yml up --build
    fi
    ;;

  singularity|apptainer)
    SIF_IMAGE="./control_system/mri_preprocessing.sif"

    if [ ! -f "$SIF_IMAGE" ]; then
      echo ""
      echo "ERROR: Singularity image not found at $SIF_IMAGE"
      echo ""
      echo "To deploy on HPC sites, build or copy a .sif image:"
      echo ""
      echo "  Option A — Build locally (requires root):"
      echo "    sudo ${RUNTIME} build mri_preprocessing.sif control_system/mri_preprocessing.singularity.def"
      echo ""
      echo "  Option B — Pull from an existing Docker/OCI image:"
      echo "    ${RUNTIME} pull mri_preprocessing.sif docker://<your-image>:tag"
      echo ""
      echo "Then copy the .sif file to control_system/ on your HPC site."
      echo ""
      exit 1
    fi

    RAW_BIND="${DATA_DIRECTORY_PATH}:/FL_system/data/raw"
    PROJECT_BIND="${PROJECT_DIRECTORY_PATH}:/FL_system"

    echo "Using ${RUNTIME} with image: $SIF_IMAGE"
    echo "Binding raw data : $DATA_DIRECTORY_PATH → /FL\_system/data/raw"
    echo "Binding project : $PROJECT_DIRECTORY_PATH → /FL\_system"
    echo ""
    echo "Once the prompt appears, run your pipeline scripts inside the container:"
    echo "  python code/preprocessing/01_scanDicom.py --scan-dir /FL_system/data/raw --save-dir /FL_system/data"
    echo "  bash code/preprocessing/00_preprocess.sh              (runs all steps)"
    echo ""

    ${RUNTIME} exec \
      --bind "$RAW_BIND,$PROJECT_BIND" \
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