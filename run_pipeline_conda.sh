#!/usr/bin/env bash
# =============================================================================
# MRI Preprocessing — Conda-only pipeline runner
# =============================================================================
#
# For HPC sites that don't support Docker or Singularity. Requires conda/mamba.
#
# Usage:
#   1) Clone this repo onto HPC
#   2) Run `./setup_conda.sh`      (one-time: creates conda env + installs niftyreg)
#      OR skip if niftyreg already available as an HPC module
#   3) Run `./start_control.sh`    (same script as today — auto-detects conda fallback)
#   4) Run `bash code/preprocessing/00_preprocess.sh`
#
# ── NiftyReg availability ─────────────────────────────────────────────────
#
# niftyreg is NOT bundled via conda (CUDA build). Options:
#   Option A — Use an existing HPC module (preferred)
#     module load niftyreg
#   Option B — Build manually (requires gcc, cmake on site)
#     ./scripts/install_niftyreg.sh
#   Option C — Copy a pre-built `.sif` image from a Docker build and run via Singularity
#
# =============================================================================

set -euo pipefail

# ── Detect script root ─────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_YML="${SCRIPT_DIR}/environment.yml"
ENV_NAME="mri_preproc"
NIFTYREG_MODULE_AVAILABLE=false
NIFTYREG_SYSTEM_INSTALL=false

# ── Prompt paths ───────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────┐"
echo "│ MRI Preprocessing — Conda Pipeline                   │"
echo "└─────────────────────────────────────────────────────┘"
echo ""

echo "Please enter the raw DICOM data path:"
read -r DATA_DIRECTORY_PATH

echo "Please enter the NIfTI output path:"  
read -r NIFTI_DIRECTORY_PATH

PROJECT_DIRECTORY_PATH="${SCRIPT_DIR}"

# ── Export env vars for all pipeline scripts ──────────────────────
export PROJECT_DIRECTORY_PATH
export DATA_DIRECTORY_PATH
export NIFTI_DIRECTORY_PATH

# ── Check for existing conda env ──────────────────────────────────
CONDAPATH=""
if command -v mamba &>/dev/null; then
  CONDAPATH=$(mamba info --base 2>/dev/null) || true
  CMD=mamba
elif command -v conda &>/dev/null; then
  CONDAPATH=$(conda info --base 2>/dev/null) || true
  CMD=conda
fi

if [[ -z "${CONDAPATH}" ]]; then
  echo ""
  echo "ERROR: Neither conda nor mamba found. Install one of:"
  echo "  Conda:  https://docs.conda.io/en/latest/miniconda.html"
  echo "  Mamba:  https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html"
  echo ""
  echo "Then re-run this script."
  exit 1
fi

if [[ -d "${CONDAPATH}/envs/${ENV_NAME}" ]]; then
  echo "Environment ${ENV_NAME} already exists. Activating..."
else
  echo ""
  echo "Installing conda environment ${ENV_NAME}..."
  ${CMD} env create -f "${ENV_YML}" --yes
fi

# ── Activate env ──────────────────────────────────────────────────
if ${CMD} env list | grep -q "^${ENV_NAME}"; then
  eval "$(${CMD} shell.bash hook)"
  ${CMD} activate "${ENV_NAME}"
  echo "→ ${ENV_NAME} activated successfully."
else
  echo "ERROR: Could not activate ${ENV_NAME} — did the install succeed?"
  exit 1
fi

# ── Check for niftyreg ───────────────────────────────────────────
if module load niftyreg 2>/dev/null; then
  NIFTYREG_MODULE_AVAILABLE=true
  echo "→ Found niftyreg via system module."
fi

if ! command -v reg_f3d &>/dev/null; then
  echo ""
  echo "WARNING: reg_f3d not found in PATH. niftyreg required for step 05."
  echo ""
  echo "Install options:"
  echo "  1) module load niftyreg                        ← preferred if available"
  echo "  2) ${SCRIPT_DIR}/scripts/install_niftyreg.sh   ← build from source"
  echo ""
  echo "Exiting — please install niftyreg and re-run."
  exit 1
fi

# ── Resolve paths into container-equivalent dirs ─────────────────
# All pipeline scripts expect paths under /FL_system/
# We bind them directly since we're running natively (no container)

# ── Verify dependencies ──────────────────────────────────────────
echo ""
echo "✓ dcm2niix:    $(command -v dcm2niix 2>/dev/null || echo 'NOT FOUND')"
echo "✓ reg_f3d:     $(command -v reg_f3d 2>/dev/null || echo 'NOT FOUND')"
echo "✓ python:      $(python --version 2>&1)"
echo "✓ pydicom:     $(python -c 'import pydicom; print(pydicom.__version__)' 2>&1)"
echo ""

echo "──────────────────────────────────────────────────────────"
echo "Pipeline ready. Run from project root:"
echo "  bash code/preprocessing/00_preprocess.sh"
echo "──────────────────────────────────────────────────────────"

# Run the pipeline directly
cd "${PROJECT_DIRECTORY_PATH}"
bash code/preprocessing/00_preprocess.sh