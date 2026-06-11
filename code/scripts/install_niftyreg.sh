#!/usr/bin/env bash
# =============================================================================
# Install niftyreg v2.0.0 from source
# =============================================================================
#
# niftyreg is a C++/CUDA registration toolkit required for step 05 (alignScans).
# It is NOT available via conda-forge, so this helper builds it locally.
#
# Usage:
#   bash scripts/install_niftyreg.sh [install_prefix]
#
# Default install prefix: ~/mri_niftyreg
#
# Requirements on host:
#   - gcc, g++, cmake (conda provides these or use system modules)
#   - CUDA toolkit (optional, for GPU-accelerated registration)
#
# =============================================================================

set -euo pipefail

PREFIX="${1:-${HOME}/mri_niftyreg}"
BUILD_DIR="/tmp/niftyreg_build_${RANDOM}"

echo "┌─────────────────────────────────────────────────────┐"
echo "│ NiftyReg v2.0.0 Installer                            │"
echo "└─────────────────────────────────────────────────────┘"
echo ""

# ── Check build requirements ──────────────────────────────────────
if ! command -v cmake &>/dev/null; then
  echo "ERROR: cmake not found. Install via: conda install cmake"
  exit 1
fi

if ! command -v make &>/dev/null; then
  echo "WARNING: make not found, trying ninja..."
  if ! command -v ninja &>/dev/null; then
    exit 1
  fi
fi

# ── Check for CUDA (optional, for GPU mode) ───────────────────────
CUDA_FOUND=false
if command -v nvcc &>/dev/null; then
  CUDA_VERSION=$(nvcc --version | grep -i "release" | awk -F',' '{gsub(/ /, "", $3); print $3}')
  echo "✓ CUDA ${CUDA_VERSION} found — building with GPU acceleration"
  CUDA_FOUND=true
else
  echo "⚠ CUDA not found — building without GPU acceleration (CPU-only mode)"
fi

# ── Create build directory ────────────────────────────────────────
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
mkdir -p "${PREFIX}"

echo "→ Cloning niftyreg v2.0.0..."
git clone --branch v2.0.0 https://github.com/KCL-BMEIS/niftyreg.git "${BUILD_DIR}/niftyreg-git"

echo "→ Configuring build (CUDA=${CUDA_FOUND})..."
cd "${BUILD_DIR}/niftyreg-git"
mkdir -p build
cd build

CMAKE_CUDA_FLAG="-DBUILD_CUDA=ON"
if [[ "${CUDA_FOUND}" = false ]]; then
  CMAKE_CUDA_FLAG="-DBUILD_CUDA=OFF"
fi

cmake .. \
  ${CMAKE_CUDA_FLAG} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}"

echo "→ Compiling (this takes ~10 minutes)..."
make install

echo "→ Cleaning up build files..."
rm -rf "${BUILD_DIR}"

echo ""
echo "═════════════════════════════════════════════"
echo "✓ NiftyReg installed to: ${PREFIX}"
echo ""
echo "Add to PATH before running pipeline:"
echo "  export PATH=${PREFIX}/bin:\${PATH}"
echo ""
echo "Then run:"
echo "  ./run_pipeline_conda.sh"
echo "═════════════════════════════════════════════" 