#!/bin/bash
set -euo pipefail

# Diagnostic / troubleshooting helper for NiftyReg.
#
# The CUDA-enabled NiftyReg build is performed at image BUILD time (see
# control_system/dockerfile). This script is NOT part of the startup path —
# use it to:
#   1) verify GPU access + that /usr/local/bin/reg_f3d is CUDA-linked, and
#   2) rebuild NiftyReg from source if the binary is missing or broken.

echo "═════════════════════════════════════════════"
echo "NiftyReg status check (build-time image)"
echo "═════════════════════════════════════════════"

# Informational only — build does not require a GPU; runtime coregistration does.
if nvidia-smi &>/dev/null; then
  echo "GPU detected:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "WARNING: No GPU detected (nvidia-smi unavailable). Run with --gpus all / --nv for coregistration."
fi
echo ""

if [ -x /usr/local/bin/reg_f3d ]; then
  ldd /usr/local/bin/reg_f3d > /tmp/reg_f3d_ldd_out 2>&1 || true
  if grep -q cuda /tmp/reg_f3d_ldd_out; then
    echo "✓ reg_f3d found and CUDA libraries linked — nothing to do."
    rm -f /tmp/reg_f3d_ldd_out
    exit 0
  else
    echo "WARNING: reg_f3d present but CUDA libraries NOT linked (CPU-only build)."
  fi
else
  echo "reg_f3d not found. Attempting to build from source at /niftyreg-src..."
fi

if [ ! -d /niftyreg-src ]; then
  echo "Source tree missing — cloning niftyreg v2.0.0 (requires network)..."
  git clone --branch v2.0.0 --depth 1 https://github.com/KCL-BMEIS/niftyreg.git /niftyreg-src
fi

echo "Building niftyreg with CUDA support..."
# CHECK_GPU=OFF: skip the configure-time card probe (see dockerfile); with it
# on, building without a reachable GPU silently disables CUDA.
mkdir -p /niftyreg-build
cmake -S /niftyreg-src -B /niftyreg-build -DUSE_CUDA=ON -DCHECK_GPU=OFF -DCMAKE_BUILD_TYPE=Release
make -C /niftyreg-build -j"$(nproc)"
make -C /niftyreg-build install
ldconfig

if ldd /usr/local/bin/reg_f3d | grep -q cuda; then
  echo "✓ Rebuild complete: CUDA libraries linked successfully"
else
  echo "✗ WARNING: CUDA libraries NOT linked — reg_f3d will run on CPU"
fi

rm -f /tmp/reg_f3d_ldd_out
echo ""
echo "NiftyReg ready."
