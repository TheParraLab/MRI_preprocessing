#!/bin/bash
set -euo pipefail

echo "═════════════════════════════════════════════"
echo "NiftyReg CUDA build at runtime"
echo "═════════════════════════════════════════════"

# Check GPU access is available
if ! nvidia-smi &>/dev/null; then
  echo "ERROR: No GPU detected. Run with --gpus all"
  exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

if [ -d "/niftyreg-src/build" ] && [ -f "/usr/local/bin/reg_f3d" ]; then
  echo "NiftyReg already installed, skipping build."
else
  echo "Building niftyreg with CUDA support..."
  mkdir -p /niftyreg-src/build
  cd /niftyreg-src/build
  cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)
  make install
  echo "Install complete."

  # Verify CUDA is linked
  if ldd /usr/local/bin/reg_f3d | grep -q cuda; then
    echo "✓ CUDA libraries linked successfully"
  else
    echo "✗ WARNING: CUDA libraries NOT linked — reg_f3d will run on CPU"
  fi
fi

echo ""
echo "MRI Preprocessing container ready"
echo "Run: docker exec -it control bash"

# Keep container running
tail -f /dev/null
