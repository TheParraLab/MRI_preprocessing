#!/bin/bash

echo "MRI Preprocessing container started"
echo "NiftyReg is prebuilt into the image (CUDA-enabled)."
echo "Container is ready for preprocessing tasks."
echo ""
echo "Attach with:"
echo "  docker exec -it \"$(hostname)\" bash   # or your container name"
echo "Then run a step, e.g.:"
echo "  python code/preprocessing/01_scanDicom.py --scan_dir /FL_system/data/raw --save_dir /FL_system/data"
echo "or the full pipeline:"
echo "  bash code/preprocessing/00_preprocess.sh"
echo ""
echo "Diagnose NiftyReg/GPU if needed: install_niftyreg_runtime.sh"

# Keep container running
tail -f /dev/null
