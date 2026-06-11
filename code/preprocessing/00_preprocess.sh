#!/bin/bash

# Pass --scan-dir and --save-dir through to downstream Python scripts
# Defaults to container paths; override for conda/native HPC:
#   bash 00_preprocess.sh --scan-dir /path/raw --save-dir /path/output

SCAN_DIR=""
SAVE_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --scan-dir)  SCAN_DIR="$2"; shift ;;
    --save-dir)  SAVE_DIR="$2"; shift ;;
    *)  shift ;;
  esac
done

# Build step 01 args (already supports --scan-dir / --save-dir)
STEP01_ARGS=()
if [ -n "$SCAN_DIR" ]; then STEP01_ARGS+=("--scan-dir" "$SCAN_DIR"); fi
if [ -n "$SAVE_DIR" ]; then STEP01_ARGS+=("--save-dir" "$SAVE_DIR"); fi

python /FL_system/code/preprocessing/01_scanDicom.py "${STEP01_ARGS[@]}"
echo "01 Completed"
python /FL_system/code/preprocessing/02_parseDicom.py
echo "02 Completed"
python /FL_system/code/preprocessing/03_saveNifti.py
echo "03 Completed"
python /FL_system/code/preprocessing/04_saveRAS.py
echo "04 Completed"
python /FL_system/code/preprocessing/05_alignScans.py
echo "05 Completed"
python /FL_system/code/preprocessing/06_genInputs.py
echo "06 Completed"