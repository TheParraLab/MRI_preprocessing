# =============================================================================
# 00_preprocess.sh — MRI preprocessing pipeline orchestrator
#
# Usage:
#   bash 00_preprocess.sh                          (runs all 6 steps)
#   bash 00_preprocess.sh --start-step 3            (steps 03-06 only)
#   bash 00_preprocess.sh --stop-step 4             (steps 01-04 only)
#   bash 00_preprocess.sh --steps 1,3,5             (only listed steps)
#   bash 00_preprocess.sh --scan-dir /path/raw      (override scan path)
#   bash 00_preprocess.sh --save-dir /path/output   (override save path)
# =============================================================================

set -euo pipefail

SCAN_DIR=""
SAVE_DIR=""
START_STEP=1
STOP_STEP=6
STEPS_FILTER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --scan-dir)   SCAN_DIR="$2"; shift 2 ;;
    --save-dir)   SAVE_DIR="$2"; shift 2 ;;
    --start-step) START_STEP="$2"; shift 2 ;;
    --stop-step)  STOP_STEP="$2"; shift 2 ;;
    --steps)      STEPS_FILTER="$2"; shift 2 ;;
    *)            shift ;;
  esac
done

should_run() {
  if [ "$1" -lt "$START_STEP" ] || [ "$1" -gt "$STOP_STEP" ]; then
    return 1
  fi
  if [ -n "$STEPS_FILTER" ] && [[ ! ",$STEPS_FILTER," == *",$1,"* ]]; then
    return 1
  fi
  return 0
}

# Step 01
if should_run 1; then
  python /FL_system/code/preprocessing/01_scanDicom.py
else
  echo "Skipping step 01"
fi
echo "01 Completed"

# Step 02
if should_run 2; then
  python /FL_system/code/preprocessing/02_parseDicom.py
else
  echo "Skipping step 02"
fi
echo "02 Completed"

# Step 03
if should_run 3; then
  python /FL_system/code/preprocessing/03_saveNifti.py
else
  echo "Skipping step 03"
fi
echo "03 Completed"

# Step 04
if should_run 4; then
  python /FL_system/code/preprocessing/04_saveRAS.py
else
  echo "Skipping step 04"
fi
echo "04 Completed"

# Step 05
if should_run 5; then
  python /FL_system/code/preprocessing/05_alignScans.py
else
  echo "Skipping step 05"
fi
echo "05 Completed"

# Step 06
if should_run 6; then
  python /FL_system/code/preprocessing/06_genInputs.py
else
  echo "Skipping step 06"
fi
echo "06 Completed"

echo "Pipeline complete."