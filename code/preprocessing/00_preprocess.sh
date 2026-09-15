# =============================================================================
# 00_preprocess.sh — MRI preprocessing pipeline orchestrator
#
# Usage:
#   bash 00_preprocess.sh                       (runs all 6 steps)
#   bash 00_preprocess.sh --start_step 3        (steps 03-06 only)
#   bash 00_preprocess.sh --stop_step 4         (steps 01-04 only)
#   bash 00_preprocess.sh --steps 1,3,5         (only listed steps)
#
# Path arguments (raw/nifti/ras/coreg/inputs) are NOT taken as flags. Each
# step script resolves its own data directories from the ENV (set by the
# launcher: start_control.sh / run_pipeline_conda.sh) or a container-default.
# This orchestrator only enforces step ordering.
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3}

START_STEP=1
STOP_STEP=6
STEPS_FILTER=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]
  --start_step N    First step to run (1-6, default 1)
  --stop_step N     Last step to run (1-6, default 6)
  --steps 1,3,5     Comma-separated list of steps to run (overrides start/stop)
  -h, --help        Show this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start_step) START_STEP="$2"; shift 2 ;;
        --stop_step)  STOP_STEP="$2";  shift 2 ;;
        --steps)      STEPS_FILTER="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Validate step ranges
if (( START_STEP < 1 || START_STEP > 6 )); then
    echo "ERROR: --start_step must be 1-6, got ${START_STEP}" >&2
    exit 1
fi
if (( STOP_STEP < 1 || STOP_STEP > 6 )); then
    echo "ERROR: --stop_step must be 1-6, got ${STOP_STEP}" >&2
    exit 1
fi
if (( START_STEP > STOP_STEP )); then
    echo "ERROR: --start_step (${START_STEP}) cannot be > --stop_step (${STOP_STEP})" >&2
    exit 1
fi

should_run() {
    local n="$1"
    if (( n < START_STEP || n > STOP_STEP )); then
        return 1
    fi
    if [[ -n "$STEPS_FILTER" ]]; then
        [[ ",${STEPS_FILTER}," == *",$n,"* ]] || return 1
    fi
    return 0
}

# Derive the pipeline script directory from this file's location works whether
# the repo lives at /FL_system (in-container) or a local checkout (native HPC).
PIPELINE_DIR="${SCRIPT_DIR}"

# ── Pipeline version banner ─────────────────────────────────────────
MRI_VERSION=$(${PYTHON} -c "import code; print(code.__version__)" 2>/dev/null || echo "dev")
echo "MRI_preprocessing v${MRI_VERSION} — pipeline starting"
echo "Pipeline dir: ${PIPELINE_DIR}"
echo ""

# Step 01
if should_run 1; then
    ${PYTHON} "${PIPELINE_DIR}/01_scanDicom.py"
else
    echo "Skipping step 01"
fi
echo "01 Completed"

# Step 02
if should_run 2; then
    ${PYTHON} "${PIPELINE_DIR}/02_parseDicom.py"
else
    echo "Skipping step 02"
fi
echo "02 Completed"

# Step 03
if should_run 3; then
    ${PYTHON} "${PIPELINE_DIR}/03_saveNifti.py"
else
    echo "Skipping step 03"
fi
echo "03 Completed"

# Step 04
if should_run 4; then
    ${PYTHON} "${PIPELINE_DIR}/04_saveRAS.py"
else
    echo "Skipping step 04"
fi
echo "04 Completed"

# Step 05
if should_run 5; then
    ${PYTHON} "${PIPELINE_DIR}/05_alignScans.py"
else
    echo "Skipping step 05"
fi
echo "05 Completed"

# Step 06
if should_run 6; then
    ${PYTHON} "${PIPELINE_DIR}/06_genInputs.py"
else
    echo "Skipping step 06"
fi
echo "06 Completed"

echo "Pipeline complete."
