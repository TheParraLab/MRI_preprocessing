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

# Source variables. Each line must be KEY=VALUE. Robust to:
#   - Windows line endings (CRLF) — stripped before parsing.
#   - Leading/trailing whitespace around key or value.
#   - Inline ` # comment` (whitespace-prefixed) — stripped, but `KEY=a#b`
#     keeps the `#b` literal (dotenv semantics: '#' is only a comment when
#     preceded by whitespace).
#   - A missing `=` or an invalid variable name.
# On any malformed line we name the file + line number and abort before
# the value can reach apptainer and produce an inscrutable "could not open
# image <garbage>" error.
env_malformed=0
line_no=0
while IFS= read -r raw_line || [ -n "$raw_line" ]; do
  line_no=$((line_no + 1))

  # Strip CR (CRLF files) and surrounding whitespace.
  line="${raw_line%$'\r'}"
  while [[ -n "$line" && "${line:0:1}" == " " || "${line:0:1}" == $'\t' ]]; do line="${line:1}"; done
  while [[ -n "$line" && "${line: -1}" == " " || "${line: -1}" == $'\t' ]];   do line="${line:0:${#line}-1}"; done

  # Skip blank lines and full-line comments.
  if [ -z "$line" ]; then continue; fi
  if [ "${line:0:1}" = "#" ]; then continue; fi

  # Inline comment: truncate at the first whitespace-preceded '#' (space or
  # tab). `KEY=a#b` keeps `#b` (the '#' is not whitespace-preceded, so it is
  # part of the value per dotenv semantics). ${line%%pat*} is a no-op when
  # no match, so this line is safe either way.
  line="${line%%[[:space:]]#*}"
  while [[ "${line: -1}" == " " || "${line: -1}" == $'\t' ]]; do line="${line:0:${#line}-1}"; done
  if [ -z "$line" ]; then continue; fi

  # Must contain an '=' separator. A bare word on its own line is malformed.
  if [[ "$line" != *"="* ]]; then
    echo "ERROR: ${ENV_FILE}:${line_no}: expected KEY=VALUE, got: ${line}"
    env_malformed=1
    continue
  fi

  key="${line%%=*}"
  value="${line#*=}"

  # Key must be a valid shell identifier.
  if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "ERROR: ${ENV_FILE}:${line_no}: invalid variable name '${key}'"
    env_malformed=1
    continue
  fi

  export "${key}=${value}"
done < "$ENV_FILE"

if [ "${env_malformed}" -eq 1 ]; then
  echo ""
  echo "Aborting: fix the malformed .env line(s) above and re-run."
  exit 1
fi

# ── Validate required paths ─────────────────────────────────────
required_vars=(
  COMPOSE_PROJECT_NAME
  DATA_DIRECTORY_PATH
  NIFTI_DIRECTORY_PATH
  RAS_DIRECTORY_PATH
  COREG_DIRECTORY_PATH
  INPUTS_DIRECTORY_PATH
)

optional_reg_vars=(REGISTRY_URL IMAGE_REPOSITORY IMAGE_TAG)
for var in "${optional_reg_vars[@]}"; do
  export "${var:-}"
done

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

# ── Build the intended image reference ───────────────────────────
# Registry URL is optional (e.g. local Docker build-and-run); only prefix it
# when set so the result is always a well-formed reference.
image_ref() {
  local repo="${IMAGE_REPOSITORY:-mri_preprocessing}"
  local tag="${IMAGE_TAG:-latest}"
  if [ -n "${REGISTRY_URL:-}" ]; then
    echo "${REGISTRY_URL%/}/${repo}:${tag}"
  else
    echo "${repo}:${tag}"
  fi
}

# ── Deployment finalizer (runs on exit / Ctrl-C) ────────────────
# All supported runtimes spawn a long-lived foreground process, so sessions
# normally end with a signal; the EXIT trap is the one place we can reliably
# record what actually ran. Appends a "runtime" section to the manifest with
# the runtime-resolved identity (image ID / sif / env), complementing the
# *intended* reference written at start.
deployment_finalize() {
  [ "${FINALIZED:-false}" = true ] && return 0
  FINALIZED=true

  local manifest="${DEPLOY_LOG_DIR:-}/manifest.json"
  [ -n "${DEPLOY_LOG_DIR:-}" ] || return 0
  [ -f "$manifest" ] || return 0
  command -v python3 &>/dev/null || return 0
  [ -n "${RUNTIME:-}" ] || return 0

  local image_id="unresolved" sif_bytes=0 env_sha="none"
  case "$RUNTIME" in
    docker|docker-compose)
      if [ -n "${CONTAINER_NAME:-}" ] && docker inspect "${CONTAINER_NAME}" &>/dev/null; then
        image_id=$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}" 2>/dev/null || echo unresolved)
      fi
      python3 - "$manifest" "docker" "${CONTAINER_NAME:-}" "${COMPOSE_PROJECT_NAME:-}" "$image_id" <<'PYEOF'
import json, sys
path, rtype, container, project, image_id = sys.argv[1:6]
m = json.load(open(path))
m["runtime"] = {
    "type": rtype,
    "container_name": container or None,
    "compose_project": project or None,
    "resolved_image_id": image_id,
}
with open(path, "w") as f:
    json.dump(m, f, indent=2)
    f.write("\n")
PYEOF
      ;;
    singularity|apptainer)
      [ -f "${SIF_IMAGE:-}" ] && sif_bytes=$(stat -c %s "${SIF_IMAGE}" 2>/dev/null || echo 0)
      python3 - "$manifest" "$RUNTIME" "${SIF_IMAGE:-}" "$sif_bytes" <<'PYEOF'
import json, sys
path, rtype, sif, size = sys.argv[1:5]
m = json.load(open(path))
m["runtime"] = {
    "type": rtype,
    "sif_path": sif or None,
    "sif_bytes": int(size),
}
with open(path, "w") as f:
    json.dump(m, f, indent=2)
    f.write("\n")
PYEOF
      ;;
    conda|mamba)
      if [ -f "${project_directory_path}/environment.yml" ]; then
        env_sha=$(sha256sum "${project_directory_path}/environment.yml" | awk '{print $1}')
      fi
      python3 - "$manifest" "$RUNTIME" "${CONDA_ENV_NAME:-mri_preproc}" "$env_sha" <<'PYEOF'
import json, sys
path, rtype, env_name, env_sha = sys.argv[1:5]
m = json.load(open(path))
m["runtime"] = {
    "type": rtype,
    "conda_env": env_name,
    "environment_yml_sha256": env_sha,
}
with open(path, "w") as f:
    json.dump(m, f, indent=2)
    f.write("\n")
PYEOF
      ;;
  esac
}

# ── Create timestamped deployment log directory ─────────────────
DEPLOYMENT_ID=$(date +%Y%m%d_%H%M%S)
DEPLOY_LOG_DIR="${project_directory_path}/deployments/${DEPLOYMENT_ID}"
mkdir -p "$DEPLOY_LOG_DIR"

# Record which source tree this deployment started from.
# (The runtime-detected image/container identity is appended later at exit,
# see deployment_finalize().)
GIT_COMMIT="unknown"
GIT_DIRTY=true
if git -C "$project_directory_path" rev-parse --git-dir &>/dev/null; then
  GIT_COMMIT=$(git -C "$project_directory_path" rev-parse HEAD)
  if [ -z "$(git -C "$project_directory_path" status --porcelain)" ]; then
    GIT_DIRTY=false
  fi
fi

# Write a minimal manifest so deployments can be audited later
cat > "${DEPLOY_LOG_DIR}/manifest.json" <<MANIFEST
{
  "deployment_id": "${DEPLOYMENT_ID}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "runtime_env_file": "${ENV_FILE}",
  "image": "$(image_ref)",
  "source": {
    "git_commit": "${GIT_COMMIT}",
    "git_dirty": ${GIT_DIRTY}
  },
  "paths": {
    "raw_data": "${DATA_DIRECTORY_PATH}",
    "nifti": "${NIFTI_DIRECTORY_PATH}",
    "ras": "${RAS_DIRECTORY_PATH}",
    "coreg": "${COREG_DIRECTORY_PATH}",
    "inputs": "${INPUTS_DIRECTORY_PATH}"
  }
}
MANIFEST

# Snapshot the .env used for this deployment (so deployments are self-contained)
cp "$ENV_FILE" "${DEPLOY_LOG_DIR}/.env.snapshot"

export DEPLOY_LOG_DIR
# Record the runtime-resolved identity of what actually ran (image ID, sif,
# conda env) in the manifest when this session ends — normally via Ctrl-C.
trap 'deployment_finalize' EXIT INT TERM
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
  echo "         conda activate ${CONDA_ENV_NAME:-mri_preproc}"
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

    COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME}-${DEPLOYMENT_ID}"
    CONTAINER_NAME="control-${DEPLOYMENT_ID}"
    # Inside the container the deployment dir is mounted at /deployment.
    export LOG_DIR="/deployment/logs"
    export COMPOSE_PROJECT_NAME CONTAINER_NAME DEPLOY_LOG_DIR
    ${COMPOSE_CMD} -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" up --build
    ;;

  singularity|apptainer)
    # ── Validate the SIF path before attempting anything ───────────
    # A malformed SIF_PATH (a directory, an empty string, one containing
    # '=' or ' ' — the class of .env corruption that produced the
    # 'could not open image <garbage>' error) is diagnosed locally with
    # a readable message, not handed to apptainer to surface.
    SIF_IMAGE="${SIF_PATH:-./control_system/mri_preprocessing.sif}"
    if [ -z "$SIF_IMAGE" ]; then
      echo "ERROR: SIF_PATH is empty. Set SIF_PATH to the .sif file in .env"
      echo "       (e.g. SIF_PATH=$PWD/control_system/mri_preprocessing.sif)."
      exit 1
    fi
    if [ "${SIF_IMAGE: -1}" = "/" ]; then
      echo "ERROR: SIF_PATH is a directory: ${SIF_IMAGE}"
      echo "       Set SIF_PATH to a file path (…/mri_preprocessing.sif), not a directory."
      exit 1
    fi
    if [[ "$SIF_IMAGE" == *" "* || "$SIF_IMAGE" == *"="* ]]; then
      echo "ERROR: SIF_PATH contains whitespace or '=': ${SIF_IMAGE}"
      echo "       It may have been corrupted by .env parsing (two values on one line)."
      echo "       Check the SIF_PATH line in .env: it should be a single absolute path to a .sif file."
      exit 1
    fi
    # If it exists, it must be a regular file (not a directory).
    if [ -e "$SIF_IMAGE" ] && [ ! -f "$SIF_IMAGE" ]; then
      echo "ERROR: SIF_PATH exists but is not a regular file: ${SIF_IMAGE}"
      [ -d "$SIF_IMAGE" ] && echo "       It is a directory — that is not a valid .sif location."
      exit 1
    fi

    REGISTRY_REF="$(image_ref)"

    if [ ! -f "$SIF_IMAGE" ]; then
      # Ensure the parent directory exists (so the pull actually lands somewhere).
      sif_parent="$(dirname "$SIF_IMAGE")"
      if [ ! -d "$sif_parent" ]; then
        mkdir -p "$sif_parent" || { echo "ERROR: Cannot create ${sif_parent}/ for the .sif"; exit 1; }
      fi
      echo "No local .sif found. Pulling from registry: ${REGISTRY_REF}"
      echo "       (This can take a while for a large CUDA image — be patient.)"
      "${RUNTIME}" pull "$SIF_IMAGE" "docker://${REGISTRY_REF}" || {
        echo ""
        echo "ERROR: Failed to pull image from ${REGISTRY_REF}"
        echo "Check that the registry URL is correct and accessible, and that"
        echo "this node has outbound network access."
        exit 1
      }
      echo "Image cached at ${SIF_IMAGE}"
    fi
    # Final sanity: the file the exec command will reference must now be a
    # non-empty regular file. This is the exact failure that produced the
    # user's 'could not open image' error — catch it here with a clean message.
    if [ ! -f "$SIF_IMAGE" ] || [ ! -s "$SIF_IMAGE" ] || [ ! -r "$SIF_IMAGE" ]; then
      echo "ERROR: SIF image is not a readable, non-empty file: ${SIF_IMAGE}"
      rm -f "$SIF_IMAGE" 2>/dev/null || true
      echo "       Removed the partial/empty file; re-run to retry the pull."
      exit 1
    fi

    # Writable base for /FL_system/data. The pipeline writes to the data
    # TOP-LEVEL (step 03 progress pickles, Data_table*.csv), to /FL_system/data/tmp
    # (01/02 scratch), and to checkpoints — none of which are covered by the five
    # subdirectory binds below. Mount a real host directory as the base first,
    # then overlay the five per-purpose dirs on top (same layering docker-compose.yml uses).
    DATA_BASE_DIR="${project_directory_path}/mri_data_base"
    mkdir -p "${DATA_BASE_DIR}/tmp" || {
      echo "ERROR: Cannot create writable data base at ${DATA_BASE_DIR}."
      echo "       The pipeline needs a writable host dir backing /FL_system/data."
      exit 1
    }
    if [ ! -w "$DATA_BASE_DIR" ]; then
      echo "ERROR: Data base ${DATA_BASE_DIR} is not writable (check storage permissions/quota)."
      exit 1
    fi

    # ── Validate every data path before it can reach apptainer ──────
    # Each must be a non-empty absolute path, free of the .env-corruption
    # signatures ('=' from a merged line, or leading/trailing whitespace).
    _validate_path() {
      local name="$1" val="$2"
      if [ -z "$val" ]; then
        echo "ERROR: ${name} is empty in .env."
        return 1
      fi
      if [[ "$val" != /* ]]; then
        echo "ERROR: ${name} must be an absolute path, got: ${val}"
        return 1
      fi
      if [[ "$val" == *"="* ]]; then
        echo "ERROR: ${name} contains '=' — looks like two assignments on one line in .env."
        echo "       Current value: ${val}"
        return 1
      fi
      if [[ "$val" == " "* || "$val" == *" " ]]; then
        echo "WARNING: ${name} contains spaces: ${val}"
      fi
      if [ ! -d "$val" ]; then
        echo "WARNING: ${name} does not exist yet (will be created): ${val}"
      fi
      return 0
    }
    for p in DATA_DIRECTORY_PATH NIFTI_DIRECTORY_PATH RAS_DIRECTORY_PATH COREG_DIRECTORY_PATH INPUTS_DIRECTORY_PATH; do
      _validate_path "$p" "${!p}" || exit 1
    done

    Binds=(
      "${DATA_BASE_DIR}:/FL_system/data"
      "${DATA_DIRECTORY_PATH}:/FL_system/data/raw"
      "${NIFTI_DIRECTORY_PATH}:/FL_system/data/nifti"
      "${RAS_DIRECTORY_PATH}:/FL_system/data/RAS"
      "${COREG_DIRECTORY_PATH}:/FL_system/data/coreg"
      "${INPUTS_DIRECTORY_PATH}:/FL_system/data/inputs"
      "${DEPLOY_LOG_DIR}:/deployment/"
    )

    # Pass GPUs into the container only when the host node actually has a
    # GPU (nvidia-smi on PATH). On a CPU-only HPC partition, --nv causes
    # Singularity/Apptainer to fail or hang; reg_f3d falls back to CPU either
    # way, so omitting --nv is safe and matches the container's own GPU-less
    # path (see control_system/scripts/install_niftyreg_runtime.sh).
    Additional=( )
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
      Additional+=( --nv )
      echo "GPU detected on host — passing --nv into the container."
    else
      echo "No GPU detected on this host — launching without --nv (CPU-only)."
      echo "Coregistration (step 05) will fall back to CPU if niftyreg/CUDA is not loaded."
    fi

    # Use REPEATED --bind flags (one per mount) rather than a single
    # comma-joined string: classic Singularity 2.x requires the repeated
    # form, Apptainer/Singularity-CE accepts both. This also removes any
    # ambiguity about which token is the image, which is what surfaced the
    # user's 'could not open image …DATA_DIRECTORY_PATH=…' error.
    BindFlags=( )
    for b in "${Binds[@]}"; do BindFlags+=( --bind "$b" ); done
    EnvFlags=(
      -e DATA_DIRECTORY_PATH="$DATA_DIRECTORY_PATH"
      -e NIFTI_DIRECTORY_PATH="$NIFTI_DIRECTORY_PATH"
      -e RAS_DIRECTORY_PATH="$RAS_DIRECTORY_PATH"
      -e COREG_DIRECTORY_PATH="$COREG_DIRECTORY_PATH"
      -e INPUTS_DIRECTORY_PATH="$INPUTS_DIRECTORY_PATH"
      -e LOG_DIR="/deployment/logs"
    )

    echo "Using ${RUNTIME} with image: $SIF_IMAGE"
    echo "Registry reference: ${REGISTRY_REF}"
    echo "Bindings:"
    for b in "${Binds[@]}"; do echo "  ${b}"; done
    echo ""
    echo "Pipeline scripts are baked into the image."
    echo "Once the prompt appears, run:"
    echo "  python code/preprocessing/01_scanDicom.py --scan-dir /FL_system/data/raw --save-dir /FL_system/data"
    echo "  bash code/preprocessing/00_preprocess.sh              (runs all steps)"
    echo ""

    # Record the exact command into the deployment manifest for offline
    # reproduction, then launch.
    {
      printf 'exec: '
      for tok in ${RUNTIME} ${Additional[@]+"${Additional[@]}"} ${BindFlags[@]+"${BindFlags[@]}"} ${EnvFlags[@]+"${EnvFlags[@]}"} --pwd /FL_system "$SIF_IMAGE" bash; do
        printf '%q ' "$tok"
      done
      printf '\n'
    } >> "${DEPLOY_LOG_DIR}/manifest_exec.txt"

    ${RUNTIME} exec \
      ${Additional[@]+"${Additional[@]}"} \
      ${BindFlags[@]+"${BindFlags[@]}"} \
      ${EnvFlags[@]+"${EnvFlags[@]}"} \
      --pwd /FL_system \
      "$SIF_IMAGE" bash
    ;;

  conda|mamba)
    ENV_YML="${script_directory}/environment.yml"
    ENV_NAME="${CONDA_ENV_NAME:-mri_preproc}"

    # Host-local deployment log dir (no container mount for bare conda runs).
    export LOG_DIR="${DEPLOY_LOG_DIR}/logs"

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