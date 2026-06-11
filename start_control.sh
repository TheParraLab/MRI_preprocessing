# Start the MRI Preprocessing container

# Prompt the user for paths
echo "Please enter the raw data path:"
read data_directory_path

echo "Please enter the NIfTI output path:"
read nifti_directory_path

# Determine the script's directory
script_directory=$(dirname "$(readlink -f "$0")")
project_directory_path=$(realpath "$script_directory/")

# Export environment variables
export PROJECT_DIRECTORY_PATH="${project_directory_path}"
export DATA_DIRECTORY_PATH="${data_directory_path}"
export NIFTI_DIRECTORY_PATH="${nifti_directory_path}"

# Detect platform (WSL check)
if grep -qi Microsoft /proc/version; then
  echo "Running on WSL"
  WSL=true
elif grep -qi WSL /proc/version; then
  echo "Running on WSL 2"
  WSL=true
else
  echo "Running on pure Linux"
  WSL=false
fi

# Auto-detect container runtime: prefer docker, fallback to singularity/apptainer
detect_runtime() {
  if command -v docker &>/dev/null && docker info &>/dev/null; then
    # Check docker-compose availability
    if command -v docker compose &>/dev/null; then
      echo "docker"
      return 0
    elif command -v docker-compose &>/dev/null; then
      echo "docker-compose"
      return 0
    fi
  fi
  # Fallback: check for singularity or apptainer (most modern HPC use Apptainer)
  if command -v singularity &>/dev/null; then
    echo "singularity"
    return 0
  elif command -v apptainer &>/dev/null; then
    echo "apptainer"
    return 0
  fi
  return 1
}

RUNTIME=$(detect_runtime) || {
  echo ""
  echo "ERROR: No container runtime found. Please install one of:"
  echo ""
  echo "DOCKER (recommended for development):"
  echo "  https://docs.docker.com/get-docker/"
  echo ""
  echo "SINGULARITY/APPTAINER (for HPC clusters):"
  echo "  https://apptainer.org/docs/user/latest/quick_start.html#installation"
  echo ""
  echo "Then build or copy a .sif image to your site:"
  echo "  cd ${script_directory}"
  echo "  sudo singularity build control_system/mri_preprocessing.sif control_system/mri_preprocessing.singularity.def"
  echo "  # OR copy an existing .sif file here instead"
  exit 1
}

# Start the container
case "$RUNTIME" in
  docker|docker-compose)
    if [ "$WSL" = true ]; then
      echo "Using Docker (WSL): docker-compose-wsl.yml"
      COMPOSE_CMD=$(command -v docker compose &>/dev/null && echo "docker compose" || echo "docker-compose")
      ${COMPOSE_CMD} -f ./control_system/docker-compose-wsl.yml up --build
    else
      echo "Using Docker: docker-compose.yml"
      COMPOSE_CMD=$(command -v docker compose &>/dev/null && echo "docker compose" || echo "docker-compose")
      ${COMPOSE_CMD} -f ./control_system/docker-compose.yml up --build
    fi
    ;;
  singularity|apptainer)
    SING_CMD="$RUNTIME"
    
    # Check for .sif image
    SIF_IMAGE="./control_system/mri_preprocessing.sif"
    if [ ! -f "$SIF_IMAGE" ]; then
      echo "WARNING: Singularity image not found at $SIF_IMAGE"
      echo ""
      echo "To deploy on HPC sites without Docker, build or copy a .sif image:"
      echo ""
      echo "  Option A — Build locally (requires root):"
      echo "    sudo singularity build mri_preprocessing.sif control_system/mri_preprocessing.singularity.def"
      echo ""
      echo "  Option B — Pull from an existing Docker/OCI image:"
      echo "    ${SING_CMD} pull mri_preprocessing.sif docker://<your-image>:tag"
      echo ""
      echo "Then copy the .sif file to control_system/ on your HPC site."
      echo ""
      exit 1
    fi
    
    # Resolve paths for binding
    RAW_BIND="${DATA_DIRECTORY_PATH}:/FL_system/data/raw"
    PROJECT_BIND="${PROJECT_DIRECTORY_PATH}:/FL_system"
    
    echo "Using ${RUNTIME} with image: $SIF_IMAGE"
    echo "Binding raw data:    $DATA_DIRECTORY_PATH → /FL_system/data/raw"
    echo "Binding project dirs:$PROJECT_DIRECTORY_PATH → /FL_system"
    echo ""
    echo "Once the prompt appears, run your pipeline scripts inside the container:"
    echo "  python code/preprocessing/01_scanDicom.py --scan_dir /FL_system/data/raw --save_dir /FL_system/data"
    echo "  bash code/preprocessing/00_preprocess.sh              (runs all steps)"
    echo ""
    
    # Remove nvidia runtime option for singularity — GPU is handled via system CUDA packages instead
    ${SING_CMD} exec \
      --bind "$RAW_BIND,$PROJECT_BIND" \
      -e DATA_DIRECTORY_PATH="$DATA_DIRECTORY_PATH" \
      -e NIFTI_DIRECTORY_PATH="$NIFTI_DIRECTORY_PATH" \
      -e PROJECT_DIRECTORY_PATH="$PROJECT_DIRECTORY_PATH" \
      "$SIF_IMAGE" bash
    ;;
esac
