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

# Detect platform
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

# Start the container
if [ "$WSL" = true ]; then
  echo "Using docker-compose-wsl.yml"
  docker compose -f ./control_system/docker-compose-wsl.yml up --build
else
  echo "Using docker-compose.yml"
  docker compose -f ./control_system/docker-compose.yml up --build
fi
