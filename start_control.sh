echo "This script must be run from the base project directory"
echo "i.e. the directory containing the start_control.sh file itself"

# Prompt the user for the data directory path
echo "Please enter the raw data path:"
read data_directory_path

# Determine the script's directory
script_directory=$(dirname "$(readlink -f "$0")")
project_directory_path=$(realpath "$script_directory/")
echo "Project directory path: ${project_directory_path}"

# Exporting environmental variables to allow the container the knowledge of its location and the data location on the base machine
# Project Path
export PROJECT_DIRECTORY_PATH="${project_directory_path}"
# Raw Data Path
export DATA_DIRECTORY_PATH="${data_directory_path}"

docker network create flwr-network

# Check if running in WSL, WSL2, or Linux
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

# Use the provided path as a volume in Docker Compose
# Previously exported paths are used as environment variables in the docker-compose.yml files
if [ "$WSL" = true ]; then
  echo "Using docker-compose-wsl.yml"
  docker compose -f ./control_system/docker-compose-wsl.yml up --build
else
    echo "Using docker-compose.yml"
  docker compose -f ./control_system/docker-compose.yml up --build
fi
