#!/usr/bin/env bash
set -euo pipefail

script_directory=$(dirname "$(readlink -f "$0")")
ENV_FILE="${script_directory}/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: .env file not found at ${ENV_FILE}"
  exit 1
fi

while IFS='=' read -r key value || [ -n "$key" ]; do
  [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
  export "${key}=${value}"
done < "$ENV_FILE"

container="${CONTAINER_NAME:-control}"
project="${COMPOSE_PROJECT_NAME:-MRI_preprocessing}"

echo "MRI Preprocessing - Direct CLI Access (${container})"
echo "====================================================="
echo ""

if ! docker container ls --project="${project}" --format "{{.Names}}" | grep -q "^${container}$"; then
  echo "Error: Container ${container} in project ${project} is not running."
  echo "Start it first with: bash start_control.sh"
  exit 1
fi

echo "Accessing container: ${container}"
echo "Navigate to /FL_system/code/preprocessing/ to run preprocessing scripts."
echo ""

docker exec -it "${container}" bash
