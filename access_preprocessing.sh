#!/bin/bash

echo "MRI Preprocessing - Direct CLI Access"
echo "====================================="
echo ""
echo "This script provides direct access to the preprocessing container"
echo "without starting the webserver component."
echo ""

# Check if the control container is running
if ! docker ps --format "table {{.Names}}" | grep -q "^control$"; then
    echo "Error: The control container is not running."
    echo "Please start the system first with: bash start_control.sh"
    echo "And choose 'n' when asked about the webserver component."
    exit 1
fi

echo "Accessing the control container..."
echo "You are now in the preprocessing environment."
echo "Navigate to /FL_system/code/preprocessing/ to run preprocessing scripts."
echo ""

# Execute interactive bash session in the container
docker exec -it control bash
