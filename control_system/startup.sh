#!/bin/bash

if [ "$NO_WEBSERVER" = "true" ]; then
    echo "MRI Preprocessing container started without webserver"
    echo "Container is ready for preprocessing tasks"
    echo "You can execute preprocessing commands by running:"
    echo "  docker exec -it control bash"
    echo "  Then navigate to /FL_system/code/preprocessing/ to run preprocessing scripts"
    # Keep container running
    tail -f /dev/null
else
    echo "Starting MRI Preprocessing with webserver on port 5000"
    cd /app
    python app.py
fi
