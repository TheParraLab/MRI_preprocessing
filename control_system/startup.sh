#!/bin/bash

echo "MRI Preprocessing container started"
echo "Container is ready for preprocessing tasks"
echo "You can execute preprocessing commands by running:"
echo "  docker exec -it control bash"
echo "  Then navigate to /FL_system/code/preprocessing/ to run preprocessing scripts"

# Keep container running
tail -f /dev/null
