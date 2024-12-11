#!/bin/bash

python /FL_system/code/preprocessing/01_scanDicom.py
echo "01 Completed" # Used by script.js to check status of the process
python /FL_system/code/preprocessing/02_parseDicom.py
echo "02 Completed" # Used by script.js to check status of the process
python /FL_system/code/preprocessing/03_saveNifti.py
echo "03 Completed" # Used by script.js to check status of the process
python /FL_system/code/preprocessing/04_saveRAS.py
echo "04 Completed" # Used by script.js to check status of the process
python /FL_system/code/preprocessing/05_alignScans.py
echo "05 Completed" # Used by script.js to check status of the process
python /FL_system/code/preprocessing/06_genInputs.py
echo "06 Completed" # Used by script.js to check status of the process