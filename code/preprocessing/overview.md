This directory contains all necessary scripts to perform pre-processing of the supplied MRI data.

The scripts are run inside of the control server docker container. The following scripts are provided:
    - '00_preprocess.sh' : Shell script to run all pre-processing steps, reports completion status for dashboard updates.
    - '01_scanDicom.py' : Python script to scan and extract DICOM data from the headers of the supplied MRI data.
        - Assumes data is provided in a directory for each subject, with an additional directory for each scan.
        - Outputs a 'Data_table.csv' containing all extracted DICOM information.
    - '02_parseDicom.py' : Python script to parse the extracted DICOM data and generate a CSV file.
        - This script filters out unnecessary scans, and generates a 'Data_table_timing.csv' file.
        - This file contains the timing information for each scan, in the format of Major and Minor blocks.
    - '03_saveNifti.py' : Converts the required DICOM directories into nifti files.
        - This script uses the 'Data_table_timing.csv' file to determine which scans to convert.
        - Saves files in '/data/nifti/{sesionID}/' directory and names them according to the Major/Minor ordering.
    - '04_saveRAS.py' : Converts the nifti files into RAS orientation.
        - This script uses the 'Data_table_timing.csv' file to determine which scans to convert.
        - Saves files in '/data/nifti/{sessionID}/' directory and names them according to the Major/Minor ordering.
    - '05_alignScans.py' : Converts the RAS oriented nifti files into BIDS format.
    - '06_genInputs.py' : Generates the input files for the next stage of the pipeline.
        - This script generates the 'Data_table_inputs.csv' file, which contains the paths to the BIDS formatted nifti files.