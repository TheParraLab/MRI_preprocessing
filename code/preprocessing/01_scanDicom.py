# Package imports
import os
import threading
import glob
import pydicom as pyd
import numpy as np
import pandas as pd
# Function imports
from multiprocessing import Queue, Manager, cpu_count
from typing import Callable, List, Any
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
# Custom imports
from toolbox import ProgressBar, get_logger
from DICOM import DICOMextract

# Global variables for progress bar
Progress = None
manager = Manager()
progress_queue = manager.Queue()

# Define necessary parameters
SAVE_DIR = '/FL_system/data/' # Location to save the constructed Data_table.csv
SCAN_DIR = '/FL_system/data/raw/' # Location to recursively scan for dicom files
TEST = False # If True, only the first 2 dicom files will be scanned
N_TEST = 10 # Number of dicom files to scan if TEST is True
PARALLEL = True # If True, the scan will be parallelized
LOGGER = get_logger('01_scanDicom', f'{SAVE_DIR}/logs/')
DEBUG = 0

#### Preprocessing | Step 1: Extract DICOM data ####
# This script scans the input directory for dicom files and extracts necessary header information
#
# The extracted information is saved to /FL_system/data/Data_table.csv
#
# This step will begin by performing a recursive search for all dicom files in the input directory
# The first 2 files found in each directory will be checked to ensure they are from the same series
# TODO: Add detection for how many series are present in a directory, maybe look at first and last file and check series #?
#   - If they are, only the first file will be recorded
#   - If they are not, all files will be recorded
# After the scan, the following information will be extracted from the dicom headers:
#   - PatientID
#   - StudyDate
#   - Path
#   - SeriesDescription
#   - AcquisitionTime
#   - SeriesTime
#   - ContentTime
#   - StudyTime
#   - TriggerTime
#   - InjectionTime
#   - Modality
#   - Laterality
#   - NumSlices
#   - Orientation
#   - SliceThickness
#   - BreastSize
#   - DiffusionBValue
#   - ImageType
#   - SeriesNumber
# The extracted information will be saved to /data/Data_table.csv
# NOTE: This script is expected to run inside of the provided control_system docker container and called through the available web interface

#########################################
## Parallelization and Progress functions
#########################################
def progress_updater(queue, progress_bar):
    while True:
        item = queue.get()
        if item is None:
            break
        index, status = item
        progress_bar.update(index, status)

        queue.task_done()

def progress_wrapper(item, target, progress_queue, *args, **kwargs):
    result = target(item, *args, **kwargs)
    progress_queue.put((None, f'Processing'))
    return result

def run_with_progress(target: Callable[..., Any], items: List[Any], Parallel: bool=True, *args, **kwargs) -> List[Any]:
    """Run a function with a progress bar"""
    # Initialize using a manager to allow for shared progress queue
    manager = Manager()
    progress_queue = manager.Queue()
    target_name = target.func.__name__ if isinstance(target, partial) else target.__name__

    # Debugging information
    LOGGER.debug(f'Running {target_name} with progress bar')
    LOGGER.debug(f'Number of items: {len(items)}')
    LOGGER.debug(f'Parallel: {Parallel}')

    # Initialize progress bar
    Progress = ProgressBar(len(items))
    updater_thread = threading.Thread(target=progress_updater, args=(progress_queue, Progress))
    updater_thread.start()
    
    # Pass the progress queue to the target function
    target = partial(progress_wrapper, target=target, progress_queue=progress_queue, *args, **kwargs)

    # Run the target function with a progress bar
    results = []
    if Parallel:
        with ProcessPoolExecutor(max_workers=cpu_count()-1) as executor:
            futures = [executor.submit(target, item, *args, **kwargs) for item in items]
            for future in futures:
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    LOGGER.error(f'Error in parallel processing: {e}')
    else:
        for item in items:
            try:
                result = target(item)
                results.append(result)
            except Exception as e:
                LOGGER.error(f'Error in sequential processing: {e}')

    # Close the progress bar
    progress_queue.put(None)
    print('\n')
    updater_thread.join()

    LOGGER.debug(f'Completed {target_name} with progress bar')
    LOGGER.debug(f'Number of results: {len(results)}')

    # Check if results is a list of tuples before returning zip(*results)
    if results and isinstance(results[0], tuple):
        return zip(*results)
    return results

#############################
## Main functions
#############################
def extractDicom(f, debug=0):
    try:
        LOGGER.debug(f'Extracting information for file: {f}')
        extract = DICOMextract(f)

        result = {
            'PATH': f,
            'Orientation': extract.Orientation(),
            'ID': extract.ID(),
            'DATE': extract.Date(),
            'Series_desc': extract.Desc(),
            'Modality': extract.Modality(),
            'AcqTime': extract.Acq(),
            'SrsTime': extract.Srs(),
            'ConTime': extract.Con(),
            'StuTime': extract.Stu(),
            'TriTime': extract.Tri(),
            'InjTime': extract.Inj(),
            'Lat': extract.LR(),
            'NumSlices': extract.NumSlices(),
            'Thickness': extract.Thickness(),
            'BreastSize': extract.BreastSize(),
            'DWI': extract.DWI(),
            'Type': extract.Type(),
            'Series': extract.Series()
        }
        LOGGER.debug(f'Completed extraction for file: {f}')
        return result
    except Exception as e:
        LOGGER.error(f'Error extracting information for file: {f} | {e}')
        return None

def find_multiple_subdirs(directory, debug=0):
    # Find the first subdirectory with multiple subdirectories
    # Targets the starting directory to parallelize
    for root, dirs, _ in os.walk(directory):
        if len(dirs) > 1:
            LOGGER.debug(f'Found multiple subdirectories in {root}')
            return root, dirs
    LOGGER.debug(f'No subdirectories with multiple subdirectories found in {directory}')
    return None, None

def findDicom(directory, debug=0):
    # Find all dicom files in the directory through a recursive search
    # Will only record the first file found for each directory if files have the same series number
    # TODO: upgrade series detection to not look for the first 2 files but distant files
    
    dicom_files = []

    for root, dirs, files in os.walk(directory):
        i = 0
        for file in files:
            if file.endswith('.dcm'):
                # Check if first 2 samples are from the same series
                if i == 0:
                    series = pyd.dcmread(os.path.join(root, file)).SeriesNumber
                elif i == 1:
                    if series == pyd.dcmread(os.path.join(root, file)).SeriesNumber:
                        # If both are, only record the first file found
                        LOGGER.debug(f'folder {root} contains multiple files for a single series')
                        LOGGER.debug(f'Only the first file will be recorded')
                        break # Move to next directory
                    else:
                        LOGGER.debug(f'folder {root} contains multiple series')
                        LOGGER.debug(f'All files will be recorded')
                dicom_files.append(os.path.join(root, file))
                i+=1
    return dicom_files

#############################
## Main script
#############################
if __name__ == '__main__':
    # Print the current configuration
    LOGGER.info('Starting scanDicom: Step 01')
    LOGGER.info(f'SCAN_DIR: {SCAN_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if TEST:
        LOGGER.info(f'Running in test mode: {TEST}')
        LOGGER.info(f'N_TEST: {N_TEST}')

    # Check if the Data_table.csv already exists
    if 'Data_table.csv' in os.listdir(SAVE_DIR):
        LOGGER.error('Data_table.csv already exists. Skipping step 01')
        LOGGER.error('To re-run this step, delete the existing Data_table.csv file')
        exit()

    # Finding main directory and subdirectories
    root, dirs = find_multiple_subdirs(SCAN_DIR, debug=DEBUG)
    directories = [os.path.join(root, d) for d in dirs] # List of subdirectories to scan
    if TEST:
        directories = directories[:N_TEST]
        LOGGER.info(f'Running in test mode with {N_TEST} directories')
                    
    # Scan the directories for dicom files
    LOGGER.info('Scanning directories for dicom files')
    dicom_files = run_with_progress(findDicom, directories, Parallel=PARALLEL, debug=DEBUG)
    dicom_files = [f for sublist in dicom_files for f in sublist] # Flatten the list of lists
    LOGGER.info(f'Found {len(dicom_files)} dicom files in the input directory')

    # Extract the dicom information
    LOGGER.info('Extracting information from dicom files')
    INFO = run_with_progress(extractDicom, dicom_files, Parallel=PARALLEL, debug=DEBUG)
    Data_table = pd.DataFrame(INFO) # Convert the extracted information to a pandas dataframe
    Data_table.to_csv(f'{SAVE_DIR}Data_table.csv', index=False) # Save the extracted information to a csv file
    LOGGER.info('DICOM information extraction completed and saved to Data_table.csv')
    exit()