# Package imports
import os
import time
import argparse
import pydicom as pyd
import numpy as np
import pandas as pd
# Function imports
from multiprocessing import Manager, cpu_count
from typing import Callable, List, Any
from functools import partial
from concurrent.futures import ThreadPoolExecutor
# Custom imports
from toolbox import get_logger
from DICOM import DICOMextract

parser = argparse.ArgumentParser(description='Scan DICOM files and extract information')
parser.add_argument('-m', '--multiprocess', action='store_true', help='Use multiprocessing')
parser.add_argument ('-d', '--debug', type=int, help='0 for no debug, 1 for debug, 2 for verbose', default=0)
args = parser.parse_args()
# Define necessary parameters
SAVE_DIR = '/FL_system/data/' # Location to save the constructed Data_table.csv
SCAN_DIR = '/FL_system/data/raw/' # Location to recursively scan for dicom files
TEST = False # If True, only the first 2 dicom files will be scanned
N_TEST = 100 # Number of dicom files to scan if TEST is True
PARALLEL = args.multiprocess # If True, the script will run in parallel
DEBUG = args.debug # Debug level for logging
LOGGER = get_logger('01_scanDicom', f'{SAVE_DIR}/logs/')

# Profiler
PROFILE = False
if PROFILE:
    import yappi
    import pstats
    import io

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

def run_function(target: Callable[..., Any], items: List[Any], Parallel: bool=True, *args, **kwargs) -> List[Any]:
    """Run a function with a progress bar"""

    target_name = target.func.__name__ if isinstance(target, partial) else target.__name__

    # Debugging information
    LOGGER.debug(f'Running {target_name} {" in parallel" if Parallel else "sequentially"}')
    LOGGER.debug(f'Number of items: {len(items)}')

    # Run the target function with a progress bar
    results = []
    if Parallel:
        max_workers = min(32, 2 * cpu_count())
        LOGGER.debug(f'Using ThreadPoolExecutor with max_workers={max_workers}')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(target, item, *args, **kwargs) for item in items]
            for i, future in enumerate(futures):
                try:
                    LOGGER.debug(f'Waiting for future {i} to complete') 
                    result = future.result(timeout=300)
                    results.append(result)
                    LOGGER.debug(f'Future {i} completed successfully')
                except TimeoutError:
                    LOGGER.error(f'Timeout error for item {i}')
                except Exception as e:
                    LOGGER.error(f'Error in parallel processing for item {i}: {e}')
    else:
        for item in items:
            try:
                result = target(item)
                results.append(result)
            except Exception as e:
                LOGGER.error(f'Error in sequential processing: {e}')


    LOGGER.debug(f'Completed {target_name} {" in parallel" if Parallel else "sequentially"}')
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
            'ScanDur': extract.ScanDur(),
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

def find_all_dicom_dirs(directory, debug=0):
    """
    Find all directories containing DICOM files (.dcm) in the given directory.

    Args:
        directory (str): The root directory to search.
        debug (int): Debug level for logging.

    Returns:
        List[str]: A list of directories containing DICOM files.
    """
    dicom_dirs = []

    for root, _, files in os.walk(directory, followlinks=False):
        # Check if any file in the current directory ends with '.dcm'
        if any(file.endswith('.dcm') for file in files):
            dicom_dirs.append(root)
            if debug > 0:
                LOGGER.debug(f'Found DICOM files in {root}')

    if not dicom_dirs:
        LOGGER.warning(f'No directories containing DICOM files found in {directory}')
    else:
        LOGGER.info(f'Found {len(dicom_dirs)} directories containing DICOM files')

    return dicom_dirs

def findDicom(directory, debug=0):
    # Find all dicom files in the directory through a recursive search
    # Will only record the first file found for each directory if files have the same series number
    # TODO: upgrade series detection to not look for the first 2 files but distant files
    
    dicom_files = []

    for root, dirs, files in os.walk(directory):
        found_series = []
        for file in files:
            if file.endswith('.dcm'):
                file_path = os.path.join(root, file)
                start_time = time.time()
                try:
                    data = pyd.dcmread(file_path, stop_before_pixels=True, force=True)
                except Exception as e:
                    LOGGER.error(f'Error reading file {file} in folder {root} | {e}')
                    continue
                if debug > 0:
                    LOGGER.debug(f'File {file} read in {time.time() - start_time:.2f} seconds')
                if data.SeriesNumber in found_series:
                    continue
                else:
                    dicom_files.append(os.path.join(root, file))
                    found_series.append(data.SeriesNumber)       
        LOGGER.debug(f'{root} contains series {found_series} | {len(found_series)} series found')
    return dicom_files

#############################
## Main script
#############################
def main():
    # Print the current configuration
    LOGGER.info('Starting scanDicom: Step 01')
    LOGGER.info(f'SCAN_DIR: {SCAN_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if PROFILE:
        LOGGER.info(f'Profiling is enabled')

    # Check if the Data_table.csv already exists
    if 'Data_table.csv' in os.listdir(SAVE_DIR):
        LOGGER.error('Data_table.csv already exists. Skipping step 01')
        LOGGER.error('To re-run this step, delete the existing Data_table.csv file')
        exit()

    # Finding main directory and subdirectories
    LOGGER.info('Finding all directories containing DICOM files')
    dicom_dirs = find_all_dicom_dirs(SCAN_DIR, debug=DEBUG)
    if TEST:
        dicom_dirs = dicom_dirs[:N_TEST]
        LOGGER.info(f'Running in test mode with {N_TEST} directories')
                    
    # Scan the directories for dicom files
    LOGGER.info('Analyzing DICOM directories')
    dicom_files = run_function(findDicom, dicom_dirs, Parallel=PARALLEL, debug=DEBUG)
    dicom_files = [f for sublist in dicom_files for f in sublist] # Flatten the list of lists
    LOGGER.info(f'Found {len(dicom_files)} dicom files in the input directory')

    # Extract the dicom information
    LOGGER.info('Extracting information from dicom files')
    INFO = run_function(extractDicom, dicom_files, Parallel=PARALLEL, debug=DEBUG)
    Data_table = pd.DataFrame(INFO) # Convert the extracted information to a pandas dataframe
    Data_table.to_csv(f'{SAVE_DIR}Data_table.csv', index=False) # Save the extracted information to a csv file
    LOGGER.info('DICOM information extraction completed and saved to Data_table.csv')

if __name__ == '__main__':
    if PROFILE:
        LOGGER.info('Profiling enabled')
        yappi.start()
        LOGGER.info('Starting main function')
        main()
        LOGGER.info('Main function completed')
        yappi.stop()
        profile_output_path = 'step01_profile.yappi'
        LOGGER.info(f'Writing profile results to {profile_output_path}')
        yappi.get_func_stats().save(profile_output_path, type='pstat')
        LOGGER.info(f'Profile results saved to {profile_output_path}')
    else:
        main()
    exit()