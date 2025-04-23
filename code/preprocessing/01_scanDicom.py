# Standard imports
import os
import time
import argparse
# Third-party imports
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

# Global variables for progress bar
Progress = None
manager = Manager()
progress_queue = manager.Queue()

# Define command line arguments
parser = argparse.ArgumentParser(description='Extract DICOM data to build Data_table.csv')
parser.add_argument('--test', nargs='?', const=100, type=int, help='Run in test mode with an optional number of dicom directories to scan (default: 100)')
parser.add_argument('--multi', '-m', nargs='?', const=cpu_count()-1, type=int, help='Run with multiprocessing enabled, using provided number of cpus (default: max-1)')
parser.add_argument('-p', '--profile', action='store_true', help='Run with profiler enabled')
parser.add_argument('--save_dir', nargs='?', default='/FL_system/data/', type=str, help='Location to save the constructed Data_table.csv (default: /FL_system/data/)')
parser.add_argument('--scan_dir', nargs='?', default='/FL_system/data/raw/', type=str, help='Location to recursively scan for dicom files (default: /FL_system/data/raw/)')
parser.add_argument('--dir_idx', type=int, help='Index of the folder to process from dirs_to_process.txt')
parser.add_argument('--dir_list', type=str, default='dirs_to_process.txt', help='Path to the directory list file')
args = parser.parse_args()

# Apply cli arguments
SAVE_DIR = args.save_dir
SCAN_DIR = args.scan_dir
TEST = args.test is not None # If True, the script will run in test mode
N_TEST = args.test if TEST else 100 # Number of dicom directories to scan if TEST is True
PARALLEL = args.multi is not None # If True, the script will run with multiprocessing enabled
N_CPUS = args.multi if PARALLEL else cpu_count()-1 # Number of cpus to use if PARALLEL is True
PROFILE = args.profile # If True, the script will run with the profiler enabled
# Profiler
if PROFILE:
    import yappi
    import pstats
    import io

# Define necessary parameters
LOGGER = get_logger('01_scanDicom', f'{SAVE_DIR}/logs/')
DEBUG = 0


#### Preprocessing | Step 1: Extract DICOM data ####
# This script scans the input directory for dicom files and extracts necessary header information
#
# The extracted information is saved to /FL_system/data/Data_table.csv
#
# This step will begin by performing a recursive search for all dicom files in the input directory

# The extracted information will be saved to /data/Data_table.csv
# NOTE: This script is expected to run inside of the provided control_system docker container and called through the available web interface

#########################################
## Parallelization and Progress functions
#########################################

def run_function(target: Callable[..., Any], items: List[Any], Parallel: bool=True, N_CPUS: int=0, *args, **kwargs) -> List[Any]:
    """Run a function with a progress bar"""

    target_name = target.func.__name__ if isinstance(target, partial) else target.__name__
    if N_CPUS == 0:
        N_CPUS = cpu_count() - 1
    else:
        N_CPUS = min(N_CPUS, cpu_count() - 1)
            
    # Debugging information
    LOGGER.debug(f'Running {target_name} {" in parallel" if Parallel else "sequentially"}')
    LOGGER.debug(f'Number of items: {len(items)}')

    # Run the target function with a progress bar
    results = []
    if Parallel:
        max_workers = min(32, 2 * N_CPUS)
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
def main(out_name='Data_table.csv', SAVE_DIR='', SCAN_DIR='', idx=None, dir_list=''):
    if idx is not None:
        assert os.path.exists(dir_list), f'Directory list file {dir_list} does not exist'
        SAVE_DIR = os.path.join(SAVE_DIR, 'tmp/')
        with open(dir_list, 'r') as f:
            Dirs = f.readlines()
            Dir = Dirs[idx].strip()
        SCAN_DIR = Dir

    # Print the current configuration
    LOGGER.info('Starting scanDicom: Step 01')
    LOGGER.info(f'SCAN_DIR: {SCAN_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if PROFILE:
        LOGGER.info(f'Profiling is enabled')

    # Check if the Data_table.csv already exists
    if out_name in os.listdir(SAVE_DIR):
        LOGGER.error(f'{out_name} already exists. Skipping step 01')
        LOGGER.error(f'To re-run this step, delete the existing {out_name} file')
        exit()
    if idx is None:
        # Finding main directory and subdirectories
        LOGGER.info('Finding all directories containing DICOM files')
        dicom_dirs = find_all_dicom_dirs(SCAN_DIR, debug=DEBUG)
        if TEST:
            dicom_dirs = dicom_dirs[:N_TEST]
            LOGGER.info(f'Running in test mode with {N_TEST} directories')
    else:
        with open(dir_list, 'r') as f:
            Dirs = f.readlines()
        Dir = Dirs[idx].strip()
        dicom_dirs = find_all_dicom_dirs(Dir, debug=DEBUG)


    # Scan the directories for dicom files
    LOGGER.info('Analyzing DICOM directories')
    dicom_files = run_function(findDicom, dicom_dirs, Parallel=PARALLEL, debug=DEBUG)
    dicom_files = [f for sublist in dicom_files for f in sublist] # Flatten the list of lists
    LOGGER.info(f'Found {len(dicom_files)} dicom files in the input directory')
    # Extract the dicom information
    LOGGER.info('Extracting information from dicom files')
    INFO = run_function(extractDicom, dicom_files, Parallel=PARALLEL, debug=DEBUG)
    Data_table = pd.DataFrame(INFO) # Convert the extracted information to a pandas dataframe
    Data_table.to_csv(f'{SAVE_DIR}{out_name}', index=False) # Save the extracted information to a csv file
    LOGGER.info(f'DICOM information extraction completed and saved to {out_name}')

    if idx is not None:
        if idx == len(Dirs) - 1:
            LOGGER.info('Last script, compiling results')
            Tables = []
            while len(Tables) < len(Dirs):
                LOGGER.info('Waiting for all tables to be compiled')
                time.sleep(5)
                Tables = os.listdir(SAVE_DIR)
            LOGGER.info('Compiling results')
            Data_table = pd.DataFrame()
            for table in Tables:
                if table.endswith('.csv'):
                    LOGGER.info(f'Compiling {table}')
                    Data_table = pd.concat([Data_table, pd.read_csv(f'{SAVE_DIR}{table}')], ignore_index=True)
            SAVE_DIR = SAVE_DIR.replace('tmp/', '')
            Data_table.to_csv(f'{SAVE_DIR}Data_table.csv', index=False) # Save the extracted information to a csv file


if __name__ == '__main__':
    if PROFILE:
        LOGGER.info('Profiling enabled')
        yappi.start()
        LOGGER.info('Starting main function')

    if args.dir_idx is None:
        main(SCAN_DIR=SCAN_DIR, SAVE_DIR=SAVE_DIR)
    else:
        LOGGER.info(f'Processing single directory: {args.dir_idx}')
        main(out_name=f'Data_table_{args.dir_idx}.csv', idx=args.dir_idx, dir_list=args.dir_list, SCAN_DIR=SCAN_DIR, SAVE_DIR=SAVE_DIR)

    if PROFILE:
        LOGGER.info('Main function completed')
        yappi.stop()
        profile_output_path = 'step01_profile.yappi'
        LOGGER.info(f'Writing profile results to {profile_output_path}')
        yappi.get_func_stats().save(profile_output_path, type='pstat')
        LOGGER.info(f'Profile results saved to {profile_output_path}')
    exit()