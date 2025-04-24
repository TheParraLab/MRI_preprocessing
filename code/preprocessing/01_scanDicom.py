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
from toolbox import get_logger, run_function
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

#### Preprocessing | Step 1: Extract DICOM data ####
# This script scans the input directory for dicom files and extracts necessary header information
#
# The extracted information is saved to /FL_system/data/Data_table.csv
#
# This step will begin by performing a recursive search for all dicom files in the input directory

# The extracted information will be saved to /data/Data_table.csv
# NOTE: This script is expected to run inside of the provided control_system docker container and called through the available web interface

#############################
## Main functions
#############################
def extractDicom(f):
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

def find_all_dicom_dirs(directory):
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

def findDicom(directory):
    # Find all dicom files in the directory through a recursive search
    # Will only record the first file found for each directory if files have the same series number    
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
def main(out_name='Data_table.csv', SAVE_DIR='', SCAN_DIR=''):
    # Validate input directories
    assert os.path.exists(SCAN_DIR), f'SAVE_DIR {SCAN_DIR} does not exist. Please provide a valid directory.'

    # Create the save directory if it does not exist
    if not os.path.exists(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR)
            LOGGER.info(f'Created directory {SAVE_DIR}')
        except Exception as e:
            LOGGER.error(f'Error creating directory {SAVE_DIR}: {e}')

    # Print the current configuration
    LOGGER.info('Starting scanDicom: Step 01')
    LOGGER.info(f'SCAN_DIR: {SCAN_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if PROFILE:
        LOGGER.info(f'Profiling is enabled')

    # Check if the output already exists
    if out_name in os.listdir(SAVE_DIR):
        LOGGER.error(f'{out_name} already exists. Skipping step 01')
        LOGGER.error(f'To re-run this step, delete the existing {out_name} file')
        exit()

    # Finding main directory and subdirectories
    LOGGER.info('Finding all directories containing DICOM files')
    dicom_dirs = find_all_dicom_dirs(SCAN_DIR)
    if TEST:
        dicom_dirs = dicom_dirs[:N_TEST]
        LOGGER.info(f'Running in test mode with {N_TEST} directories')

    # Scan the directories for dicom files
    LOGGER.info('Analyzing DICOM directories')
    dicom_files = run_function(LOGGER, findDicom, dicom_dirs, Parallel=PARALLEL, P_type='thread')
    dicom_files = [f for sublist in dicom_files for f in sublist] # Flatten the list of lists
    LOGGER.info(f'Found {len(dicom_files)} dicom files in the input directory')
    # Extract the dicom information
    LOGGER.info('Extracting information from dicom files')
    INFO = run_function(LOGGER, extractDicom, dicom_files, Parallel=PARALLEL, P_type='thread')
    Data_table = pd.DataFrame(INFO) # Convert the extracted information to a pandas dataframe
    Data_table.to_csv(f'{SAVE_DIR}{out_name}', index=False) # Save the extracted information to a csv file
    LOGGER.info(f'DICOM information extraction completed and saved to {out_name}')


if __name__ == '__main__':
    # Start the profiler if enabled
    if PROFILE:
        LOGGER.info('Profiling enabled')
        yappi.start()
        LOGGER.info('Starting main function')

    # Create the scan directory when necessary
    if not os.path.exists(SAVE_DIR):
        # Use try-except to handle directory creation, in case parallel processes try to create the same directory
        try:
            os.makedirs(SAVE_DIR)
            LOGGER.info(f'Created directory {SAVE_DIR}')
        except Exception as e:
            LOGGER.error(f'Error creating directory {SAVE_DIR}: {e}')
    
    # If not running on an HPC
    if args.dir_idx is None:
        main(SCAN_DIR=SCAN_DIR, SAVE_DIR=SAVE_DIR)
    # If running on an HPC
    else:
        assert os.path.exists(args.dir_list), f'Directory list file {args.dir_list} does not exist'
        # Save to temporary directory
        SAVE_DIR = os.path.join(SAVE_DIR, 'tmp/')
        with open(args.dir_list, 'r') as f:
            Dirs = f.readlines()
            Dir = Dirs[args.dir_idx].strip()
        SCAN_DIR = Dir # Set the scan directory to the one specified by the index
        LOGGER.info(f'Processing single directory: {args.dir_idx}')
        main(out_name=f'Data_table_{args.dir_idx}.csv', idx=args.dir_idx, SCAN_DIR=SCAN_DIR, SAVE_DIR=SAVE_DIR)

        if args.dir_idx == len(Dirs) - 1:
           LOGGER.info('Last script, compiling results')
            Tables = []
            while len(Tables) < len(Dirs):
                LOGGER.info('Waiting for all tables to be compiled')
                time.sleep(5)
                Tables = os.listdir(SAVE_DIR)
                Tables = [table for table in Tables if table.endswith('.csv')]
            LOGGER.info('All tables present, compiling...')
            Data_table = pd.DataFrame()
            for table in Tables:
                LOGGER.info(f'Compiling {table}')
                Data_table = pd.concat([Data_table, pd.read_csv(f'{SAVE_DIR}{table}')], ignore_index=True)
            SAVE_DIR = SAVE_DIR.replace('tmp/', '')
            Data_table.to_csv(f'{SAVE_DIR}Data_table.csv', index=False)
            LOGGER.info(f'Compiled results saved to {SAVE_DIR}Data_table.csv')
            subprocess.run(['rm', '-r', f'{SAVE_DIR}tmp/'], check=True)
            LOGGER.info(f'Deleted temporary directory {SAVE_DIR}tmp/')

    # Finalize the profiler if enabled
    if PROFILE:
        LOGGER.info('Main function completed')
        yappi.stop()
        profile_output_path = 'step01_profile.yappi'
        LOGGER.info(f'Writing profile results to {profile_output_path}')
        yappi.get_func_stats().save(profile_output_path, type='pstat')
        LOGGER.info(f'Profile results saved to {profile_output_path}')
    exit()