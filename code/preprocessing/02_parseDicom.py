# Package imports
import os
import glob
import threading
import pickle
import shutil
import argparse
import time
import subprocess

import pydicom as pyd
import numpy as np
import pandas as pd
import statistics as stat
# Function imports
from multiprocessing import Queue, Manager, cpu_count, Lock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, List, Any
from functools import partial
# Custom imports
from toolbox import get_logger, run_function
from DICOM import DICOMfilter, DICOMorder

# Global variables for progress bar
Progress = None
manager = Manager()
disk_space_lock = Lock()

# Set up parsing arguments
parser = argparse.ArgumentParser(description='Parse DICOM data')
parser.add_argument('--multi', '-m', nargs='?', const=cpu_count()-1, type=int, help='Run with multiprocessing enabled, using provided number of cpus (default: max-1)')
parser.add_argument('--save_dir', type=str, default='/FL_system/data/', help='Directory to save the updated tables')
parser.add_argument('--dir_idx', type=int, help='Index of the folder to process from dirs_to_process.txt')
parser.add_argument('--dir_list', type=str, default='dirs_to_process.txt', help='Path to the directory list file')
parser.add_argument('--load_table', type=str, default='', help='Load table to use for the job')
args = parser.parse_args()


# Define necessary parameters
SAVE_DIR = args.save_dir
COMPUTED_FLAGS = ['slope', 'sub', 'subtract', 'secondary']
PARALLEL = args.multi is not None
TEST = False
N_TEST = 25
N_CPUS = args.multi if PARALLEL else cpu_count() - 1

# Initialize logger
LOGGER = get_logger('02_parseDicom', f'{SAVE_DIR}/logs/')
DISK_SPACE_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5 GB
stop_flag = manager.Event()

# Profiler
PROFILE = False
if PROFILE:
    import yappi
    import pstats
    import io
    #yappi.set_clock_type('cpu')

#### Preprocessing | Step 2: Parse DICOM data ####
# This script uses the extracted dicom data to filter and order the identified scans
# 
# The script is meant to be run after the data has been extracted and saved to /data/Data_table.csv 
#
# The following filters are applied to the data:
# - T2 modality
# - Breast implants | TODO: Implement alternative filtering not based on breast size
# - Laterality | Majority side is determined and scans not on the majority side are removed
# - Number of slices | Majority number of slices is determined and scans not having the majority number of slices are removed
# - Derived images | SeriesDescription and Type are checked for keywords indicating derived images
#
# The data is then ordered based on the following criteria:
# - Trigger time | The time since the start of the scan is used to order the scans
#
# The goal of this script is to isolate the primary sequence of scans and remove any derived images or other unwanted scans
#
# The filtered and ordered data is saved to /data/Data_table_timing.csv

#########################################
## Parallelization and Progress functions
#########################################
# Wrapper for progress updates
def check_disk_space(directory: str) -> bool:
    """Check if there is enough disk space available."""
    statvfs = os.statvfs(directory)
    available_space = statvfs.f_frsize * statvfs.f_bavail
    if available_space < DISK_SPACE_THRESHOLD*2:
        LOGGER.debug(f'Available space: {available_space}')
    return available_space > DISK_SPACE_THRESHOLD

def save_progress(data, filename):
    """Save progress to a file."""
    LOGGER.info(f'Saving progress to {filename}')
    if os.path.exists(f'{SAVE_DIR}{filename}'):
        os.remove(f'{SAVE_DIR}{filename}')
    with open(f'{SAVE_DIR}{filename}', 'wb') as f:
        pickle.dump(data, f)

def load_progress(filename):
    """Load progress from a file."""
    if os.path.exists(f'{SAVE_DIR}{filename}'):
        LOGGER.info(f'Loading progress from {filename}')
        with open(f'{SAVE_DIR}{filename}', 'rb') as f:
            return pickle.load(f)
    return None
#############################
## Main functions
#############################
# Function to save removed scans to a .csv file
def save_to_csv(tup):
    key, item = tup
    item.to_csv(f'{SAVE_DIR}removal_log/Removed_{key}.csv', index=False)

# Function to order the DICOM data
def orderDicom(Data_subset):
    Data_subset = Data_subset.reset_index(drop=True)
    order = DICOMorder(Data_subset, logger=LOGGER)
    order.order('TriTime')
    order.findPre()
    return order.dicom_table

# Function to filter the DICOM data
def filterDicom(Data_subset):
    # Filter each subset of data based on the criteria
    Data_subset = Data_subset.reset_index(drop=True)
    filter = DICOMfilter(Data_subset, logger=LOGGER, tmp_save=SAVE_DIR.replace('tmp/', 'tmp_data/'))
    filter.removeT2()
    if 'Type' in Data_subset.columns:
        filter.removeComputed(COMPUTED_FLAGS)
    else:
        LOGGER.error("'Type' column not found in Data_subset")#filter.removeImplants()
    #filter.removeSide()
    filter.removeSlices()
    # Check if 'Type' column exists before removing computed flags

    #filter.removeTimes(['TriTime']) # Omitted, Pre scans have unknown trigger time
    #filter.removeDWI()

    return filter.dicom_table, filter.removed, filter.temporary_relocations

# Function to split the data table based on the unique identifier
def split_table(ID):
    global Data_table
    LOGGER.debug(f'Splitting table for ID: {ID}')
    return Data_table[Data_table['SessionID'] == ID].copy()

# Function to aggregate removed scans
def agg_removed(removed_table: dict):
    global Remove_Tables
    for key, value in removed_table.items():
        assert key in Remove_Tables, f'Key {key} not found in Remove_Tables'
        Remove_Tables[key] = pd.concat([Remove_Tables[key], value], ignore_index=True)

# Function to initialize the data
def init_data(load_table: str='', target: int=None):
    global Data_table
    Data_table = pd.read_csv(f'{load_table}', low_memory=False)
    if target is not None:
        try:
            Data_table = Data_table[Data_table['ID'] == target]
            LOGGER.info(f'Filtering data for target ID: {target}')
        except Exception as e:
            LOGGER.error(f'Error filtering data for target ID {target}: {e}')
            raise
    # Create a unique identifier for each session/exam
    Data_table['SessionID'] = Data_table['ID'] + '_' + Data_table['DATE'].astype(str)
    global Remove_Tables
    Remove_Tables = {}
    Remove_Tables['T2'] = pd.DataFrame()
    Remove_Tables['Slices'] = pd.DataFrame()
    Remove_Tables['Computed'] = pd.DataFrame()

def relocate(commands, relocations):
    if not commands:
        LOGGER.warning('No commands supplied to relocate')
        return
    with disk_space_lock:
        try:
            LOGGER.debug(commands[0][1])
            LOGGER.debug('/'.join(commands[0][1].split('/')[0:-2]))
        except:
            LOGGER.warning(commands)
        if not check_disk_space('/'.join(commands[0][1].split('/')[0:-2])):
            if not stop_flag.is_set():
                LOGGER.warning('Disk space is running low.  Pausing...')
                stop_flag.set()
                LOGGER.warning('Stop flag set')
            return
    destinations = [cmd[1] for cmd in commands]
    destinations = list(set(destinations))
    for dest in destinations:
        if not os.path.exists(dest):
            os.makedirs(dest)
        else:
            LOGGER.warning(f'{dest} already exists')
    try:
        for command in commands:
            shutil.copy(command[0], command[1])
        with disk_space_lock:
            relocations.remove(commands)
    except Exception as e:
        LOGGER.error(f'Error in relocating files: {e}', exc_info=True)
#############################
## Main script
#############################
def main(out_name: str=f'Data_table_timing.csv', SAVE_DIR: str='', target: str=None):
    global Data_table, Remove_Tables

    # Create the save directory if it does not exist
    if not os.path.exists(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR)
            LOGGER.info(f'Created directory: {SAVE_DIR}')
        except Exception as e:
            LOGGER.error(f'Error creating directory {SAVE_DIR}: {e}')
    
    # Print the current configuration
    LOGGER.info('Starting parseDicom: Step 02')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'COMPUTED_FLAGS: {COMPUTED_FLAGS}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if PROFILE:
        LOGGER.info('Profiling enabled')

    # Check if the output already exists
    if out_name in os.listdir(SAVE_DIR):
        LOGGER.error(f'{out_name} already exists. Skipping step 02')
        LOGGER.error(f'To re-run this step, delete the existing {out_name} file')
        exit()

    progress = load_progress('parseDicom_progress.pkl')
    if progress:
        LOGGER.info(f'Progress file found. {len(progress)} items remaining')
        temporary_relocation = manager.list(progress)
    else:
        # Load in the data table
        init_data(args.load_table, target)

        # Get the unique identifiers
        Iden_uniq = np.unique(Data_table['SessionID'])
        PRE_TABLE = Data_table.copy()
        if TEST:
            Iden_uniq = Iden_uniq[:N_TEST]
            LOGGER.info(f'Running in test mode with {N_TEST} sessions')
        if PARALLEL:
            LOGGER.debug('Running in parallel mode')

        # Split the data table into subsets based on the unique identifiers
        Data_subsets = run_function(LOGGER, split_table, Iden_uniq, Parallel=PARALLEL, P_type='process')
        #Data_subsets = [group for _, group in Data_table.groupby('SessionID')]
        # Filter the data based on the criteria defined in DICOMfilter and filterDicom
        results, removed, temporary_relocation = run_function(LOGGER, filterDicom, Data_subsets, Parallel=PARALLEL, P_type='process')
        temporary_relocation = list(temporary_relocation)
        temporary_relocation = manager.list([item for sublist in temporary_relocation for item in sublist])

        # Filtered results and removed scans are concatenated into a single table
        results = list(results)
        removed = list(removed)
        Data_table = pd.concat(results)
        Data_table = Data_table.reset_index(drop=True)
        Iden_uniq_after = Data_table['SessionID'].unique()
        run_function(LOGGER, agg_removed, removed, Parallel=False)
        # Display the results of the filtering process
        LOGGER.info('Filtering Results:')
        LOGGER.info(f'Initial number of unique sessions: {len(Iden_uniq)}')
        LOGGER.info(f'Final number of unique sessions: {len(Iden_uniq_after)}')
        LOGGER.info(f'Number of removed sessions: {len(Iden_uniq) - len(Iden_uniq_after)}')
        for key, value in Remove_Tables.items():
            LOGGER.info(f'Number of {key} scans removed: {len(value)}')

        # Order the data based on the criteria defined in DICOMorder and orderDicom
        results = run_function(LOGGER, orderDicom, results, Parallel=PARALLEL, P_type='process')
        Data_table = pd.concat(results)
        Data_table = Data_table.reset_index(drop=True)

        fully_removed = pd.DataFrame()
        for ID in Iden_uniq:
            if ID not in Iden_uniq_after:
                LOGGER.debug(f'Session {ID} was completely removed')
                fully_removed = pd.concat([fully_removed, PRE_TABLE[PRE_TABLE['SessionID'] == ID]], ignore_index=True)
        #fully_removed.to_csv(f'{SAVE_DIR}fully_removed.csv', index=False)
        LOGGER.info('Saving sequence information to updated file')

        # Save a .csv for each item in the full_removed dictionary
        if not os.path.exists(f'{SAVE_DIR}removal_log'):
            os.mkdir(f'{SAVE_DIR}removal_log')
        run_function(LOGGER, save_to_csv, list(Remove_Tables.items()), Parallel=PARALLEL, P_type='process')

        Data_table.to_csv(f'{SAVE_DIR}{out_name}', index=False)
        LOGGER.info(f'Timing information saved to {out_name}.csv')
    LOGGER.info(f'Number of temporary relocations: {len(temporary_relocation)}')


    #save_progress(list(temporary_relocation), 'parseDicom_progress.pkl')
    #exit()


    run_function(LOGGER, partial(relocate, relocations=list(temporary_relocation)), list(temporary_relocation), Parallel=PARALLEL, P_type='process')

    if not stop_flag.is_set():
        LOGGER.info('redirection complete without stop flag')
        LOGGER.info('Removing progress file')
        if os.path.exists('parseDicom_progress.pkl'):
            os.remove('parseDicom_progress.pkl')
    else:
        LOGGER.info('Nifti conversion complete with stop flag')
        if os.path.exists('parseDicom_progress.pkl'):
            os.remove('parseDicom_progress.pkl')
        save_progress(list(temporary_relocation), 'parseDicom_progress.pkl')
        LOGGER.info('checkpoint file saved')

if __name__ == '__main__':
    # Start the profiler if enabled
    if PROFILE:
        LOGGER.info('Profiling enabled')
        yappi.start()
        LOGGER.info('Starting main function')
    
    # Create the save directory when necessary
    if not os.path.exists(SAVE_DIR):
        # Use try-except to handle directory creation, in case parallel processes try to create the same directory
        try:
            os.makedirs(SAVE_DIR)
            LOGGER.info(f'Created directory: {SAVE_DIR}')
        except Exception as e:
            LOGGER.error(f'Error creating directory: {e}')
    
    # If not running on an HPC
    if args.dir_idx is None:
        main(SAVE_DIR=SAVE_DIR)
    # If running on an HPC
    else:
        PARALLEL = False
        assert os.path.exists(args.dir_list), f'Directory list file {args.dir_list} does not exist'
        # Save to a temporary directory
        SAVE_DIR = os.path.join(SAVE_DIR, 'tmp/')
        with open(args.dir_list, 'rb') as f:
            items = pickle.load(f)
        target = items[args.dir_idx].strip()
        LOGGER.info(f'Processing single directory: {args.dir_idx}')
        main(out_name=f'Data_table_timing_{args.dir_idx}.csv', SAVE_DIR=SAVE_DIR, target=target)

        if args.dir_idx == len(items) - 1:
            LOGGER.info('Last script, compiling results')
            Tables = []
            while len(Tables) < len(items):
                LOGGER.info('Waiting for all tables to be compiled')
                time.sleep(5)
                Tables = os.listdir(SAVE_DIR)
                Tables = [table for table in Tables if table.endswith('.csv')]
            LOGGER.info('All tables present, compiling...')
            Data_table = pd.DataFrame()
            for table in Tables:
                LOGGER.info(f'Compiling {table}')
                try:
                    tmp_table = pd.read_csv(os.path.join(SAVE_DIR, table))
                    Data_table = pd.concat([Data_table, tmp_table], ignore_index=True)
                except pd.errors.EmptyDataError:
                    LOGGER.error(f'{table} appears to be empty, skipping...')
                    continue
                except Exception as e:
                    LOGGER.error(f'Error compiling {table}: {e}')
                    break
            SAVE_DIR = SAVE_DIR.replace('tmp/', '')
            Data_table.to_csv(f'{SAVE_DIR}Data_table_timing.csv', index=False)
            LOGGER.info(f'Compiled results saved to {SAVE_DIR}Data_table_timing.csv')
            try:
                subprocess.run(['rm', '-r', f'{SAVE_DIR}tmp/'], check=True)
                LOGGER.info(f'Deleted temporary directory {SAVE_DIR}tmp/')
            except Exception as e:
                LOGGER.error(f'Error deleting temporary directory {SAVE_DIR}tmp/: {e}')
    
    # Finalize the profiler if enabled
    if PROFILE:
        LOGGER.info('Main function completed')
        yappi.stop()
        profile_output_path = 'step02_profile.yappi'
        LOGGER.info(f'Writing profile results to {profile_output_path}')
        yappi.get_func_stats().save(profile_output_path, type='pstat')
        LOGGER.info(f'Profile results saved to {profile_output_path}')
    exit()