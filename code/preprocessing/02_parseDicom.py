# Package imports
import os
import glob
import threading
import pickle
import shutil
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
from toolbox import ProgressBar, get_logger
from DICOM import DICOMfilter, DICOMorder

# Global variables for progress bar
Progress = None
manager = Manager()
disk_space_lock = Lock()
progress_queue = manager.Queue()

# Define necessary parameters
SAVE_DIR = '/FL_system/data/' # Location to saveupdated tables and load the constructed Data_table.csv ['/FL_system/data/']
COMPUTED_FLAGS = ['slope', 'sub', 'subtract', 'secondary']
DEBUG = 0
TEST = False
PROGRESS = False
N_TEST = 25
PARALLEL = True
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
def progress_wrapper(item, target, progress_queue, *args, **kwargs):
    result = target(item, *args, **kwargs)
    progress_queue.put((None, f'Processing'))
    return result

def check_disk_space(directory: str) -> bool:
    """Check if there is enough disk space available."""
    statvfs = os.statvfs(directory)
    available_space = statvfs.f_frsize * statvfs.f_bavail
    if available_space < DISK_SPACE_THRESHOLD*2:
        LOGGER.debug(f'Available space: {available_space}')
    return available_space > DISK_SPACE_THRESHOLD

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

    if PROGRESS:
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
                if stop_flag.is_set():
                    LOGGER.info('stop flag is set, exiting')
                    break
                try:
                    #result = future.result(timeout=600)
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    LOGGER.error(f'Error in parallel processing: {e}', exc_info = True)
    else:
        for item in items:
            if stop_flag.is_set():
                LOGGER.info('stop flag is set, exiting')
                break
            try:
                result = target(item)
                results.append(result)
            except Exception as e:
                LOGGER.error(f'Error in sequential processing: {e}', exec_info=True)

    if PROGRESS:
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

def progress_updater(queue, progress_bar):
    while True:
        item = queue.get()
        if item is None:
            break
        index, status = item
        progress_bar.update(index, status)

        queue.task_done()

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
    filter = DICOMfilter(Data_subset, debug=DEBUG, logger=LOGGER)
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
def init_data():
    global Data_table
    Data_table = pd.read_csv(f'{SAVE_DIR}Data_table.csv', low_memory=False)
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
            print(commands[0][1])
            print('/'.join(commands[0][1].split('/')[0:-2]))
        except:
            print(commands)
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
def main():
    global Data_table, Remove_Tables
    LOGGER.info('Starting parseDicom: Step 02')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'COMPUTED_FLAGS: {COMPUTED_FLAGS}')
    LOGGER.info(f'DEBUG: {DEBUG}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if TEST:
        LOGGER.info(f'TEST: {TEST}')
        LOGGER.info(f'N_TEST: {N_TEST}')
    if PROFILE:
        LOGGER.info('Profiling enabled')

    # Check if the Data_table_timing.csv already exists
    if 'Data_table_timing.csv' in os.listdir(SAVE_DIR):
        LOGGER.error('Data_table_timing.csv already exists')
        LOGGER.error(f'To reprocess data, please remove Data_table_timing.csv from {SAVE_DIR}')
        exit()

    progress = load_progress('saveNifti_progress.pkl')
    if progress:
        LOGGER.info(f'Progress file found. {len(progress)} items remaining')
        temporary_relocation = manager.list(progress)
    else:
        # Load in the data table
        init_data()

        # Get the unique identifiers
        Iden_uniq = np.unique(Data_table['SessionID'])
        PRE_TABLE = Data_table.copy()
        if TEST:
            Iden_uniq = Iden_uniq[:N_TEST]
            LOGGER.info(f'Running in test mode with {N_TEST} sessions')
        if PARALLEL:
            LOGGER.debug('Running in parallel mode')

        # Split the data table into subsets based on the unique identifiers
        Data_subsets = run_with_progress(split_table, Iden_uniq, Parallel=PARALLEL)
        # Filter the data based on the criteria defined in DICOMfilter and filterDicom
        results, removed, temporary_relocation = run_with_progress(filterDicom, Data_subsets, Parallel=PARALLEL)
        temporary_relocation = list(temporary_relocation)
        temporary_relocation = manager.list([item for sublist in temporary_relocation for item in sublist])
        print(temporary_relocation)
        print(len(temporary_relocation))
        #print(temporary_relocation.keys())
        # Filtered results and removed scans are concatenated into a single table
        results = list(results)
        removed = list(removed)
        Data_table = pd.concat(results)
        Data_table = Data_table.reset_index(drop=True)
        Iden_uniq_after = Data_table['SessionID'].unique()
        run_with_progress(agg_removed, removed, Parallel=False)
        # Display the results of the filtering process
        LOGGER.info('Filtering Results:')
        LOGGER.info(f'Initial number of unique sessions: {len(Iden_uniq)}')
        LOGGER.info(f'Final number of unique sessions: {len(Iden_uniq_after)}')
        LOGGER.info(f'Number of removed sessions: {len(Iden_uniq) - len(Iden_uniq_after)}')
        for key, value in Remove_Tables.items():
            LOGGER.info(f'Number of {key} scans removed: {len(value)}')

        # Order the data based on the criteria defined in DICOMorder and orderDicom
        results = run_with_progress(orderDicom, results, Parallel=PARALLEL)
        Data_table = pd.concat(results)
        Data_table = Data_table.reset_index(drop=True)

        fully_removed = pd.DataFrame()
        for ID in Iden_uniq:
            if ID not in Iden_uniq_after:
                LOGGER.debug(f'Session {ID} was completely removed')
                fully_removed = pd.concat([fully_removed, PRE_TABLE[PRE_TABLE['SessionID'] == ID]], ignore_index=True)
        fully_removed.to_csv(f'{SAVE_DIR}fully_removed.csv', index=False)
        LOGGER.info('Saving sequence information to updated file')

        # Save a .csv for each item in the full_removed dictionary
        if not os.path.exists(f'{SAVE_DIR}removal_log'):
            os.mkdir(f'{SAVE_DIR}removal_log')
        run_with_progress(save_to_csv, list(Remove_Tables.items()), Parallel=PARALLEL)

        Data_table.to_csv(f'{SAVE_DIR}Data_table_timing.csv', index=False)
        LOGGER.info('Timing information saved to Data_table_timing.csv')
    LOGGER.info(f'Number of temporary relocations: {len(temporary_relocation)}')
    run_with_progress(partial(relocate, relocations=temporary_relocation), temporary_relocation, Parallel=PARALLEL)

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
    if PROFILE:
        LOGGER.info('Profiling enabled')
        yappi.start()
        LOGGER.info('Starting main function')
        main()
        LOGGER.info('Main function completed')
        yappi.stop()
        profile_output_path = 'step02_profile.yappi'
        LOGGER.info(f'Writing profile results to {profile_output_path}')
        yappi.get_func_stats().save(profile_output_path, type='pstat')
        LOGGER.info(f'Profile results saved to {profile_output_path}')
    else:
        main()
    exit()