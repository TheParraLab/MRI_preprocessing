# Package imports
import os
import glob
import threading
import pydicom as pyd
import numpy as np
import pandas as pd
import statistics as stat
# Function imports
from multiprocessing import Queue, Manager, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, List, Any
from functools import partial
# Custom imports
from toolbox import ProgressBar, get_logger
from DICOM import DICOMfilter, DICOMorder

# Global variables for progress bar
Progress = None
manager = Manager()
progress_queue = manager.Queue()

# Define necessary parameters
SAVE_DIR = '/FL_system/data/' # Location to saveupdated tables and load the constructed Data_table.csv ['/FL_system/data/']
COMPUTED_FLAGS = ['slope', 'sub', 'subtract', 'secondary']
DEBUG = 0
TEST = True
N_TEST = 100
PARALLEL = True
# Initialize logger
LOGGER = get_logger('02_parseDicom', f'{SAVE_DIR}/logs/')

# Profiler
PROFILE = True
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
    if Parallel:
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(target, item, *args, **kwargs) for item in items]
            results = [future.result() for future in futures]
    else:
        results = [target(item) for item in items]

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
    order = DICOMorder(Data_subset)
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

    return filter.dicom_table, filter.removed

# Function to split the data table based on the unique identifier
def split_table(ID):
    global Data_table
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
    results, removed = run_with_progress(filterDicom, Data_subsets, Parallel=PARALLEL)
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
    #Filter pre table to only include Iden_uniq sessions
    PRE_TABLE = PRE_TABLE[PRE_TABLE['SessionID'].isin(Iden_uniq)]
    PRE_TABLE.to_csv(f'{SAVE_DIR}Data_table_TESTING.csv', index=False)

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