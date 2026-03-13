# Package imports
import os
#import glob
#import threading
import pickle
import shutil
import argparse
import time
import subprocess
import re
import random

#import pydicom as pyd
import numpy as np
import pandas as pd
#import statistics as stat
# Function imports
from multiprocessing import Manager, cpu_count, Lock#, Queue
#from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, List, Any
from functools import partial
from collections import defaultdict
# Custom imports
from toolbox import get_logger, run_function
from DICOM import DICOMfilter, DICOMorder, DICOMsplit

# Global variables for progress bar
Progress = None
manager = Manager()
disk_space_lock = Lock()

def parse_args():
    """
    Parse command-line arguments for the DICOM parsing script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Parse DICOM data')
    parser.add_argument('--multi', '-m', nargs='?', const=cpu_count()-1, type=int, help='Run with multiprocessing enabled, using provided number of cpus (default: max-1)')
    parser.add_argument('--save_dir', type=str, default='/FL_system/data/', help='Directory to save the updated tables')
    parser.add_argument('--dir_idx', type=int, help='Index of the folder to process from dirs_to_process.txt')
    parser.add_argument('--dir_list', type=str, default='dirs_to_process.txt', help='Path to the directory list file')
    parser.add_argument('--load_table', type=str, default='/FL_system/data/Data_table.csv', help='Load table to use for the job')
    parser.add_argument('--filter_only', action='store_true', help='Run only the filtering step without ordering')
    parser.add_argument('--move', action='store_true', help='Move files to temporary locations')
    return parser.parse_args()


# Define necessary parameters
args = None
SAVE_DIR = ''
COMPUTED_FLAGS = ['slope', 'sub', 'subtract']#, 'secondary'] # Keywords to identify derived images, removed secondary  for now due to some primary images being marked as such.
DESCRIPTION_FLAGS= ['loc', 'pjn', 'calib']
PARALLEL = False
TEST = False
N_TEST = 25
N_CPUS = cpu_count() - 1
MOVE = False

# Initialize logger
LOGGER = None
DISK_SPACE_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5 GB
stop_flag = None

def configure_runtime(parsed_args):
    """
    Initialize global variables and logger for script execution.

    Args:
        parsed_args (argparse.Namespace): The parsed command-line arguments.
    """
    global args, SAVE_DIR, PARALLEL, N_CPUS, MOVE, LOGGER, stop_flag
    args = parsed_args
    SAVE_DIR = args.save_dir
    PARALLEL = args.multi is not None
    N_CPUS = args.multi if PARALLEL else cpu_count() - 1
    MOVE = args.move
    LOGGER = get_logger('02_parseDicom', f'{SAVE_DIR}/logs/')
    stop_flag = manager.Event()

# Profiler
PROFILE = False
if PROFILE:
    import yappi
    #import pstats
    #import io
    #yappi.set_clock_type('cpu')

#### Preprocessing | Step 2: Parse DICOM data ####
# This script uses the extracted dicom data to filter and order the identified scans
# 
# The script is meant to be run after the data has been extracted and saved to /data/Data_table.csv 
#
# The following filters are applied to the data:
# - T2 modality
# - Breast implants | Scans with breast implants are identified and removed based on the SeriesDescription and Type fields
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
    """
    Check if there is enough disk space available.

    Args:
        directory (str): The directory path to check for available space.

    Returns:
        bool: True if available space exceeds the threshold, False otherwise.
    """
    statvfs = os.statvfs(directory)
    available_space = statvfs.f_frsize * statvfs.f_bavail
    if available_space < DISK_SPACE_THRESHOLD*2:
        LOGGER.debug(f'Available space: {available_space}')
    return available_space > DISK_SPACE_THRESHOLD

def save_progress(data: list, filename: str) -> None:
    """
    Save the current relocation progress to a file.

    Args:
        data (list): The data structure (e.g. list of temporary relocations) to save.
        filename (str): The name of the file to save the data to.
    """
    LOGGER.info(f'Saving progress to {filename}')
    if os.path.exists(f'{SAVE_DIR}{filename}'):
        os.remove(f'{SAVE_DIR}{filename}')
    with open(f'{SAVE_DIR}{filename}', 'wb') as f:
        pickle.dump(data, f)

def load_progress(filename: str) -> Any:
    """
    Load saved relocation progress from a file.

    Args:
        filename (str): The name of the file to load progress from.

    Returns:
        Any: The loaded progress data, or None if the file does not exist.
    """
    if os.path.exists(f'{SAVE_DIR}{filename}'):
        LOGGER.info(f'Loading progress from {filename}')
        with open(f'{SAVE_DIR}{filename}', 'rb') as f:
            return pickle.load(f)
    return None
#############################
## Main functions
#############################
def save_to_csv(tup: tuple) -> None:
    """
    Save removed scans to a corresponding CSV file.

    Args:
        tup (tuple): A tuple containing a string key (removal category) and a
                     pandas DataFrame (the items removed).

    TODO: Enhance error handling to avoid potential issues when saving files
          concurrently if paths collide.
    """
    key, item = tup
    item.to_csv(f'{SAVE_DIR}removal_log/Removed_{key}.csv', index=False)

def orderDicom(Data_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Order the provided DICOM data subset based on scan timings.

    Trigger time is typically in ms post-injection.
    Acquisition time is typically in HHMMSS format.

    Args:
        Data_subset (pd.DataFrame): Subset of data specific to a single SessionID.

    Returns:
        pd.DataFrame: Ordered dataframe representing the primary sequence of scans.
    """
    Data_subset = Data_subset.reset_index(drop=True)
    SessionID = Data_subset['SessionID'].values[0]
    order = DICOMorder(Data_subset, logger=LOGGER)
    order.order('TriTime', secondary_param='AcqTime')
    if order.dicom_table.empty:
        LOGGER.error(f'No scans remaining after ordering for {SessionID}')
        return order.dicom_table
    else:
        order.findPre()
        return order.dicom_table

def splitDicom(Data_subset: pd.DataFrame) -> tuple:
    """
    Separate scans containing multiple post images within a single directory.

    Args:
        Data_subset (pd.DataFrame): Subset of data specific to a single SessionID.

    Returns:
        tuple: (Updated dataframe, List of files to be relocated)
    """
    Data_subset = Data_subset.reset_index(drop=True)
    splitter = DICOMsplit(Data_subset, logger=LOGGER)
    if splitter.SCAN:
        splitter.scan_all()
        splitter.sort_scans()
        return splitter.dicom_table, splitter.temporary_relocations
    else:
        return Data_subset, []

def filterDicom(Data_subset: pd.DataFrame) -> tuple:
    """
    Filter the provided DICOM data subset based on defined criteria to isolate
    the primary scan sequence.

    Args:
        Data_subset (pd.DataFrame): Subset of data specific to a single SessionID.

    Returns:
        tuple: (Filtered dataframe, Removed items dictionary, Temporary relocations list)

    TODO: Deep review of the DISCO and steady-state isolation path. If a sequence
          fails both approaches, it throws the scans into `Sequence_Failure` but might
          discard perfectly valid scans in edge cases where sequences mix modalities
          unusually. Should probably provide a more detailed secondary fallback.
    """
    Data_subset = Data_subset.reset_index(drop=True)
    dicom_filter = DICOMfilter(Data_subset, logger=LOGGER, tmp_save=SAVE_DIR.replace('tmp/', 'tmp_data/'))
    dicom_filter.Types(COMPUTED_FLAGS)
    dicom_filter.Description(DESCRIPTION_FLAGS)
    
    if len(dicom_filter.dicom_table) < 2:
        dicom_filter.logger.error(f'Not enough scans for {dicom_filter.Session_ID}, removing...')
        dicom_filter.removed['N_samples'] = dicom_filter.dicom_table
        dicom_filter.dicom_table = pd.DataFrame()
        return dicom_filter.dicom_table, dicom_filter.removed, dicom_filter.temporary_relocations

    #filter.removeImplants()
    #dicom_filter.removeSide()
    #dicom_filter.removeSlices() # Temporarily removed to allow both DISCO and steady state scans to be processed
    #dicom_filter.removeTimes(['TriTime']) # Omitted, Pre scans have unknown trigger time
    #dicom_filter.removeDWI()

    # Labelling DISCO scans
    disco_pattern = re.compile(r'disco', re.IGNORECASE)
    dicom_filter.dicom_table['IS_DISCO'] = dicom_filter.dicom_table['Series_desc'].str.contains(disco_pattern, na=False)
    
    if dicom_filter.dicom_table['IS_DISCO'].sum() > 0:
        # If DISCO files are found
        dicom_filter.logger.debug(f'DISCO scans detected | {dicom_filter.Session_ID}')
        dicom_filter.disco_table = dicom_filter.dicom_table.loc[dicom_filter.dicom_table['IS_DISCO'] == True]
        dicom_filter.dicom_table = dicom_filter.dicom_table.loc[dicom_filter.dicom_table['IS_DISCO'] == False]
        if len(dicom_filter.dicom_table) > 2:
            # Attempt to isolate the primary sequence of scans using steady state information
            dicom_filter.logger.debug(f'Will attempt to determine steady state sequence | {dicom_filter.Session_ID}')
            if not dicom_filter.isolate_sequence():
                # If unable to isolate the sequence using steady state information, attempt to use DISCO information to isolate the sequence
                dicom_filter.logger.debug(f'Failed to isolate steady state sequence | {dicom_filter.Session_ID}')
                dicom_filter.logger.debug(f'Attempting to solve with disco | {dicom_filter.Session_ID}')
                dicom_filter.dicom_table = dicom_filter.disco_table
                if not dicom_filter.isolate_sequence(): # If DISCO isolation fails, return an empty table
                    # If steady state and disco both fail
                    dicom_filter.logger.debug(f'Failed to isolate sequence using DISCO | {dicom_filter.Session_ID}')
                    dicom_filter.removed['Sequence_Failure'] = dicom_filter.dicom_table.copy()
                    dicom_filter.dicom_table = pd.DataFrame()
                else:
                    dicom_filter.logger.debug(f'Sequence isolated using DISCO | {dicom_filter.Session_ID}')
            else:
                dicom_filter.logger.debug(f'Sequence isolated using steady state information | {dicom_filter.Session_ID}')
        elif len(dicom_filter.disco_table) > 2:
            # If not enough steady state information to isolate the sequence, attempt to use DISCO information to isolate the sequence
            dicom_filter.logger.debug(f'Forced to utilize DISCO, not enough steady state information [{len(dicom_filter.dicom_table)}] | {dicom_filter.Session_ID}')
            dicom_filter.dicom_table = dicom_filter.disco_table
            if not dicom_filter.isolate_sequence(): # Attempt to isolate the primary sequence of scans using DISCO
                dicom_filter.logger.debug(f'Failed to isolate sequence using DISCO | {dicom_filter.Session_ID}')
                dicom_filter.removed['Sequence_Failure'] = dicom_filter.dicom_table.copy()
                dicom_filter.dicom_table = pd.DataFrame()
            else:
                dicom_filter.logger.debug(f'Sequence isolated using DISCO | {dicom_filter.Session_ID}')
        else:
            dicom_filter.logger.error(f'Not enough scans to identify sequence [DISCO or SS] | {dicom_filter.Session_ID}')
            dicom_filter.removed['Sequence_Failure'] = pd.concat([dicom_filter.dicom_table, dicom_filter.disco_table])
            dicom_filter.dicom_table = pd.DataFrame()
    else:
        dicom_filter.logger.debug(f'No DISCO scans detected | {dicom_filter.Session_ID}')
        if dicom_filter.isolate_sequence():
            dicom_filter.logger.debug(f'Sequence isolated using steady state information | {dicom_filter.Session_ID}')
        else:
            dicom_filter.logger.debug(f'Failed to isolate sequence using steady state information | {dicom_filter.Session_ID}')
            dicom_filter.removed['Sequence_Failure'] = dicom_filter.dicom_table.copy()
            dicom_filter.dicom_table = pd.DataFrame()
    if len(dicom_filter.dicom_table) == 0:
        LOGGER.error(f'No scans remaining after filtering for {Data_subset["SessionID"].values[0]}')
    
    return dicom_filter.dicom_table, dicom_filter.removed, dicom_filter.temporary_relocations

def split_table(ID: str) -> pd.DataFrame:
    """
    Filter the global Data_table for a specific SessionID.

    Args:
        ID (str): The unique SessionID to filter for.

    Returns:
        pd.DataFrame: A copy of the rows matching the ID.
    """
    global Data_table
    LOGGER.debug(f'Splitting table for ID: {ID}')
    return Data_table[Data_table['SessionID'] == ID].copy()

def agg_removed(removed_table: dict) -> None:
    """
    Aggregate removed scans across multiple processing runs.

    Args:
        removed_table (dict): Dictionary mapping removal categories to DataFrames.

    TODO: Using `pd.concat` in a loop can degrade performance on very large logs.
          Consider refactoring `Remove_Tables` to collect lists of DataFrames and
          concatenate them once at the end.
    """
    global Remove_Tables
    for key, value in removed_table.items():
        Remove_Tables[key] = pd.concat([Remove_Tables[key], value], ignore_index=True)

def init_data(load_table: str='', target: str=None) -> None:
    """
    Initialize data globally, reading the extracted CSV and formatting IDs.

    Args:
        load_table (str): Path to the input Data_table.csv.
        target (str, optional): An optional specific ID to filter on startup.
    """
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
    Remove_Tables = defaultdict(pd.DataFrame)  # Use defaultdict to initialize empty DataFrames for each key
    #Remove_Tables = {}
    #Remove_Tables['T2'] = pd.DataFrame()
    #Remove_Tables['Slices'] = pd.DataFrame()
    #Remove_Tables['Computed'] = pd.DataFrame()
    #Remove_Tables['No_pre'] = pd.DataFrame()
    #Remove_Tables['DISCO'] = pd.DataFrame()
    #Remove_Tables['No_post'] = pd.DataFrame()

def relocate(commands: list, relocations: list) -> None:
    """
    Relocate files to new paths based on provided commands.

    Args:
        commands (list): List of [source, destination] pairs.
        relocations (list): Global list of pending relocations, synchronized across processes.

    TODO: Thread-safety check: `shutil.copy` may hit race conditions if multiple processes
          attempt to create or interact with the exact same parent directories simultaneously
          despite `os.makedirs`. Consider robust directory locking or centralized moving.
    """
    LOGGER.debug(f'Relocate called with {len(commands)} commands')
    LOGGER.debug(f'Current relocations: {len(relocations)}')
    LOGGER.debug(f'First command: {commands[0] if commands else "None"}')
    if not commands:
        LOGGER.warning('No commands supplied to relocate')
        return
    destinations = [cmd[1] for cmd in commands]
    destinations = list(set(destinations))
    for dest in destinations:
        if not os.path.exists(dest):
            os.makedirs(dest)
        else:
            LOGGER.warning(f'{dest} already exists')
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
    try:
        for command in commands:
            LOGGER.debug(f'Linking {command[0]} to {command[1]}')
            src_path = os.path.abspath(command[0])
            dest_path = command[1]
            if os.path.exists(dest_path) or os.path.islink(dest_path):
                os.remove(dest_path)
            os.symlink(src_path, dest_path)
        with disk_space_lock:
            relocations.remove(commands)
    except Exception as e:
        LOGGER.error(f'Error in relocating files: {e}', exc_info=True)

def chunk_list(lst: list, chunk_size: int):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

#############################
## Main script
#############################
def main(out_name: str=f'Data_table_timing.csv', SAVE_DIR: str='', target: str=None) -> None:
    """
    Main orchestration function for parsing DICOM data.

    This function sequentially filters out bad scans, sorts out mixed directories,
    orders scans correctly by time, and writes out the resulting files.

    Args:
        out_name (str): Filename for the successfully ordered output CSV.
        SAVE_DIR (str): Location to save outputs, checkpoints, and logs.
        target (str, optional): A specific ID to process independently.

    TODO: Error Handling: While processing large groups, if `filterDicom` encounters
          catastrophic failure, it could crash the main script. Wrap processing steps in
          tighter try-except blocks to allow gracefully dropping broken sessions rather
          than halting the entire parallel pool.
    """
    global Data_table, Remove_Tables

    # Create the save directory if it does not exist
    if not os.path.exists(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR)
            LOGGER.info(f'Created directory: {SAVE_DIR}')
        except Exception as e:
            LOGGER.error(f'Error creating directory {SAVE_DIR}: {e}')
            exit()
            
    # Print the current configuration
    LOGGER.info('Starting parseDicom: Step 02')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'COMPUTED_FLAGS: {COMPUTED_FLAGS}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if PROFILE:
        LOGGER.info('Profiling enabled')

    # Check if the output already exists
    if out_name in os.listdir(SAVE_DIR):
        LOGGER.error(f'{out_name} already exists')
        if input('Would you like to reprocess? [Y/n]?\n').lower() != 'y':
            LOGGER.error('Stopping Processing')
            exit()

    progress = load_progress('parseDicom_progress.pkl')
    if progress:
        LOGGER.info(f'Progress file found. {len(progress)} items remaining')
        temporary_relocation = manager.list(progress)
    else:
        # Load in the data table
        init_data(args.load_table, target)
        
        # TEMP - REMOVE 16-328 protocol
        #Data_table = Data_table[Data_table['ID'].apply(lambda x: x.split('_')[1]) == '20-425']
        # Get the unique identifiers
        Iden_uniq = np.unique(Data_table['SessionID'])
        PRE_TABLE = Data_table.copy()
        if TEST:
            Iden_uniq = Iden_uniq[:N_TEST]
            LOGGER.info(f'Running in test mode with {N_TEST} sessions')
        if PARALLEL:
            LOGGER.debug('Running in parallel mode')
        # Split the data table into subsets based on the unique identifiers
        #Data_subsets = run_function(LOGGER, split_table, Iden_uniq, Parallel=PARALLEL, P_type='process')
        Data_subsets = [group.copy() for _, group in Data_table.groupby('SessionID')]
        random.shuffle(Data_subsets)
        # Filter the data based on the criteria defined in DICOMfilter and filterDicom
        results, removed, temporary_relocation = run_function(LOGGER, filterDicom, Data_subsets, Parallel=PARALLEL, P_type='process')
        #temporary_relocation = list(temporary_relocation)
        #temporary_relocation = manager.list([item for sublist in temporary_relocation for item in sublist])

        # Filtered results and removed scans are concatenated into a single table
        results = list(results)
        results = [df for df in results if not df.empty]
        removed = list(removed)
        Data_table = pd.concat(results)
        Data_table = Data_table.reset_index(drop=True)
        Data_table['SessionID'] = Data_table['ID'] + '_' + Data_table['DATE'].astype(str)
        Iden_uniq_after = Data_table['SessionID'].unique()
        Iden_uniq_after_clean = []
        for i in Iden_uniq_after:
            if i[-2:] in ('_a', '_b', '_l', '_r'):
                Iden_uniq_after_clean.append(i[:-2])
            else:
                Iden_uniq_after_clean.append(i)
        Iden_uniq_after_clean = list(set(Iden_uniq_after_clean)) # Get unique IDs without laterality suffix
        run_function(LOGGER, agg_removed, removed, Parallel=False)

        # Display the results of the filtering process
        LOGGER.info('Filtering Results:')
        LOGGER.info(f'Initial number of unique sessions: {len(Iden_uniq)}')
        LOGGER.info(f'Final number of unique sessions: {len(Iden_uniq_after_clean)}')
        LOGGER.info(f'Final number of sesions, including laterality suffix: {len(Iden_uniq_after)}')
        LOGGER.info(f'Number of removed sessions: {len(Iden_uniq) - len(Iden_uniq_after_clean)}')

        for key, value in Remove_Tables.items():
            LOGGER.info(f'===== {key} =====')
            Rem_ID = value['SessionID'].unique()
            Gone_ID = set(Rem_ID) - set(Iden_uniq_after_clean)
            LOGGER.info(f'  Number of unique sessions missing from final output: {len(Gone_ID)}')
            LOGGER.info(f'  Number of scans removed: {len(value)}')
        LOGGER.info(f'Saving filtered data to {SAVE_DIR}Data_table_filtered.csv')
        Data_table.to_csv(f'{SAVE_DIR}Data_table_filtered.csv', index=False)


        # Save a .csv for each item in the full_removed dictionary
        if not os.path.exists(f'{SAVE_DIR}removal_log'):
            os.mkdir(f'{SAVE_DIR}removal_log')
        run_function(LOGGER, save_to_csv, list(Remove_Tables.items()), Parallel=PARALLEL, P_type='process')
        fully_removed = pd.DataFrame()
        for ID in Iden_uniq:
            if ID not in Iden_uniq_after:
                LOGGER.debug(f'Session {ID} was completely removed')
                fully_removed = pd.concat([fully_removed, PRE_TABLE[PRE_TABLE['SessionID'] == ID]], ignore_index=True)
        if not fully_removed.empty:
            fully_removed.to_csv(f'{SAVE_DIR}removal_log/Removed_fully.csv', index=False)
            LOGGER.info(f'Saved fully removed sessions to {SAVE_DIR}removal_log/Removed_fully.csv')
        if args.filter_only:
            LOGGER.info('Filter only mode enabled. Exiting after filtering step.')
            return

        # Resplit the filtered data table into subsets based on the unique identifiers
        #Data_subsets = run_function(LOGGER, split_table, Iden_uniq_after, Parallel=PARALLEL, P_type='process')
        Data_subsets = [group.copy() for id, group in Data_table.groupby('SessionID') if id in Iden_uniq_after]

        # Seperating scans which contain multiple post images in a single directory
        results, redirections = run_function(LOGGER, splitDicom, Data_subsets, Parallel=PARALLEL, P_type='process')
        results = [df for df in results if not df.empty]
        Data_table = pd.concat(results)
        Data_table = Data_table.reset_index(drop=True)
        temporary_relocation = manager.list([item for sublist in redirections for item in sublist])
        Iden_uniq_after = Data_table['SessionID'].unique()
        LOGGER.info(f'Updated number of scans after splitting multi-post scans: {len(Data_table)}')
        LOGGER.info(f'Updated number of unique sessions after splitting multi-post scans: {len(Iden_uniq_after)}')
        LOGGER.info(f'Number of temporary relocations after splitting multi-post scans: {len(temporary_relocation)}')
        LOGGER.debug(f'Temporary relocations example [first 3 entries]: {temporary_relocation[0:3]}')
        # subgrouping temporary_relocation into 100n item chunks for processing
        temporary_relocation = list(chunk_list(list(temporary_relocation), 100))


        Data_table.to_csv(f'{SAVE_DIR}Data_table_split.csv', index=False)
        Data_subsets = run_function(LOGGER, split_table, Data_table['SessionID'].unique(), Parallel=PARALLEL, P_type='process')

        # Order the data based on the criteria defined in DICOMorder and orderDicom
        results = run_function(LOGGER, orderDicom, Data_subsets, Parallel=PARALLEL, P_type='process')
        Data_table = pd.concat(results)
        Data_table = Data_table.reset_index(drop=True)
        LOGGER.info('')
        LOGGER.info('Ordering complete')
        LOGGER.info(f'Final number of unique sessions: {len(Data_table["SessionID"].unique())}')
        LOGGER.info(f'Final number of scans: {len(Data_table)}')
        LOGGER.info(f'Saving ordered data to {SAVE_DIR}{out_name}')
        Data_table.to_csv(f'{SAVE_DIR}{out_name}', index=False)

    # Saving temporary relocation list to a file for review and running later
    with open(f'{SAVE_DIR}temporary_relocation.pkl', 'wb') as f:
        pickle.dump(list(temporary_relocation), f)
    print('Temporary relocation list saved to temporary_relocation.pkl')

    #save_progress(list(temporary_relocation), 'parseDicom_progress.pkl')
    #exit()

    if MOVE:
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
    configure_runtime(parse_args())
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