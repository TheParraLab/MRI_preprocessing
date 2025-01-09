import os
#import pydicom as pyd
import glob
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Queue, Manager, cpu_count, Lock
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
from typing import Callable, List, Any
from functools import partial
# Custom imports
from toolbox import ProgressBar, get_logger
from DICOM import DICOMfilter, DICOMorder

# Global variables for progress bar and lock
Progress = None
manager = Manager()
disk_space_lock = Lock()
progress_queue = manager.Queue()
LOGGER = get_logger('03_saveNifti', '/FL_system/data/logs/')

# Define necessary directories
LOAD_DIR = '/FL_system/data/' # Location to load the constructed Data_table_timing.csv ['/FL_system/data/']
SAVE_DIR = '/FL_system/data/nifti/' # Location to save the nifti files ['/FL_system/data/nifti/']
DEBUG = 0
TEST = False
PROGRESS = False
N_TEST = 200
PARALLEL = True
DISK_SPACE_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5 GB
stop_flag = manager.Event()

#### Preprocessing | Step 3: Save Nifti Files ####
# This script is for generating the nifti files for the selected scans
#
# This script utilizes the dcm2niix tool to convert the dicom files to nifti files
# It requires the Data_table_timing.csv file to be present in the /data/ directory, this is produced in the previous step

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
    if os.path.exists(f'{LOAD_DIR}{filename}'):
        os.remove(f'{LOAD_DIR}{filename}')
    with open(f'{LOAD_DIR}{filename}', 'wb') as f:
        pickle.dump(data, f)

def load_progress(filename):
    """Load progress from a file."""
    if os.path.exists(f'{LOAD_DIR}{filename}'):
        LOGGER.info(f'Loading progress from {filename}')
        with open(f'{LOAD_DIR}{filename}', 'rb') as f:
            return pickle.load(f)
    return None

def progress_wrapper(item, target, progress_queue, *args, **kwargs):
    if stop_flag.is_set():
        return
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
    if PROGRESS:
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
                    result = future.result(timeout=600)
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

    # Close the progress bar
    if PROGRESS:
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
    while not stop_flag.is_set():
        try:
            item = queue.get(timeout=1)
            if item is None:
                break
            index, status = item
            progress_bar.update(index, status)
            queue.task_done()
        except:
            continue

def run_cmd(command, commands):
    SessionID = command[2].split(os.sep)[-1]
    output_dir = command[2]
    file_name = command[4]
    if os.file.exists(f'{output_dir}{os.sep}{file_name}.nii'):
        LOGGER.debug(f'Nifti file already exists for {file_name}')
        return

    if stop_flag.is_set():
        LOGGER.info('stop flag is set, exiting')
        return
    with disk_space_lock:
        if not check_disk_space(SAVE_DIR):
            if not stop_flag.is_set():
                LOGGER.warning('Disk space is running low.  Pausing...')
                stop_flag.set()
                LOGGER.warning('Stop flag set')
            return
    if not os.path.isdir(f'{SAVE_DIR}{SessionID}'):
        try:
            os.mkdir(f'{SAVE_DIR}{SessionID}')
            if DEBUG > 0:
                LOGGER.debug(f'Created directory for {SessionID}')
        except FileExistsError:
            LOGGER.error(f'Directory for {SessionID} already exists')
    try:
        if DEBUG == 0:
            result = subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())
        with disk_space_lock:
            commands.remove(command)
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else 'No error message available'
        LOGGER.error(f'Error converting {command[-1]}: {error_message}')
    progress_queue.put((None, f'Converting'))
    
def makeNifti(Data_subset):
    # Convert all dicom files to nifti files
    Data_subset = Data_subset.reset_index(drop=True)
    SessionID = np.unique(Data_subset['SessionID'])[0]
    
    #if not os.path.isdir(f'{SAVE_DIR}{SessionID}'):
    #    os.mkdir(f'{SAVE_DIR}{SessionID}')
    #    if DEBUG > 0:
    #        LOGGER.debug(f'Created directory for {SessionID}')
    #else:
        #LOGGER.debug(f'Found existing directory for {SessionID}')

    Descriptor = [f'{int(M):02}' for M in Data_subset['Major']]
    LoadPATH = Data_subset['PATH']

    commands = []
    for i in range(len(Data_subset)):
        commands.append(['dcm2niix', '-o', f'{SAVE_DIR}{SessionID}', '-f', Descriptor[i], LoadPATH[i]])
    return commands

def split_table(ID):
    global progress_queue
    progress_queue.put((None, f'Splitting {ID}'))
    return Data_table[Data_table['SessionID'] == ID]

if __name__ == '__main__':
    LOGGER.info('Starting saveNifti: Step 03')
    LOGGER.info(f'LOAD_DIR: {LOAD_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if TEST:
        LOGGER.info(f'Running in test mode: {TEST}')
        LOGGER.info(f'Number of test sessions: {N_TEST}')

    #if os.path.exists(SAVE_DIR):
    #    if len(os.listdir(SAVE_DIR)) > 0:
    #        LOGGER.error('Nifti directory already exists')
    #        LOGGER.error('To reprocess data, please remove nifti directory from /FL_system/data/ or remove its contents')
    #        exit()
    #    else:
    #        LOGGER.warning('Nifti directory already exists, but is empty')
    #else:
    #    os.mkdir(SAVE_DIR)
        # Load progress if available
    progress = load_progress('saveNifti_progress.pkl')
    if progress:
        LOGGER.info(f'Progress file found. {len(progress)} items remaining')
        commands = manager.list(progress)
    else:
        LOGGER.info('No progress file found. Starting from scratch')
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)

        # Load the timing information
        Data_table = pd.read_csv(f'{LOAD_DIR}Data_table_timing.csv')
        SessionIDs = Data_table['SessionID']
        Iden_uniq = np.unique(SessionIDs)

        # In testing mode, only process the first N_TEST sessions
        if TEST:
            Iden_uniq = Iden_uniq[:N_TEST]
        

        # Splitting the datatable into subsets
        Data_subsets = run_with_progress(split_table, Iden_uniq, Parallel=PARALLEL)
        # Building the commands for conversion
        commands = run_with_progress(makeNifti, Data_subsets, Parallel=PARALLEL)
        commands = manager.list([item for sublist in commands for item in sublist])

    # Running the commands
    #run_with_progress(run_cmd, commands, Parallel=PARALLEL)
    run_with_progress(partial(run_cmd, commands=commands), commands, Parallel=PARALLEL)

    if not stop_flag.is_set():
        LOGGER.info('Nifti conversion complete without stop flag')
        LOGGER.info('Removing progress file')
        if os.path.exists('saveNifti_progress.pkl'):
            os.remove('saveNifti_progress.pkl')
    else:
        LOGGER.info('Nifti conversion complete with stop flag')
        save_progress(list(commands), 'saveNifti_progress.pkl')
        LOGGER.info('checkpoint file saved')
    stop_flag.set()

