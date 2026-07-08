import os
#import pydicom as pyd
import glob
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Queue, Manager, cpu_count, Lock
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import signal
import subprocess
import time
from typing import Callable, List, Any
from functools import partial
# Custom imports
from toolbox import ProgressBar, get_logger, run_function
from DICOM import DICOMfilter, DICOMorder

# Global variables for progress bar and lock
#Progress = None
manager = Manager()
disk_space_lock = Lock()
#progress_queue = manager.Queue()
LOGGER = get_logger('03_saveNifti', '/FL_system/data/logs/')

# Define necessary directories
LOAD_DIR = '/FL_system/data/' # Location to load the constructed Data_table_timing.csv ['/FL_system/data/']
SAVE_DIR = '/FL_system/data/nifti/' # Location to save the nifti files ['/FL_system/data/nifti/']
DEBUG = 0
TEST = False
#PROGRESS = False
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
    if available_space < DISK_SPACE_THRESHOLD * 2:
        LOGGER.warning(f'Disk space low: {available_space / 1e9:.1f} GB available (threshold: {DISK_SPACE_THRESHOLD / 1e9:.1f} GB)')
    return available_space > DISK_SPACE_THRESHOLD

def check_source_files(source_path: str) -> bool:
    """Check if the source path file exists contains files."""
    return (len(glob.glob(f'{source_path}/*')) > 0) or (len(glob.glob(f'{source_path}/*/*')) > 0)

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



def run_with_progress(target: Callable[..., Any], items: List[Any], Parallel: bool=True, *args, **kwargs) -> List[Any]:
    """Run a function with a progress bar"""
    # Initialize using a manager to allow for shared progress queue
    #manager = Manager()
    #progress_queue = manager.Queue()
    target_name = target.func.__name__ if isinstance(target, partial) else target.__name__

    # Debugging information
    LOGGER.debug(f'Running {target_name} with progress bar')
    LOGGER.debug(f'Number of items: {len(items)}')
    LOGGER.debug(f'Parallel: {Parallel}')

    # Initialize progress bar
    #if PROGRESS:
    #    Progress = ProgressBar(len(items))
    #    updater_thread = threading.Thread(target=progress_updater, args=(progress_queue, Progress))
    #    updater_thread.start()
    
    # Pass the progress queue to the target function
    #target = partial(progress_wrapper, target=target, progress_queue=progress_queue, *args, **kwargs)

    # Run the target function with a progress bar
    results = []
    t_start = time.time()
    items_index = 0
    if Parallel:
        max_workers = cpu_count() - 1
        LOGGER.info(f'Submitting {len(items)} tasks to ProcessPoolExecutor ({max_workers} workers)')
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, item in enumerate(items):
                if stop_flag.is_set():
                    LOGGER.info(f'[STOP] Stop flag set before submitting item {idx+1}/{len(items)}. Cancelling.')
                    for f in futures:
                        f.cancel()
                    break
                futures.append((idx, executor.submit(target, item, *args, **kwargs)))
                num_done = (idx + 1) % 50 == 0 or idx + 1 == len(items)
                if num_done:
                    elapsed = time.time() - t_start
                    LOGGER.info(f'[{target_name}] Submitted: {idx+1}/{len(items)} items, {elapsed:.0f}s elapsed')
            for idx, future in futures:
                if stop_flag.is_set():
                    LOGGER.info(f'[STOP] Stop flag set after processing {idx+1}/{len(futures)} futures. Cancelling remaining.')
                    for _, f in futures[idx+1:]:
                        f.cancel()
                    break
                try:
                    result = future.result(timeout=1800)
                    results.append(result)
                    if (idx + 1) % 50 == 0 or idx + 1 == len(futures):
                        elapsed = time.time() - t_start
                        LOGGER.info(f'[{target_name}] Progress: {idx+1}/{len(futures)} items, {elapsed:.0f}s elapsed')
                except Exception as e:
                    LOGGER.error(f'[ERROR] Future {idx} ({target_name} item {idx}) failed: {e}', exc_info=True)
    else:
        for items_index, item in enumerate(items):
            if stop_flag.is_set():
                LOGGER.info(f'[STOP] Stop flag set after processing {items_index+1}/{len(items)} items')
                break
            try:
                result = target(item)
                results.append(result)
                if (items_index + 1) % 50 == 0 or items_index + 1 == len(items):
                    elapsed = time.time() - t_start
                    LOGGER.info(f'[{target_name}] Progress: {items_index+1}/{len(items)} items, {elapsed:.0f}s elapsed')
            except Exception as e:
                LOGGER.error(f'[ERROR] Sequential item {items_index} failed: {e}', exc_info=True)

    elapsed_total = time.time() - t_start
    LOGGER.info(f'[{target_name}] Completed in {elapsed_total:.0f}s. {len(results)} results collected')

    # Check if results is a list of tuples before returning zip(*results)
    if results and isinstance(results[0], tuple):
        LOGGER.info(f'[*] Unzipping tuple results for {target_name}')
        return list(zip(*results))
    LOGGER.info(f'[*] Returning {len(results)} results for {target_name}')
    return results

#def progress_updater(queue, progress_bar):
#    while not stop_flag.is_set():
##        try:
 #           item = queue.get(timeout=1)
 #           if item is None:
 #               break
 #           index, status = item
 #           progress_bar.update(index, status)
 #           queue.task_done()
 #       except:
 #           continue

def run_cmd(command, commands):
    SessionID = command[2].split(os.sep)[-1]
    output_dir = command[2]
    file_name = command[4]
    input_file = command[-1]
    input_dir = '/'.join(input_file.split('/')[:-1])
    LOGGER.info(f'[START] {file_name} | input: {input_file} | output: {output_dir}{os.sep}{file_name}.nii')

    if os.path.exists(f'{output_dir}{os.sep}{file_name}.nii'):
        LOGGER.info(f'[SKIP] Nifti file already exists: {file_name}')
        commands.remove(command)
        return

    if stop_flag.is_set():
        LOGGER.info(f'[ABORT] Stop flag set before starting {file_name}')
        return

    with disk_space_lock:
        if not check_disk_space(SAVE_DIR):
            if not stop_flag.is_set():
                LOGGER.warning(f'[ABORT] Disk space low, setting stop flag before {file_name}')
                stop_flag.set()
            return
        if not check_source_files(input_dir):
            if not stop_flag.is_set():
                LOGGER.warning(f'[ABORT] No source files in {input_dir}, setting stop flag')
                stop_flag.set()
            return

    if not os.path.isdir(f'{SAVE_DIR}{SessionID}'):
        try:
            os.mkdir(f'{SAVE_DIR}{SessionID}')
            if DEBUG > 0:
                LOGGER.debug(f'Created directory for {SessionID}')
        except FileExistsError:
            LOGGER.warning(f'Directory for {SessionID} already exists')

    LOGGER.info(f'[RUN] Executing dcm2niix for {file_name}')
    t0 = time.time()
    try:
        if DEBUG == 0:
            result = subprocess.run(command, check=True, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            result = subprocess.run(command, check=True, timeout=600, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())
        elapsed = time.time() - t0
        LOGGER.info(f'[DONE] {file_name} completed in {elapsed:.1f}s from {command[-1]}')
        try:
            commands.remove(command)
        except ValueError:
            LOGGER.warning(f'  Command for {file_name} not in commands list (already removed)')
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        LOGGER.error(f'[TIMEOUT] {file_name} exceeded 600s after {elapsed:.1f}s. Command: {" ".join(command)}')
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        LOGGER.error(f'[FAIL] {file_name} failed after {elapsed:.1f}s')
        error_message = e.stderr.decode() if e.stderr else 'No error message available'
        LOGGER.error(f'  Error converting {command[-1]}: {error_message[:500]}')
    
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
    return Data_table[Data_table['SessionID'] == ID].reset_index(drop=True)

def handle_keyboard_interrupt(signum, frame):
    LOGGER.info('[SIGINT] Keyboard interrupt received. Setting stop flag for graceful shutdown...')
    stop_flag.set()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
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
        LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Step: splitting table for {len(Iden_uniq)} sessions')
        Data_subsets = run_with_progress(split_table, Iden_uniq, Parallel=PARALLEL)
        # Building the commands for conversion
        LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Step: building dcm2niix commands')
        commands = run_with_progress(makeNifti, Data_subsets, Parallel=PARALLEL)
        LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Step: flattening commands list')
        flat_commands = [item for sublist in commands for item in sublist]
        LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Created {len(flat_commands)} commands, transferring to manager.list()...')
        commands = manager.list(flat_commands)
        LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Number of commands: {len(commands)}')
    LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Step: separating priority (raw) from redirected commands')
    raw_cmds = [item for item in commands if 'raw' in item[-1]]
    LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Found {len(raw_cmds)} priority commands')
    commands_priority = manager.list(raw_cmds)
    redirected_cmds = [item for item in commands if 'raw' not in item[-1]]
    LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Found {len(redirected_cmds)} redirected commands')
    commands_redirected = manager.list(redirected_cmds)
    if len(commands_priority) > 0:
        LOGGER.debug(f'Number of priority commands: {len(commands_priority)}')
        run_with_progress(partial(run_cmd, commands=commands), commands_priority, Parallel=PARALLEL)
        if not stop_flag.is_set():
            LOGGER.info('Priority commands complete without stop flag')
            LOGGER.info('Running non-priority from temporary files')
            if len(commands_redirected) > 0:
                LOGGER.debug(f'Number of redirected commands: {len(commands_redirected)}')
                run_with_progress(partial(run_cmd, commands=commands), commands_redirected, Parallel=PARALLEL)
                if not stop_flag.is_set():
                    LOGGER.info('Nifti conversion complete without stop flag')
                    LOGGER.info('Removing progress file')
                    if os.path.exists('saveNifti_progress.pkl'):
                        os.remove('saveNifti_progress.pkl')
                else:
                    LOGGER.info('Nifti conversion complete with stop flag')
                    save_progress(list(commands), 'saveNifti_progress.pkl')
                    LOGGER.info('checkpoint file saved')
        else:
            LOGGER.info('Nifti conversion complete with stop flag')
            save_progress(list(commands), 'saveNifti_progress.pkl')
            LOGGER.info('checkpoint file saved')
    elif len(commands_redirected) > 0:
        LOGGER.debug(f'Number of redirected commands: {len(commands_redirected)}')
        run_with_progress(partial(run_cmd, commands=commands), commands_redirected, Parallel=PARALLEL)
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

