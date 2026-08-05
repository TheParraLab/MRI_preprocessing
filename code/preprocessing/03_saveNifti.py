import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Manager, cpu_count, Lock
from concurrent.futures import ProcessPoolExecutor
import signal
import subprocess
import time
from typing import Callable, List, Any
import shutil
from functools import partial
import argparse
# Custom imports
from toolbox import get_logger

parser = argparse.ArgumentParser(description='Convert DICOM files to NIfTI format')
parser.add_argument('--multi', '-m', action='store_true', help='Use multiprocessing')
parser.add_argument('--test', action='store_true', help='Run in test mode with first 200 sessions')
args = parser.parse_args()

LOGGER = get_logger('03_saveNifti', '/FL_system/data/logs/')

LOAD_DIR = '/FL_system/data/'
SAVE_DIR = '/FL_system/data/nifti/'
DEBUG = 0
TEST = args.test
N_TEST = 200
PARALLEL = args.multi
DISK_SPACE_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5 GB

manager = None
disk_space_lock = None
completed_commands = None
stop_flag = None

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
    target_name = target.func.__name__ if isinstance(target, partial) else target.__name__

    LOGGER.debug(f'Running {target_name} with progress bar')
    LOGGER.debug(f'Number of items: {len(items)}')
    LOGGER.debug(f'Parallel: {Parallel}')

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
            try:
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
            except KeyboardInterrupt:
                LOGGER.info(f'[INTERRUPT] Letting in-flight workers complete, cancelling queued futures from index {idx+1}/{len(futures)}.')
                for _, f in futures[idx+1:]:
                    f.cancel()
                raise
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


def run_cmd(command, disk_space_lock, stop_flag, completed_commands):
    SessionID = command[2].split(os.sep)[-1]
    output_dir = command[2]
    file_name = command[4]
    input_file = command[-1]
    input_dir = '/'.join(input_file.split('/')[:-1])
    LOGGER.info(f'[START] {file_name} | input: {input_file} | output: {output_dir}{os.sep}{file_name}.nii')

    if os.path.exists(f'{output_dir}{os.sep}{file_name}.nii'):
        LOGGER.info(f'[SKIP] Nifti file already exists: {file_name}')
        completed_commands.add(tuple(command))
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
            result = subprocess.run(command, check=True, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:
            result = subprocess.run(command, check=True, timeout=600, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            LOGGER.debug(result.stdout.decode())
        elapsed = time.time() - t0
        LOGGER.info(f'[DONE] {file_name} completed in {elapsed:.1f}s from {command[-1]}')
        completed_commands.add(tuple(command))
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

def split_table(ID, Data_table):
    return Data_table[Data_table['SessionID'] == ID].reset_index(drop=True)

def handle_keyboard_interrupt(signum, frame):
    LOGGER.info('[SIGINT] Keyboard interrupt received. In-flight sessions will complete, queued ones cancelled...')
    raise KeyboardInterrupt('Interrupted')


if __name__ == '__main__':
    manager = Manager()
    disk_space_lock = Lock()
    stop_flag = manager.Event()
    completed_commands = manager.Set()

    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    LOGGER.info('Starting saveNifti: Step 03')
    LOGGER.info(f'LOAD_DIR: {LOAD_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')

    if not shutil.which('dcm2niix'):
        LOGGER.error('dcm2niix not found in PATH. Please install dcm2niix before running this script.')
        exit(1)

    if TEST:
        LOGGER.info(f'Running in test mode with {N_TEST} sessions')

    completed_commands.clear()
    progress = load_progress('saveNifti_progress.pkl')

    if progress:
        LOGGER.info(f'Resuming from checkpoint. {len(progress)} items remaining')
        commands = manager.list(progress)
    else:
        LOGGER.info('No checkpoint found. Starting fresh.')
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)

        Data_table = pd.read_csv(f'{LOAD_DIR}Data_table_timing.csv')
        SessionIDs = Data_table['SessionID']
        Iden_uniq = np.unique(SessionIDs)

        if TEST:
            Iden_uniq = Iden_uniq[:N_TEST]

        LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Splitting table for {len(Iden_uniq)} sessions')
        Data_subsets = []
        for sid in Iden_uniq:
            subset = split_table(sid, Data_table)
            if not subset.empty:
                Data_subsets.append(subset)

        t_start_phase = time.time()
        LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Building dcm2niix commands')
        all_commands = []
        for i, subset in enumerate(Data_subsets):
            cmds = makeNifti(subset)
            all_commands.extend(cmds)
            if (i + 1) % 50 == 0 or i + 1 == len(Data_subsets):
                elapsed = time.time() - t_start_phase
                LOGGER.info(f'Built commands for {i+1}/{len(Data_subsets)} sessions ({elapsed:.0f}s)')

        commands = manager.list(all_commands)
        LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Total commands: {len(commands)}')

    LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Separating priority (raw) from redirected commands')
    raw_cmds = [item for item in commands if 'raw' in item[-1]]
    commands_priority = manager.list(raw_cmds)
    redirected_cmds = [item for item in commands if 'raw' not in item[-1]]
    commands_redirected = manager.list(redirected_cmds)
    LOGGER.info(f'Priority: {len(commands_priority)}, Redirected: {len(commands_redirected)}')

    progress_path = os.path.join(LOAD_DIR, 'saveNifti_progress.pkl')

    def save_checkpoint():
        remaining = [c for c in commands if tuple(c) not in completed_commands]
        save_progress(remaining, 'saveNifti_progress.pkl')
        LOGGER.info(f'Checkpoint saved with {len(remaining)} remaining items')

    def cleanup():
        if os.path.exists(progress_path):
            os.remove(progress_path)
            LOGGER.info('Checkpoint file removed')

    if commands_priority:
        run_with_progress(run_cmd, commands_priority, Parallel=PARALLEL, disk_space_lock=disk_space_lock, stop_flag=stop_flag, completed_commands=completed_commands)

        if not stop_flag.is_set() and commands_redirected:
            run_with_progress(run_cmd, commands_redirected, Parallel=PARALLEL, disk_space_lock=disk_space_lock, stop_flag=stop_flag, completed_commands=completed_commands)

        if not stop_flag.is_set():
            cleanup()
        else:
            save_checkpoint()

    elif commands_redirected:
        run_with_progress(run_cmd, commands_redirected, Parallel=PARALLEL, disk_space_lock=disk_space_lock, stop_flag=stop_flag, completed_commands=completed_commands)

        if not stop_flag.is_set():
            cleanup()
        else:
            save_checkpoint()

