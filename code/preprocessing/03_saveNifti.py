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
import argparse
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
# Deployment-isolated logs; see toolbox.get_log_dir() for resolution order.
LOGGER = get_logger('03_saveNifti')

# Define necessary directories
LOAD_DIR = '/FL_system/data/' # Location to load the constructed Data_table_timing.csv ['/FL_system/data/']
SAVE_DIR = '/FL_system/data/nifti/' # Location to save the nifti files ['/FL_system/data/nifti/']
parser = argparse.ArgumentParser(description='Convert DICOM files to NIfTI format')
parser.add_argument('--multi', '-m', action='store_true', help='Use multiprocessing')
args = parser.parse_args()

DEBUG = 0
TEST = False
N_TEST = 200
PARALLEL = args.multi
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

    results = []
    t_start = time.time()
    items_index = 0
    if Parallel:
        max_workers = cpu_count() - 1
        LOGGER.info(f'Running {len(items)} tasks through ProcessPoolExecutor ({max_workers} workers)')
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending = {}
            next_idx = 0

            while pending or next_idx < len(items):
                if stop_flag.is_set():
                    for f in pending.values():
                        f.cancel()
                    pending.clear()
                    break

                # Submit up to max_workers worth of pending tasks
                while next_idx < len(items) and len(pending) < max_workers:
                    fut = executor.submit(target, items[next_idx], *args, **kwargs)
                    pending[fut] = next_idx
                    next_idx += 1

                if not pending:
                    break

                time.sleep(0.5)

                # Collect any finished futures
                done_futs = [f for f in pending if f.done()]
                for f in done_futs:
                    idx = pending.pop(f)
                    try:
                        result = f.result(timeout=1800)
                        results.append(result)
                        results_len = idx + 1
                        if results_len % 50 == 0 or results_len == len(items):
                            elapsed = time.time() - t_start
                            LOGGER.info(f'[{target_name}] Progress: {results_len}/{len(items)} ({elapsed:.0f}s)')
                    except Exception as e:
                        LOGGER.error(f'[ERROR] Item {idx} failed: {e}', exc_info=True)

            for f in pending.values():
                f.cancel()
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

def audit_nifti_directory():
    """Audit the NIfTI directory against Data_table_timing.csv (post-conversion).

    Per session: expected files derive from the table's Major column
    ('{Major:02}.nii'), compared with what is actually on disk. Catches (a)
    rows with no file (conversion failed / timed out but step 03 moved on),
    (b) files with no row (leftovers from a previous run — 'unrequested
    data' that a later alignment would be tempted to pair), and (c) duplicate
    Majors in the table (dual-pre collisions: two rows, one filename).

    Pure audit: logs per-finding at ERROR/WARNING and writes
    <deployment log dir>/nifti_audit.json. Never aborts the run — the origin
    of each mismatch is already reported where it happened (step 02 ordering,
    per-command run_cmd failures above), and step 06 decides what to trust.

    Returns True if every session on disk has exact parity with its table rows.
    """
    timing_csv = f'{LOAD_DIR}Data_table_timing.csv'
    if not os.path.exists(timing_csv):
        LOGGER.error(f'[AUDIT] Timing table {timing_csv} not found, skipping NIfTI audit')
        return False

    from toolbox import get_log_dir
    table = pd.read_csv(timing_csv, low_memory=False)
    table['SessionID'] = table['SessionID'].astype(str)
    expected_by_session = {}
    for sid, grp in table.groupby('SessionID'):
        majors = [int(m) for m in grp['Major']]
        expected_by_session[str(sid)] = majors

    disk_sessions = set(
        d for d in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, d)))
    audit, n_missing_rows, n_extra_files, n_dup = 0, 0, 0, 0

    for sid in sorted(disk_sessions):
        sdir = os.path.join(SAVE_DIR, sid)
        if sid not in expected_by_session:
            LOGGER.error(f'[AUDIT] Session {sid} exists in nifti dir but has NO timing-table rows '
                         f'({len(os.listdir(sdir))} files on disk — unrequested data)')
            audit += 1
            continue

        majors = expected_by_session[sid]
        exp_names = sorted(set(f'{m:02d}.nii' for m in majors))
        on_disk = set(f for f in os.listdir(sdir)
                      if f.endswith('.nii') and not f.endswith('_RAS.nii'))

        missing = [f for f in exp_names if f not in on_disk]
        extra = sorted(f for f in on_disk if f not in set(exp_names))
        mc = {f'{m:02d}': int(c) for m, c in pd.Series(majors).value_counts().items() if c > 1}

        ok = not (missing or extra or mc)
        audit += 0 if ok else 1
        n_missing_rows += len(missing)
        n_extra_files += len(extra)
        n_dup += sum(mc.values()) - len(mc)

        for f in missing:
            LOGGER.error(f'[AUDIT] {sid}: expected {f} (from table Major column) is MISSING on disk '
                         f'— conversion for this scan failed or was skipped; step 06 cannot pair this row')
        if extra:
            LOGGER.warning(f'[AUDIT] {sid}: {len(extra)} file(s) on disk with no table row '
                           f'(leftover/unrequested): {extra}')
        if mc:
            LOGGER.error(f'[AUDIT] {sid}: duplicate Major values in table {mc} — filename collision, '
                         f'one conversion overwrote another')

    if audit == 0 and (n_missing_rows or n_extra_files or n_dup) == 0:
        LOGGER.info('[AUDIT] NIfTI directory matches timing table for all sessions present on disk')

    try:
        log_dir = get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        out_json = os.path.join(log_dir, 'nifti_audit.json')
        import json
        with open(out_json, 'w') as fh:
            json.dump({
                'timing_table': timing_csv,
                'nifti_dir': SAVE_DIR,
                'sessions_on_disk': len(disk_sessions),
                'clean': audit == 0 and not (n_missing_rows or n_extra_files or n_dup),
                'missing_files': n_missing_rows,
                'extra_files': n_extra_files,
                'duplicate_major_rows': n_dup,
            }, fh, indent=2)
        LOGGER.info(f'[AUDIT] Wrote {out_json} '
                    f'(clean={audit == 0 and not (n_missing_rows or n_extra_files or n_dup)}, '
                    f'missing={n_missing_rows}, extra={n_extra_files}, dup_majors={n_dup})')
    except Exception as e:
        LOGGER.warning(f'[AUDIT] Could not write audit json: {e}')

    return audit == 0 and not (n_missing_rows or n_extra_files or n_dup)

def handle_keyboard_interrupt(signum, frame):
    LOGGER.info('[SIGINT] Keyboard interrupt received. In-flight sessions will complete, queued ones cancelled...')
    raise KeyboardInterrupt('Interrupted')

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

    # Post-conversion audit: compare what step 02 said we needed (Major column)
    # with what dcm2niix actually produced. Surfaces missed conversions,
    # leftover 'unrequested' files, and duplicate-Major collisions that a later
    # positional alignment would otherwise silently mis-pair. Pure audit — logs,
    # writes <LOG_DIR>/nifti_audit.json, never aborts (step 06 decides trust).
    audit_nifti_directory()

    stop_flag.set()

