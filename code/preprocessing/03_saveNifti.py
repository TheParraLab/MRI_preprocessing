import os
#import pydicom as pyd
import glob
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Queue, cpu_count, Lock, Event
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import signal
import subprocess
import argparse
import time
from typing import Callable, List, Any
from functools import partial
# Custom imports
from toolbox import ProgressBar, get_logger, run_function, ensure_dir_writable, resolve_dir, _collect_future_map, _terminate_executors, WORKER_TIMEOUT, nifti_stem, is_nifti_file, glob_nifti
from DICOM import DICOMfilter, DICOMorder

# Global variables for progress bar and lock
#Progress = None
disk_space_lock = Lock()
#progress_queue = manager.Queue()
# Deployment-isolated logs; see toolbox.get_log_dir() for resolution order.
LOGGER = get_logger('03_saveNifti')

# Define necessary directories (resolve: flag > env > container default)
parser = argparse.ArgumentParser(description='Convert DICOM files to NIfTI format')
parser.add_argument('--multi', action='store_true', help='Use multiprocessing')
parser.add_argument('--cpus', type=int, default=0, help='Number of parallel workers (ProcessPoolExecutor). 0 = cpu_count()-1. Lower it if dcm2niix stalls over NFS.')
parser.add_argument('--load_dir', type=str, default=None, help='Directory to load Data_table_timing.csv from (default: $DATA_DIR or /FL_system/data/)')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the NIfTI files (default: $NIFTI_DIR or /FL_system/data/nifti/)')
args = parser.parse_args()
LOAD_DIR = resolve_dir(args.load_dir, 'DATA_DIR', '/FL_system/data/')
SAVE_DIR = resolve_dir(args.save_dir, 'NIFTI_DIR', '/FL_system/data/nifti/')

DEBUG = 0
TEST = False
N_TEST = 200
PARALLEL = args.multi
DISK_SPACE_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5 GB
stop_flag = Event()

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
    dest = f'{LOAD_DIR}{filename}'
    ensure_dir_writable(os.path.dirname(dest), context='progress file location')
    if os.path.exists(dest):
        os.remove(dest)
    with open(dest, 'wb') as f:
        pickle.dump(data, f)

def load_progress(filename):
    """Load progress from a file."""
    if os.path.exists(f'{LOAD_DIR}{filename}'):
        LOGGER.info(f'Loading progress from {filename}')
        with open(f'{LOAD_DIR}{filename}', 'rb') as f:
            return pickle.load(f)
    return None



def run_with_progress(target: Callable[..., Any], items: List[Any], Parallel: bool=True, *_extra_args, **_extra_kwargs) -> List[Any]:
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
        max_workers = args.cpus if args.cpus > 0 else (cpu_count() - 1)
        LOGGER.info(f'Running {len(items)} tasks through ProcessPoolExecutor ({max_workers} workers; --cpus={args.cpus})')
        deadline = time.monotonic() + WORKER_TIMEOUT
        executor = ProcessPoolExecutor(max_workers=max_workers)
        try:
            future_map = {executor.submit(target, items[i], *_extra_args, **_extra_kwargs): i
                          for i in range(len(items))}
            ordered = _collect_future_map(future_map, deadline, LOGGER)
            results = [r for r in ordered if r is not None]
        except KeyboardInterrupt:
            LOGGER.info('Interrupted. Terminating workers...')
            _terminate_executors(executor)
            raise
        finally:
            if time.monotonic() < deadline:
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                except Exception as e:
                    LOGGER.warning(f'Graceful shutdown failed: {e!r}; force-terminating')
                    _terminate_executors(executor)
            else:
                LOGGER.error('Worker deadline exceeded — force-terminating workers.')
                _terminate_executors(executor)
    else:
        deadline = time.monotonic() + WORKER_TIMEOUT
        for items_index, item in enumerate(items):
            if stop_flag.is_set() or time.monotonic() >= deadline:
                LOGGER.info(f'[STOP] Stop flag set or deadline exceeded after processing {items_index+1}/{len(items)} items')
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

def _parse_dcm2niix(command):
    """Extract (output_dir, file_name, input_file) from a dcm2niix argv list.

    Parses by flag (-o -> output dir, -f -> output name) rather than by
    positional index, so the result is independent of flag order or the number
    of flags (e.g. the '-z y' compression flag). The final element is always
    the input DICOM path per dcm2niix convention.
    """
    out_dir = None
    file_name = None
    n = len(command)
    i = 0
    while i < n - 1:
        tok = command[i]
        if tok == '-o' and i + 1 < n:
            out_dir = command[i + 1]
            i += 2
        elif tok == '-f' and i + 1 < n:
            file_name = command[i + 1]
            i += 2
        else:
            i += 1
    input_file = command[-1]
    return out_dir, file_name, input_file

def run_cmd(command, commands):
    output_dir, file_name, input_file = _parse_dcm2niix(command)
    if output_dir is None or file_name is None:
        LOGGER.error(f'[SKIP] Cannot parse dcm2niix command (missing -o or -f): {command}')
        try:
            commands.remove(command)
        except ValueError:
            pass
        return
    SessionID = output_dir.split(os.sep)[-1]
    input_dir = '/'.join(input_file.split('/')[:-1])
    # output will be a .nii.gz now; only reference the stem in the log
    LOGGER.info(f'[START] {file_name} | input: {input_file} | output: {output_dir}{os.sep}{file_name}.nii.gz')

    # accept either existing format (backcompat with older runs)
    if os.path.exists(f'{output_dir}{os.sep}{file_name}.nii') or os.path.exists(f'{output_dir}{os.sep}{file_name}.nii.gz'):
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
        ensure_dir_writable(f'{SAVE_DIR}{SessionID}', context=f'save dir for session {SessionID}')
        if DEBUG > 0:
            LOGGER.debug(f'Created directory for {SessionID}')

    LOGGER.info(f'[RUN] Executing dcm2niix for {file_name}')
    t0 = time.time()
    proc = None
    try:
        proc = subprocess.run(command, timeout=600,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        elapsed = time.time() - t0
        out_path = f'{output_dir}{os.sep}{file_name}.nii.gz'
        ok_files = sorted(
            f for f in os.listdir(output_dir)
            if (f.startswith(file_name + '.') or f.startswith(file_name + '_') or f == file_name)
            and (f.endswith('.nii') or f.endswith('.nii.gz'))
        ) if os.path.isdir(output_dir) else []
        if proc.returncode != 0 or (not os.path.exists(out_path) and not ok_files):
            reason = (proc.stderr or proc.stdout or '').strip()[-500:]
            listing = f'existing nifti-like files in {output_dir}: {ok_files}'
            LOGGER.error(f'[FAIL] {file_name}: dcm2niix returned rc={proc.returncode}, '
                         f'expected {out_path} present={os.path.exists(out_path)}. '
                         f'{listing}. dcm2niix reported: {reason if reason else "no output captured"}')
            return
        if not os.path.exists(out_path):
            LOGGER.info(f'[DONE-SUFFIX] {file_name}: dcm2niix wrote {ok_files} instead of the exact '
                        f'{out_path} (name suffix). Continuing.')
        LOGGER.info(f'[DONE] {file_name} completed in {elapsed:.1f}s')
        try:
            commands.remove(command)
        except ValueError:
            LOGGER.warning(f'  Command for {file_name} not in commands list (already removed)')
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        partial = (e.stderr or '').strip()[-500:]
        LOGGER.error(f'[TIMEOUT] {file_name} exceeded 600s. Command: {" ".join(command)}. '
                     f'dcm2niix reported so far: {partial if partial else "no output captured"}')
    
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
        commands.append(['dcm2niix', '-z', 'y', '-o', f'{SAVE_DIR}{SessionID}', '-f', Descriptor[i], LoadPATH[i]])
    return commands

def split_table(ID):
    return Data_table[Data_table['SessionID'] == ID].reset_index(drop=True)

def _stemish(name: str) -> str:
    n = name
    for s in ('.nii.gz', '.nii'):
        if n.endswith(s):
            n = n[:-len(s)]
            break
    return n

def audit_nifti_directory():
    """Audit the NIfTI directory against Data_table_timing.csv (post-conversion).

    Per session: expected files derive from the table's Major column
    ('{Major:02}.nii' or .nii.gz), compared with what is actually on disk. Catches (a)
    rows with no file (conversion failed / timed out but step 03 moved on),
    (b) files with no row (leftovers from a previous run — 'unrequested
    data' that a later alignment would be tempted to pair), (c) duplicate
    Majors in the table (dual-pre collisions: two rows, one filename), and
    (d) ghost sessions — table rows with no directory at all (dcm2niix failed
    before output was created), reported as all-expected-files-missing.

    Pure audit: logs per-finding at ERROR/WARNING and writes
    <deployment log dir>/nifti_audit.json. Never aborts the run — the origin
    of each mismatch is already reported where it happened (step 02 ordering,
    per-command run_cmd failures above), and step 06 decides what to trust.

    Returns True if the union of table sessions and disk sessions shows exact
    parity (no missing, extra, duplicate-Major, or ghost-session findings).
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
    all_sessions = sorted(set(expected_by_session) | disk_sessions)
    audit, n_missing_rows, n_extra_files, n_dup, n_ghost_sessions = 0, 0, 0, 0, 0

    for sid in all_sessions:
        if sid in expected_by_session and sid not in disk_sessions:
            majors = expected_by_session[sid]
            LOGGER.error(f'[AUDIT] {sid}: present in timing table but has NO directory in nifti dir '
                         f'({len(majors)} expected files — conversion failed before output was created)')
            audit += 1
            n_ghost_sessions += 1
            n_missing_rows += len(majors)
            continue

        sdir = os.path.join(SAVE_DIR, sid)
        if sid not in expected_by_session:
            LOGGER.error(f'[AUDIT] Session {sid} exists in nifti dir but has NO timing-table rows '
                         f'({len(os.listdir(sdir))} files on disk — unrequested data)')
            audit += 1
            continue

        majors = expected_by_session[sid]
        exp_names = sorted({f'{m:02d}' for m in majors})
        on_disk = set(_stemish(f) for f in os.listdir(sdir)
                      if is_nifti_file(f) and not (f.endswith('_RAS.nii') or f.endswith('_RAS.nii.gz')))

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

    clean = (audit == 0 and n_ghost_sessions == 0
             and not (n_missing_rows or n_extra_files or n_dup))
    if clean:
        LOGGER.info('[AUDIT] NIfTI directory matches timing table for all sessions')

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
                'clean': clean,
                'missing_files': n_missing_rows,
                'extra_files': n_extra_files,
                'duplicate_major_rows': n_dup,
                'ghost_sessions': n_ghost_sessions,
            }, fh, indent=2)
        LOGGER.info(f'[AUDIT] Wrote {out_json} '
                    f'(clean={clean}, '
                    f'missing={n_missing_rows}, extra={n_extra_files}, '
                    f'dup_majors={n_dup}, ghost_sessions={n_ghost_sessions})')
    except Exception as e:
        LOGGER.warning(f'[AUDIT] Could not write audit json: {e}')

    return clean

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
        commands = list(progress)
    else:
        LOGGER.info('No progress file found. Starting from scratch')
        if not os.path.exists(SAVE_DIR):
            ensure_dir_writable(SAVE_DIR, context='saveNifti output dir')

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
        LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Created {len(flat_commands)} commands, converting to list...')
        commands = list(flat_commands)
        LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Number of commands: {len(commands)}')
    LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Step: separating priority (raw) from redirected commands')
    raw_cmds = [item for item in commands if 'raw' in item[-1]]
    LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Found {len(raw_cmds)} priority commands')
    commands_priority = list(raw_cmds)
    redirected_cmds = [item for item in commands if 'raw' not in item[-1]]
    LOGGER.info(f'[{time.strftime("%H:%M:%S")}] Found {len(redirected_cmds)} redirected commands')
    commands_redirected = list(redirected_cmds)
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

