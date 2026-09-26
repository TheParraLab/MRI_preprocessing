import os
#import pydicom as pyd
import glob
import math
import pickle
import numpy as np
import nibabel as nib
import pandas as pd
import multiprocessing
from multiprocessing import Queue, cpu_count, Lock, Event
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import signal
import subprocess
import argparse
import sys
import time
from typing import Callable, List, Any
from functools import partial
# Custom imports
from toolbox import ProgressBar, get_logger, run_function, ensure_dir_writable, resolve_dir, _collect_future_map, _terminate_executors, WORKER_TIMEOUT, nifti_stem, is_nifti_file, glob_nifti, _drain_mp_log_to_parent, _init_child_logger
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
parser.add_argument('--cpus', type=int, default=0,
                    help='Number of parallel workers (ProcessPoolExecutor). 0 = auto: '
                         'min(8, max(2, (cpus-1)//2)) — sized for I/O-bound dcm2niix, not CPU. '
                         'Lower it further if dcm2niix stalls over NFS.')
parser.add_argument('--dcm2niix_timeout', type=int, default=600,
                    help='Per-session dcm2niix wall-clock timeout in seconds (default 600). '
                         'A timed-out session is cleaned up and retried on resume.')
parser.add_argument('--allow-partial', action='store_true',
                    help='Exit 0 even if the end-of-run NIfTI audit finds missing/mismatched '
                         'files (default: exit 1).')
parser.add_argument('--load_dir', type=str, default=None, help='Directory to load Data_table_timing.csv from (default: $DATA_DIR or /FL_system/data/)')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the NIfTI files (default: $NIFTI_DIR or /FL_system/data/nifti/)')
parser.add_argument('--ids_file', type=str, default=None,
                    help='CSV/txt file containing one ID per line. If provided, only process sessions whose name appears in this file.')
args = parser.parse_args()
LOAD_DIR = resolve_dir(args.load_dir, 'DATA_DIR', '/FL_system/data/')
SAVE_DIR = resolve_dir(args.save_dir, 'NIFTI_DIR', '/FL_system/data/nifti/')

DEBUG = 0
TEST = False
N_TEST = 200
PARALLEL = args.multi
DISK_SPACE_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5 GB
# Per-session dcm2niix wall-clock timeout (seconds); also the per-item budget
# used to size the parallel phase deadline (see run_with_progress).
DCM2NIIX_TIMEOUT = args.dcm2niix_timeout
# Wall-clock guard for the per-item disk-space/source NFS checks. A check that
# cannot return within this window (stale NFS handle, disk being swapped) is
# treated as "disk/source unavailable" instead of stalling the whole pool.
CHECK_TIMEOUT = float(os.environ.get('MRI_CHECK_TIMEOUT', '15'))
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



def _result(status, command, file_name, session, error=None, t0=None):
    """Structured per-item return contract for run_cmd.

    The parent never receives None from a live worker (a None result means
    the worker process died); every status is actionable:
      ok/skipped/dropped -> command is removed from the parent's `commands`
                            (done, or permanently unprocessable);
      failed/stopped     -> command stays in `commands` and is retried on
                            the next (resumed) run.
    """
    return {
        'status': status,
        'command': command,
        'file_name': file_name,
        'session': session,
        'error': error,
        'duration_s': round(time.time() - t0, 2) if t0 is not None else 0.0,
    }


def _bounded(fn, fn_args, what):
    """Run fn(*fn_args) with a wall-clock bound.

    The disk-space/source checks hit NFS metadata that can block indefinitely
    on a stale handle (e.g. the disk being swapped). A stuck call must stall
    only the worker that issued it — never the pool — so the call runs in a
    daemon thread; if it is still running after CHECK_TIMEOUT seconds it is
    reported as a failure (the thread is abandoned; it is daemon and dies
    with the process at end of run).
    """
    outcome = {}

    def _run():
        try:
            outcome['ok'] = bool(fn(*fn_args))
        except Exception as exc:
            outcome['ok'] = False
            outcome['err'] = repr(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(CHECK_TIMEOUT)
    if t.is_alive():
        LOGGER.warning(f'[CHECK] {what} did not return within {CHECK_TIMEOUT:.0f}s — '
                       f'treating as unavailable (stale NFS / disk swap?)')
        return False
    if not outcome.get('ok') and outcome.get('err'):
        LOGGER.warning(f'[CHECK] {what} raised: {outcome["err"]}')
    return outcome.get('ok', False)


def _remove_completed(commands, results, phase):
    """Parent-side bookkeeping.

    Workers run in forked processes, so any `commands.remove(...)` they
    perform only touches their own copy. The parent removes commands here,
    from the structured results, so the checkpoint (and the "N remaining"
    count) reflects reality.
    """
    removed = 0
    for res in results:
        if isinstance(res, dict) and res.get('status') in ('ok', 'skipped', 'dropped'):
            cmd = res.get('command')
            if cmd is None:
                continue
            try:
                commands.remove(cmd)
                removed += 1
            except ValueError:
                pass
    LOGGER.info(f'[{phase}] {removed} of {len(results)} completed commands removed from '
                f'bookkeeping ({len(commands)} remaining)')
    return removed


def _log_phase_summary(phase, results, n_dispatched):
    """Loud per-phase outcome summary (status counts + error details)."""
    counts = {}
    errors = []
    for res in results:
        if not isinstance(res, dict):
            continue
        counts[res.get('status')] = counts.get(res.get('status'), 0) + 1
        if res.get('error'):
            errors.append(f"{res.get('file_name') or res.get('session') or '?'}: {res['error'][:300]}")
    n_lost = n_dispatched - len(results)
    summary = ', '.join(f'{k}={v}' for k, v in sorted(counts.items())) or 'none'
    LOGGER.info(f'[{phase}] summary: {summary}'
                + (f', {n_lost} item(s) lost to worker death (kept for retry)' if n_lost > 0 else ''))
    for line in errors[:25]:
        LOGGER.error(f'[{phase}] {line}')
    if len(errors) > 25:
        LOGGER.error(f'[{phase}] ... and {len(errors) - 25} more failures '
                     f'(see per-item [FAIL]/[TIMEOUT] lines above)')


def _finalize_progress(commands):
    """Checkpoint or clear the progress file after the conversion phases.

    The file lives in LOAD_DIR (see save_progress/load_progress). The
    previous clean-run branches removed a CWD-relative copy, which only
    worked when CWD == LOAD_DIR; a stale file there made the next run
    resume from checkpoint and silently ignore --ids_file.
    """
    progress_name = 'saveNifti_progress.pkl'
    if stop_flag.is_set():
        save_progress(list(commands), progress_name)
        LOGGER.info(f'checkpoint file saved ({len(commands)} commands remaining)')
        return
    progress_path = f'{LOAD_DIR}{progress_name}'
    if os.path.exists(progress_path):
        os.remove(progress_path)
        LOGGER.info(f'Removed {progress_path} (all conversions complete)')


def run_with_progress(target: Callable[..., Any], items: List[Any], Parallel: bool=True,
                      per_item_timeout: float = None, *_extra_args, **_extra_kwargs) -> List[Any]:
    """Run a function with a progress bar"""
    target_name = target.func.__name__ if isinstance(target, partial) else target.__name__

    # Debugging information
    LOGGER.debug(f'Running {target_name} with progress bar')
    LOGGER.debug(f'Number of items: {len(items)}')
    LOGGER.debug(f'Parallel: {Parallel}')

    results = []
    t_start = time.time()
    items_index = 0
    if Parallel:
        # I/O-bound sizing: dcm2niix is network/NFS-bound, not CPU-bound.
        # Auto default follows the toolbox P_role='io' rule:
        # min(8, max(2, (cpus-1)//2)).
        auto_cpus = max(1, (cpu_count() or 2) - 1)
        max_workers = args.cpus if args.cpus > 0 else min(8, max(2, auto_cpus // 2))
        LOGGER.info(f'Running {len(items)} tasks through ProcessPoolExecutor ({max_workers} workers; --cpus={args.cpus})')
        if per_item_timeout is not None:
            waves = math.ceil(len(items) / max_workers) if max_workers > 0 else 0
            grace = 300.0
            budget = waves * per_item_timeout + grace
            deadline = time.monotonic() + budget
            LOGGER.info(f'Phase budget: {waves} wave(s) x {per_item_timeout:.0f}s per-item + {grace:.0f}s grace '
                        f'= {budget:.0f}s for {len(items)} items (per-item guard: dcm2niix subprocess timeout)')
        else:
            deadline = time.monotonic() + WORKER_TIMEOUT
        # Cross-process log routing: workers push records to _mp_q and the
        # parent drain thread forwards them into this run's file+stream
        # pipeline. Without this initializer the forked children inherit the
        # parent's in-memory log queue (whose consumer thread does not exist
        # in the child) and every worker log line is silently dropped.
        # Unbounded on purpose: logging must never block a worker even if the
        # parent's NFS flush stalls.
        _mp_q = multiprocessing.Queue()
        _mp_q_stop = threading.Event()
        _mp_q_thread = threading.Thread(target=_drain_mp_log_to_parent,
                                        args=(_mp_q, LOGGER, _mp_q_stop),
                                        daemon=False)
        _mp_q_thread.start()
        executor = ProcessPoolExecutor(max_workers=max_workers,
                                       initializer=_init_child_logger,
                                       initargs=(LOGGER.name, LOGGER._log_level,
                                                 LOGGER._file_path, LOGGER._formatter_str,
                                                 _mp_q))
        try:
            future_map = {executor.submit(target, items[i], *_extra_args, **_extra_kwargs): i
                          for i in range(len(items))}
            ordered = _collect_future_map(future_map, deadline, LOGGER, executor=executor)
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
            # Tear down the worker-log queue: signal stop, wait for drain, close pipe.
            _mp_q_stop.set()
            _mp_q_thread.join(timeout=5)
            try:
                _mp_q.close()
                _mp_q.join_thread()
            except Exception:
                pass
    else:
        for items_index, item in enumerate(items):
            if stop_flag.is_set():
                LOGGER.info(f'[STOP] Stop flag set before item {items_index+1}/{len(items)} — stopping serial run')
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

def _cleanup_partial(output_dir: str, file_name: str):
    """Remove an incomplete/truncated NIfTI output so a resumed run retries
    this conversion instead of skipping it.

    Called from `run_cmd` in the dcm2niix-failure, timeout, and
    invalid-output code paths. Safe to call repeatedly: it only removes
    files belonging to the currently-failed command.
    """
    for ext in ('.nii.gz', '.nii'):
        p = f'{output_dir}{os.sep}{file_name}{ext}'
        if os.path.exists(p):
            try:
                os.remove(p)
                LOGGER.info(f'[CLEAN] Removed partial output: {p}')
            except Exception as exc:
                # Do not block the retry on a delete that fails (e.g. file
                # already gone, permissions). The next run will retry either way.
                LOGGER.warning(f'[CLEAN] Could not remove partial output {p}: {exc!r}')


def _validate_nifti(path: str, file_name: str) -> bool:
    """Open a freshly-written NIfTI and confirm it is complete and readable.

    Guards against the failure modes that slip past dcm2niix's exit code:
      - zero-byte output
      - truncated gzip (dcm2niix wrote but the process was killed mid-write)
      - degenerate/near-empty volumes (max dimension < 4 voxels)
    Returns True if the file passed all checks. False otherwise (and the
    caller is expected to invoke `_cleanup_partial`).
    """
    base = os.path.basename(path)
    try:
        if not os.path.exists(path):
            LOGGER.error(f'[INVALID] {file_name}: {base} not found')
            return False
        if os.path.getsize(path) == 0:
            LOGGER.error(f'[INVALID] {file_name}: {base} is zero-byte')
            return False
        img = nib.load(path)
        if max(img.shape) < 4:
            LOGGER.error(f'[INVALID] {file_name}: {base} degenerate shape {img.shape} (max dim < 4)')
            return False
        # Force a pixel read. This catches a truncated gzip where the gzip
        # header is present but the compressed stream ends mid-write.
        img.get_fdata()
        return True
    except Exception as exc:
        LOGGER.error(f'[INVALID] {file_name}: {base} unreadable: {exc!r}')
        return False

def run_cmd(command, commands):
    t0 = time.time()
    output_dir, file_name, input_file = _parse_dcm2niix(command)
    if output_dir is None or file_name is None:
        LOGGER.error(f'[SKIP] Cannot parse dcm2niix command (missing -o or -f): {command}')
        return _result('dropped', command, None, None,
                       error='unparseable dcm2niix command', t0=t0)
    SessionID = output_dir.split(os.sep)[-1]
    input_dir = '/'.join(input_file.split('/')[:-1])
    # output will be a .nii.gz now; only reference the stem in the log
    LOGGER.info(f'[START] {file_name} | input: {input_file} | output: {output_dir}{os.sep}{file_name}.nii.gz')

    # accept either existing format (backcompat with older runs). Before
    # treating it as "done", validate the bytes — a pre-existing file may
    # be a leftover corrupt volume from an interrupted prior run.
    existing = None
    for ext in ('.nii.gz', '.nii'):
        candidate = f'{output_dir}{os.sep}{file_name}{ext}'
        if os.path.exists(candidate):
            existing = candidate
            break
    if existing is not None:
        if _validate_nifti(existing, file_name):
            LOGGER.info(f'[SKIP] Nifti file already exists and is valid: {file_name}')
            return _result('skipped', command, file_name, SessionID, t0=t0)
        else:
            LOGGER.warning(f'[OVERWRITE] {file_name}: existing file failed validation; '
                           f'will re-run dcm2niix to replace it.')
            _cleanup_partial(output_dir, file_name)

    if stop_flag.is_set():
        LOGGER.info(f'[ABORT] Stop flag set before starting {file_name}')
        return _result('stopped', command, file_name, SessionID, t0=t0)

    # Circuit breaker for the (swappable) disk: verify target free space and
    # source presence immediately before the write. The checks are read-only
    # NFS calls and run OUTSIDE the lock (each bounded by CHECK_TIMEOUT), so
    # a stale handle stalls only this worker. The lock serializes only the
    # I/O-free abort decision below.
    disk_ok = _bounded(check_disk_space, (SAVE_DIR,), f'disk space check for {SAVE_DIR}')
    src_ok = disk_ok and _bounded(check_source_files, (input_dir,), f'source file check for {input_dir}')
    if not disk_ok or not src_ok:
        with disk_space_lock:
            if not stop_flag.is_set():
                LOGGER.warning(f'[ABORT] Disk/source unavailable before {file_name} — setting stop flag')
                stop_flag.set()
        return _result('stopped', command, file_name, SessionID,
                       error='disk space or source files unavailable', t0=t0)

    if not os.path.isdir(f'{SAVE_DIR}{SessionID}'):
        ensure_dir_writable(f'{SAVE_DIR}{SessionID}', context=f'save dir for session {SessionID}')
        if DEBUG > 0:
            LOGGER.debug(f'Created directory for {SessionID}')

    # Re-check the stop flag after the (possibly slow) pre-flight checks so a
    # stop raised by another worker during those checks cannot race a write.
    if stop_flag.is_set():
        LOGGER.info(f'[ABORT] Stop flag set after pre-flight checks for {file_name}')
        return _result('stopped', command, file_name, SessionID, t0=t0)
    LOGGER.info(f'[RUN] Executing dcm2niix for {file_name}')
    run_t0 = time.time()
    timed_out = False
    out, err = '', ''
    try:
        # start_new_session: dcm2niix leads its own process group so a
        # timeout (or a pool terminate) kills it together with any children
        # it spawns — no orphaned writers left on the (swapped) disk.
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        try:
            out, err = proc.communicate(timeout=DCM2NIIX_TIMEOUT)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            out, err = proc.communicate()
    except FileNotFoundError as e:
        LOGGER.error(f'[FAIL] {file_name}: dcm2niix executable not found: {e!r}')
        return _result('failed', command, file_name, SessionID,
                       error=f'dcm2niix executable not found: {e!r}', t0=t0)

    if timed_out:
        elapsed = time.time() - run_t0
        partial_out = (err or '').strip()[-500:]
        _cleanup_partial(output_dir, file_name)
        LOGGER.error(f'[TIMEOUT] {file_name} exceeded {DCM2NIIX_TIMEOUT}s after {elapsed:.0f}s. '
                     f'Command: {" ".join(command)}. '
                     f'dcm2niix reported so far: {partial_out if partial_out else "no output captured"}')
        return _result('failed', command, file_name, SessionID,
                       error=f'dcm2niix timeout after {DCM2NIIX_TIMEOUT}s', t0=t0)

    elapsed = time.time() - run_t0
    out_path = f'{output_dir}{os.sep}{file_name}.nii.gz'
    ok_files = sorted(
        f for f in os.listdir(output_dir)
        if (f.startswith(file_name + '.') or f.startswith(file_name + '_') or f == file_name)
        and (f.endswith('.nii') or f.endswith('.nii.gz'))
    ) if os.path.isdir(output_dir) else []
    if proc.returncode != 0 or (not os.path.exists(out_path) and not ok_files):
        reason = (err or out or '').strip()[-500:]
        listing = f'existing nifti-like files in {output_dir}: {ok_files}'
        LOGGER.error(f'[FAIL] {file_name}: dcm2niix returned rc={proc.returncode}, '
                     f'expected {out_path} present={os.path.exists(out_path)}. '
                     f'{listing}. dcm2niix reported: {reason if reason else "no output captured"}')
        _cleanup_partial(output_dir, file_name)
        return _result('failed', command, file_name, SessionID,
                       error=f'dcm2niix rc={proc.returncode}: {reason[:300] if reason else "no output"}', t0=t0)
    if not os.path.exists(out_path):
        LOGGER.info(f'[DONE-SUFFIX] {file_name}: dcm2niix wrote {ok_files} instead of the exact '
                    f'{out_path} (name suffix). Continuing.')
    # Pick which file actually exists (exact-name preferred; else the
    # suffix variant dcm2niix may have written) and validate it.
    validate_path = out_path if os.path.exists(out_path) else (f'{output_dir}/{ok_files[0]}' if ok_files else out_path)
    if not _validate_nifti(validate_path, file_name):
        LOGGER.warning(f'[RETRY] {file_name}: output failed validation; it will remain in '
                       f'`commands` and be retried on resume.')
        _cleanup_partial(output_dir, file_name)
        return _result('failed', command, file_name, SessionID,
                       error='output failed validation', t0=t0)
    LOGGER.info(f'[DONE] {file_name} completed in {elapsed:.1f}s')
    return _result('ok', command, file_name, SessionID, t0=t0)
    
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

    Returns True on exact parity (no missing, extra, duplicate-Major, or
    ghost-session findings), False if mismatches were found, and None if the
    audit could not be run (timing table missing).
    """
    timing_csv = f'{LOAD_DIR}Data_table_timing.csv'
    if not os.path.exists(timing_csv):
        LOGGER.error(f'[AUDIT] Timing table {timing_csv} not found, skipping NIfTI audit')
        return None

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
        # Load IDs to filter by (if --ids_file provided)
        if args.ids_file is not None:
            with open(args.ids_file, 'r') as f:
                ids_to_process = set(line.strip() for line in f if line.strip())
            LOGGER.info(f'Loaded {len(ids_to_process)} IDs from {args.ids_file}')
            Iden_uniq = [s for s in Iden_uniq if s in ids_to_process]
            LOGGER.info(f'Filtered to {len(Iden_uniq)} sessions matching IDs')
            if len(Iden_uniq) == 0:
                LOGGER.warning('[IDS] No sessions from --ids_file found in Data_table_timing.csv — nothing to do')
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
        results_p = run_with_progress(partial(run_cmd, commands=commands), commands_priority,
                                      Parallel=PARALLEL, per_item_timeout=DCM2NIIX_TIMEOUT)
        _remove_completed(commands, results_p, 'priority')
        _log_phase_summary('priority', results_p, len(commands_priority))
        if not stop_flag.is_set():
            LOGGER.info('Priority commands complete without stop flag')
            LOGGER.info('Running non-priority from temporary files')
            if len(commands_redirected) > 0:
                LOGGER.debug(f'Number of redirected commands: {len(commands_redirected)}')
                results_r = run_with_progress(partial(run_cmd, commands=commands), commands_redirected,
                                              Parallel=PARALLEL, per_item_timeout=DCM2NIIX_TIMEOUT)
                _remove_completed(commands, results_r, 'redirected')
                _log_phase_summary('redirected', results_r, len(commands_redirected))
                if not stop_flag.is_set():
                    LOGGER.info('Nifti conversion complete without stop flag')
                else:
                    LOGGER.info('Nifti conversion complete with stop flag')
        else:
            LOGGER.info('Priority commands complete with stop flag; skipping redirected commands')
    elif len(commands_redirected) > 0:
        LOGGER.debug(f'Number of redirected commands: {len(commands_redirected)}')
        results_r = run_with_progress(partial(run_cmd, commands=commands), commands_redirected,
                                      Parallel=PARALLEL, per_item_timeout=DCM2NIIX_TIMEOUT)
        _remove_completed(commands, results_r, 'redirected')
        _log_phase_summary('redirected', results_r, len(commands_redirected))
        if not stop_flag.is_set():
            LOGGER.info('Nifti conversion complete without stop flag')
        else:
            LOGGER.info('Nifti conversion complete with stop flag')

    _finalize_progress(commands)

    # Post-conversion audit: compare what step 02 said we needed (Major column)
    # with what dcm2niix actually produced. Surfaces missed conversions,
    # leftover 'unrequested' files, and duplicate-Major collisions that a later
    # positional alignment would otherwise silently mis-pair. Pure audit — logs,
    # writes <LOG_DIR>/nifti_audit.json, never aborts (step 06 decides trust).
    # `clean` is True/False when the audit ran, None when it could not run
    # (timing table missing); only a real mismatch exits non-zero.
    audit_clean = audit_nifti_directory()

    stop_flag.set()
    if audit_clean is False and not args.allow_partial:
        LOGGER.error('[AUDIT] NIfTI audit reported missing/mismatched files — exiting non-zero '
                     '(use --allow-partial to exit 0 despite this)')
        sys.exit(1)

