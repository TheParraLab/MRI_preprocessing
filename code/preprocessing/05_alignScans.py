import os
import queue
import random
import shutil
import argparse
import glob
import pickle
import subprocess
import threading
import signal

from functools import partial
from multiprocessing import cpu_count, Event
from toolbox import (
    ProgressBar, get_log_dir, get_logger, run_function, resolve_dir,
    nifti_stem, is_nifti_file, glob_nifti,
)

_GPU_CHECK_INTERVAL = 10
_gpu_calls_since_check = 0

# Define command line arguments
parser = argparse.ArgumentParser(
    description='Align scans to the first post scan')
parser.add_argument(
    '--load_dir', type=str, default=None,
    help='Directory to load scans from (default: $RAS_DIR or /FL_system/data/RAS/)')
parser.add_argument(
    '--save_dir', type=str, default=None,
    help='Directory to save aligned scans (default: $COREG_DIR or /FL_system/data/coreg/)')
parser.add_argument(
    '--multi', '-m', action='store_true', help='Use multiprocessing')
parser.add_argument(
    '--dir_idx', type=int,
    help='Index of the folder to process from dirs_to_process.txt')
parser.add_argument(
    '--dir_list', type=str, default='dirs_to_process.txt',
    help='Path to the directory list file')
parser.add_argument(
    '--prune', action='store_true',
    help='Enable the deletion of the original scans once aligned')
parser.add_argument(
    '--test', nargs='?', type=int, const=10,
    help='Run in test mode, randomly sample N directories to process (default: 10)')
parser.add_argument(
    '--ids_file', type=str, default=None,
    help='CSV/txt file containing one ID per line. If provided, only process directories whose name appears in this file.'
)
parser.add_argument(
    '--cpus', type=int, default=0,
    help='Cap on parallel worker processes for --multi: exactly N reg_f3d jobs '
         'run concurrently (0 = auto: min(32, 2×(cores-1)))'
)
args = parser.parse_args()

args.load_dir = resolve_dir(args.load_dir, 'RAS_DIR', '/FL_system/data/RAS/')
args.save_dir = resolve_dir(args.save_dir, 'COREG_DIR', '/FL_system/data/coreg/')

# Centralised log directory — resolves to /deployment/logs inside containers
# (bound mount) or <repo>/logs for local/manual runs. See toolbox.get_log_dir().
LOG_DIR = get_log_dir()

LOGGER = get_logger('05_alignScans', LOG_DIR)

# Define necessary directories
LOAD_DIR = args.load_dir
SAVE_DIR = args.save_dir
PARALLEL = args.multi
TEST = args.test is not None
N_TEST = args.test if TEST else 10
PROGRESS = False
PRUNE = args.prune

stop_flag = Event()


def _check_stop():
    if stop_flag.is_set():
        raise KeyboardInterrupt('Shutdown requested')


def _check_gpu_health():
    """Verify GPU is still accessible. Raises RuntimeError if not."""
    try:
        _res = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError('nvidia-smi timed out')
    if _res.returncode != 0 or not _res.stdout.strip():
        raise RuntimeError(
            f'GPU health check failed (exit {_res.returncode}): '
            f'{_res.stderr.strip()}')


def align(session_dir: str, save_dir: str):
    """Coregister all scans in *session_dir* to the first post reference scan.

    Module-level so that ProcessPoolExecutor can pickle it and ship
    it to child processes via the ``spawn`` start method."""

    _check_stop()
    assert isinstance(session_dir, str)
    LOGGER.info(session_dir.split(os.sep)[-1])
    if session_dir.endswith(os.sep):
        LOGGER.warning('Directory has trailing slash. Removing it.')
        session_dir = session_dir[:-1]

    src_files = sorted(glob.glob(f'{session_dir}/*_RAS.nii') + glob.glob(f'{session_dir}/*_RAS.nii.gz'),
                       key=lambda p: nifti_stem(p))
    if len(src_files) < 3:
        LOGGER.error(
            f'Not enough scans in {session_dir}. '
            f'Found {len(src_files)} scans. Skipping.')
        return 'Not enough scans'

    out_dir = os.path.join(save_dir, session_dir.split(os.sep)[-1])

    # Skip if every output already exists (output is <stem>_RAS.nii.gz)
    if all(
        os.path.exists(os.path.join(out_dir, f'{nifti_stem(f)}_RAS.nii.gz'))
        for f in src_files
    ):
        LOGGER.info(f'All files already exist, skipping: {session_dir}')
        return 'already done'

    LOGGER.info(f'Processing {session_dir}')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        LOGGER.debug(f'Created directory: {out_dir}')

    reference = src_files[1]
    LOGGER.debug(f'Using {reference} as reference for coregistration')

    session_failed = False
    for f in src_files[:1] + src_files[2:]:
        _check_stop()
        out_stem  = nifti_stem(f)                    # already strips _RAS
        out_name  = f'{out_stem}_RAS.nii.gz'        # emit .nii.gz
        out_file  = os.path.join(out_dir, out_name)
        if os.path.exists(out_file):
            LOGGER.info(
                f'Skipping (already exists): {os.path.basename(f)}')
            continue
        try:
            subprocess.run(
                ['reg_f3d', '-ref', reference, '-flo', f, '-res', out_file,
                 '-be', '0.1', '-platf', '1'],
                check=True, timeout=1800)
            LOGGER.info(f'Coregistered: {os.path.basename(f)}')
        except subprocess.CalledProcessError as e:
            LOGGER.error(
                f'Error during coregistration of '
                f'{os.path.basename(f)}: {e}')
            if os.path.exists(out_file):
                os.remove(out_file)
            session_failed = True
            break
        except subprocess.TimeoutExpired:
            LOGGER.error(
                f'Coregistration of {os.path.basename(f)} timed out '
                f'(exceeded 1800s). Treating session as failed.')
            if os.path.exists(out_file):
                os.remove(out_file)
            session_failed = True
            break

        global _gpu_calls_since_check
        _gpu_calls_since_check += 1
        if _gpu_calls_since_check >= _GPU_CHECK_INTERVAL:
            try:
                _check_gpu_health()
                _gpu_calls_since_check = 0
                LOGGER.info('GPU health check passed at scan '
                            f'{os.path.basename(f)}')
            except RuntimeError as e:
                LOGGER.critical(
                    f'GPU health check failed in session '
                    f'{session_dir.split(os.sep)[-1]}: {e}. Purging and '
                    f'aborting.')
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                    LOGGER.info(f'Deleted partial output: {out_dir}')
                raise

    if session_failed:
        LOGGER.error(
            f'Purging incomplete session output directory: {out_dir}')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            LOGGER.info(f'Deleted: {out_dir}')
        return 'failed'

    # reference name may already be .nii or .nii.gz; emit as .nii.gz so
    # downstream steps see a consistent extension.
    reference_out_stem = nifti_stem(reference)
    reference_dst = os.path.join(out_dir, f'{reference_out_stem}_RAS.nii.gz')
    if not os.path.exists(reference_dst):
        subprocess.run(['cp', reference, reference_dst], check=True, timeout=600)
        LOGGER.info(f'Copied reference: {reference_out_stem}_RAS.nii.gz')
    else:
        LOGGER.info(f'REFERENCE ALREADY PRESENT: {reference_dst}')

    return 'completed'


def _progress_updater(update_queue, progress):
    """Daemon thread that pulls markers from *update_queue* and updates
    progress bar."""
    while True:
        item = update_queue.get()
        if item is None:
            break
        try:
            progress.update(item[0], item[1])
        except Exception:
            pass
        finally:
            update_queue.task_done()


def _align_wrapper(item, target, save_dir, update_queue=None):
    """Module-level progress wrapper, picklable under any start method.

    ``concurrent.futures`` pickles the submitted callable and its arguments
    into the worker call queue, so the old nested closure raised
    ``AttributeError: Can't pickle local object`` and failed every item in
    process mode.  The wrapper is module-level and is bound at the
    call site via ``functools.partial``, which is itself picklable."""
    result = target(item, save_dir)
    if update_queue is not None:
        update_queue.put((None, 'Processing'))
    return result


def run_with_progress(
    target, items, parallel=True, P_type='process',
    P_role='compute', save_dir: str = SAVE_DIR, n_workers: int = 0
):
    """Run *target* over *items* with an optional progress bar.

    Wraps ``run_function`` to inject a per-item marker into a shared queue
    that a background thread feeds to a ``ProgressBar``."""

    n = len(list(items))
    update_queue = None
    updater_thread = None

    if PROGRESS:
        global _progress_bar
        _progress_bar = ProgressBar(n)
        update_queue = queue.Queue()
        updater_thread = threading.Thread(
            target=_progress_updater, args=(update_queue, _progress_bar),
            daemon=True)
        updater_thread.start()

    results = run_function(
        LOGGER,
        partial(_align_wrapper, target=target, save_dir=save_dir,
                update_queue=update_queue),
        list(items),
        Parallel=parallel, P_type=P_type, P_role=P_role,
        stop_flag=stop_flag, N_WORKERS=n_workers)

    if PROGRESS:
        if update_queue is not None:
            update_queue.put(None)
        if updater_thread is not None:
            updater_thread.join(timeout=5)
        print()

    if results and isinstance(results[0], tuple):
        return list(zip(*results))
    return results


if __name__ == '__main__':
    # ---- NiftyReg version check (parent only) ------------------
    try:
        _res = subprocess.run(
            ['reg_f3d', '--version'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=True, timeout=30)
        LOGGER.info(f'NiftyReg version: {_res.stdout.strip()}')
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        LOGGER.error(f'Error checking NiftyReg version: {e}')

    # ---- Signal handler for graceful shutdown -----------------
    def _sigint_handler(signum, frame):
        LOGGER.info('[SIGINT] Keyboard interrupt received. In-flight sessions will complete, queued ones cancelled...')
        raise KeyboardInterrupt('Interrupted')
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    LOGGER.info('Starting alignScans: Step 05')
    LOGGER.info(f'LOAD_DIR: {LOAD_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if PARALLEL:
        LOGGER.info('Running in parallel mode')
    if PRUNE:
        LOGGER.warning(f'Pruning enabled: {PRUNE}')

    if not os.path.exists(SAVE_DIR):
        try:
            os.mkdir(SAVE_DIR)
            LOGGER.info(f'Created directory: {SAVE_DIR}')
        except Exception as e:
            LOGGER.error(f'Error creating directory {SAVE_DIR}: {e}')

    # Load IDs to filter by (if --ids_file provided)
    ids_to_process = None
    if args.ids_file is not None:
        with open(args.ids_file, 'r') as f:
            ids_to_process = set(line.strip() for line in f if line.strip())
        LOGGER.info(f'Loaded {len(ids_to_process)} IDs from {args.ids_file}')

    # ---- Determine list of directories ------------------------
    if args.dir_idx is None:
        dirs = sorted(glob.glob(f'{LOAD_DIR}*'))
        if ids_to_process is not None:
            dirs = [d for d in dirs if os.path.basename(d) in ids_to_process]
            LOGGER.info(f'Filtered to {len(dirs)} directories matching IDs')
        if TEST:
            dirs = random.sample(dirs, min(N_TEST, len(dirs)))
        LOGGER.info(f'Processing {len(dirs)} directories')
    else:
        assert os.path.exists(args.dir_list), (
            f'Directory list file {args.dir_list} does not exist')
        with open(args.dir_list, 'rb') as f:
            all_dirs = pickle.load(f)
        dir_single = all_dirs[args.dir_idx]
        if isinstance(dir_single, str):
            LOGGER.debug(f'Converting Dir to list: {dir_single}')
            dir_single = [dir_single]
        LOGGER.info(
            f'Processing index {args.dir_idx} of '
            f'{len(all_dirs)}: {dir_single}')
        dirs = dir_single

    # ---- Run coregistration ----------------------------------
    try:
        results = run_with_progress(
            align, dirs, parallel=PARALLEL, save_dir=SAVE_DIR,
            n_workers=args.cpus) or []
    except KeyboardInterrupt:
        LOGGER.info('Interrupted. Completed directories are safe to resume.')
        raise

    n_aligned = sum(1 for r in results if r == 'completed')
    n_done = sum(1 for r in results if r == 'already done')
    n_insufficient = sum(1 for r in results if r == 'Not enough scans')
    n_failed = sum(1 for r in results if r in ('failed', None))
    LOGGER.info(
        f'Run summary: {n_aligned} aligned, {n_done} already done, '
        f'{n_insufficient} insufficient scans, {n_failed} failed '
        f'of {len(dirs)} sessions')

    # ---- Prune original scans if requested --------------------
    if PRUNE:
        LOGGER.info('Pruning original scans')
        for d in dirs:
            p = d if os.path.isabs(d) else os.path.join(LOAD_DIR, d)
            if os.path.exists(p):
                try:
                    subprocess.run(['rm', '-rf', p], check=True, timeout=120)
                    LOGGER.info(f'Deleted: {p}')
                except Exception as e:
                    LOGGER.error(f'Error deleting directory {p}: {e}')
            else:
                LOGGER.warning(
                    f'Directory {p} does not exist. Skipping deletion.')

    LOGGER.info('Completed alignScans: Step 05')
    if n_failed:
        LOGGER.warning(f'{n_failed} sessions incomplete - re-run step 05 to process them')
    else:
        LOGGER.info('All files saved to coreg directory')
    LOGGER.info('Exiting alignScans: Step 05')
