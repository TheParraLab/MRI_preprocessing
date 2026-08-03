import os
import queue
import random
import argparse
import glob
import pickle
import subprocess
import threading
import signal

from multiprocessing import Manager, cpu_count
from toolbox import ProgressBar, get_logger, run_function

BASE_PATH = '/FL_system'

# Define command line arguments
parser = argparse.ArgumentParser(
    description='Align scans to the first post scan')
parser.add_argument(
    '--load_dir', type=str, default=f'{BASE_PATH}/data/RAS/',
    help='Directory to load scans from')
parser.add_argument(
    '--save_dir', type=str, default=f'{BASE_PATH}/data/coreg/',
    help='Directory to save aligned scans')
parser.add_argument(
    '--multi', '-m', action='store_true', help='Use multiprocessing')
parser.add_argument(
    '--dir_idx', type=int,
    help='Index of the folder to process from dirs_to_process.txt')
parser.add_argument(
    '--dir_list', type=str, default='dirs_to_process.txt',
    help='Path to the directory list file')
parser.add_argument(
    '--prune', '-p', action='store_true',
    help='Enable the deletion of the original scans once aligned')
parser.add_argument(
    '--test', nargs='?', type=int, const=10,
    help='Run in test mode, randomly sample N directories to process (default: 10)')
args = parser.parse_args()

LOGGER = get_logger('05_alignScans', f'{BASE_PATH}/data/logs/')

# Define necessary directories
LOAD_DIR = args.load_dir
SAVE_DIR = args.save_dir
PARALLEL = args.multi
TEST = args.test is not None
N_TEST = args.test if TEST else 10
PROGRESS = False
PRUNE = args.prune

manager = Manager()
stop_flag = manager.Event()


def _check_stop():
    if stop_flag.is_set():
        raise KeyboardInterrupt('Shutdown requested')


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

    src_files = glob.glob(f'{session_dir}/*_RAS.nii')
    src_files.sort()
    if len(src_files) < 3:
        LOGGER.error(
            f'Not enough scans in {session_dir}. '
            f'Found {len(src_files)} scans. Skipping.')
        return 'Not enough scans'

    out_dir = os.path.join(save_dir, session_dir.split(os.sep)[-1])

    # Skip if every output already exists
    if all(
        os.path.exists(os.path.join(out_dir, os.path.basename(f)))
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

    # Coregister all scans except the reference
    for f in src_files[:1] + src_files[2:]:
        _check_stop()
        dest = os.path.join(out_dir, os.path.basename(f)).replace('.nii', '')
        out_file = f'{dest}.nii'
        if os.path.exists(out_file):
            LOGGER.info(
                f'Skipping (already exists): {os.path.basename(f)}')
            continue
        try:
            subprocess.run(
                ['reg_f3d', '-ref', reference, '-flo', f, '-res', out_file,
                 '-be', '0.1', '-platf', '1'],
                check=True)
            LOGGER.info(f'Coregistered: {os.path.basename(f)}')
        except subprocess.CalledProcessError as e:
            LOGGER.error(
                f'Error during coregistration of '
                f'{os.path.basename(f)}: {e}')
            if os.path.exists(out_file):
                os.remove(out_file)

    # Copy reference scan into output directory unchanged
    reference_dst = os.path.join(out_dir, os.path.basename(reference))
    if not os.path.exists(reference_dst):
        subprocess.run(['cp', reference, reference_dst], check=True)
        LOGGER.info(
            f'Copied reference: {os.path.basename(reference)}')

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


def run_with_progress(
    target, items, parallel=True, P_type='process',
    P_role='compute', save_dir: str = SAVE_DIR
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

    # Module-level wrapper so it is picklable for spawn mode.
    # ``align`` already takes (session_dir, save_dir), so we just need to
    # inject the progress marker without creating a non-picklable closure.
    def _align_wrapper(item):
        result = target(item, save_dir)
        if update_queue is not None:
            update_queue.put((None, 'Processing'))
        return result

    results = run_function(
        LOGGER, _align_wrapper, list(items),
        Parallel=parallel, P_type=P_type, P_role=P_role,
        stop_flag=stop_flag)

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
            text=True, check=True)
        LOGGER.info(f'NiftyReg version: {_res.stdout.strip()}')
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
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

    # ---- Determine list of directories ------------------------
    if args.dir_idx is None:
        dirs = sorted(glob.glob(f'{LOAD_DIR}*'))
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
        run_with_progress(align, dirs, parallel=PARALLEL, save_dir=SAVE_DIR)
    except KeyboardInterrupt:
        LOGGER.info('Interrupted. Completed directories are safe to resume.')
        raise

    # ---- Prune original scans if requested --------------------
    if PRUNE:
        LOGGER.info('Pruning original scans')
        for d in dirs:
            p = d if os.path.isabs(d) else os.path.join(LOAD_DIR, d)
            if os.path.exists(p):
                try:
                    subprocess.run(['rm', '-rf', p], check=True)
                    LOGGER.info(f'Deleted: {p}')
                except Exception as e:
                    LOGGER.error(f'Error deleting directory {p}: {e}')
            else:
                LOGGER.warning(
                    f'Directory {p} does not exist. Skipping deletion.')

    LOGGER.info('Completed alignScans: Step 05')
    LOGGER.info('All files saved to coreg directory')
    LOGGER.info('Exiting alignScans: Step 05')
