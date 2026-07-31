import os
import queue
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
args = parser.parse_args()

LOGGER = get_logger('05_alignScans', f'{BASE_PATH}/data/logs/')

# Log niftyreg version
try:
    result = subprocess.run(
        ['reg_f3d', '--version'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=True)
    LOGGER.info(f'NiftyReg version: {result.stdout.strip()}')
except subprocess.CalledProcessError as e:
    LOGGER.error(f'Error checking NiftyReg version: {e}')

# Define necessary directories
LOAD_DIR = args.load_dir
SAVE_DIR = args.save_dir
TEST = False
N_TEST = 40
PARALLEL = args.multi
PROGRESS = False
PRUNE = args.prune

# Global progress bar reference (set lazily)
_progress_bar = None


def align(session_dir: str, save_dir: str):
    """Coregister all scans in *session_dir* to the 01_01 reference scan.

    Suggested as module-level so that ProcessPoolExecutor can pickle it
    and ship it to child processes via the ``spawn`` start method."""
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
    if all(os.path.exists(os.path.join(out_dir, os.path.basename(f)))
           for f in src_files):
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
        dest = os.path.join(out_dir, os.path.basename(f)).replace('.nii', '')
        out_file = f'{dest}.nii'
        if os.path.exists(out_file):
            LOGGER.info(f'Skipping (already exists): {os.path.basename(f)}')
            continue
        try:
            subprocess.run(
                ['reg_f3d', '-ref', reference, '-flo', f, '-res', out_file,
                 '-be', '0.1', '-platf', '1'],
                check=True)
            LOGGER.info(f'Coregistered: {os.path.basename(f)}')
        except subprocess.CalledProcessError as e:
            LOGGER.error(
                f'Error during coregistration of {os.path.basename(f)}: {e}')
            if os.path.exists(out_file):
                os.remove(out_file)

    # Copy reference scan into output directory unchanged
    reference_dst = os.path.join(out_dir, os.path.basename(reference))
    if not os.path.exists(reference_dst):
        subprocess.run(['cp', reference, reference_dst], check=True)
        LOGGER.info(f'Copied reference: {os.path.basename(reference)}')

    return 'completed'


def _progress_updater(queue, progress):
    """Daemon thread that pulls markers from *queue* and updates progress bar."""
    while True:
        item = queue.get()
        if item is None:
            break
        try:
            progress.update(item[0], item[1])
        except Exception:
            pass
        finally:
            queue.task_done()


def run_with_progress(target, items, parallel=True, P_type='process',
                      P_role='compute', save_dir=SAVE_DIR):
    """Run *target* over *items* with an optional progress bar.

    Wraps ``run_function`` to inject a per-item marker into a shared queue
    that a background thread feeds to a ``ProgressBar``."""

    n = len(items)
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

    def _wrapper(item):
        result = target(item, save_dir)
        if update_queue is not None:
            update_queue.put((None, 'Processing'))
        return result

    results = run_function(
        LOGGER, _wrapper, list(items),
        Parallel=parallel, P_type=P_type, P_role=P_role)

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
    # ---- Signal handler for graceful shutdown ---------------
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    LOGGER.info('Starting alignScans: Step 05')
    LOGGER.info(f'LOAD_DIR: {LOAD_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if TEST:
        LOGGER.info(f'Running in test mode: {TEST}')
        LOGGER.info(f'Number of test sessions: {N_TEST}')
    if PRUNE:
        LOGGER.warning(f'Pruning enabled: {PRUNE}')

    if not os.path.exists(SAVE_DIR):
        try:
            os.mkdir(SAVE_DIR)
            LOGGER.info(f'Created directory: {SAVE_DIR}')
        except Exception as e:
            LOGGER.error(f'Error creating directory {SAVE_DIR}: {e}')

    # ---- Determine list of directories ---------------------
    if args.dir_idx is None:
        dirs = sorted(glob.glob(f'{LOAD_DIR}*'))
        if TEST:
            dirs = dirs[:N_TEST]
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
            f'Processing index {args.dir_idx} of {len(all_dirs)}: '
            f'{dir_single}')
        dirs = dir_single

    # ---- Run coregistration --------------------------------
    try:
        run_with_progress(align, dirs, parallel=PARALLEL, save_dir=SAVE_DIR)
    except KeyboardInterrupt:
        LOGGER.info('Interrupted. Completed directories are safe to resume.')
        raise

    # ---- Prune original scans if requested -----------------
    if PRUNE:
        LOGGER.info('Pruning original scans')
        for d in dirs:
            p = os.path.join(LOAD_DIR, d) if not os.path.isabs(d) else d
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
