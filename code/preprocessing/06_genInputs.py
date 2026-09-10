import os
import random
import argparse
import pydicom as pyd
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Callable, List, Any
from multiprocessing import Queue, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import subprocess
import threading
from toolbox import ProgressBar, get_log_dir, get_logger


def _clean_timing(value):
    """Coerce a timing cell (TriTime/AcqTime/ScanDur) to a clean token.

    Bad clinical files can carry a raw bytes blob in a time element; pydicom
    returns it as `bytes` and it then lands in the timing CSV as its repr
    ("b'\\x16\\xba\\xa2L'"). `float()`/`split(':')` on that raises and aborts the
    whole session. Normalising here routes any non-numeric value to 'Unknown'
    so 06's existing fallback path handles it.
    """
    if isinstance(value, float) and value != value:  # NaN
        return 'Unknown'
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode('ascii')
        except UnicodeDecodeError:
            return 'Unknown'
    try:
        float(str(value).strip())
        return value
    except (ValueError, TypeError):
        return 'Unknown'


def _acq_seconds(value):
    """Parse an AcqTime cell into seconds-of-day, or None if unusable.

    Clinical AcqTime comes in two shapes:
      - positional HHMMSS[.ffffff]  e.g. "155906", "161206.3875"
      - colon HH:MM:SS[.ffffff]     e.g. "16:12:06.4"
    Non-ASCII bytes blobs (the clinical bytes repr) and 'Unknown' return None.

    A value only in the plausible time-of-day range (< 23:59:59) is read
    positionally; larger numbers (e.g. a raw-ms leak) are returned as-is so
    the caller can compare deltas.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode('ascii')
        except UnicodeDecodeError:
            return None
    v = str(value).strip()
    if not v or v.lower() == 'unknown':
        return None
    if ':' in v:
        parts = v.split(':')
        if len(parts) != 3:
            return None
        try:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return None
    try:
        n = float(v)
    except ValueError:
        return None
    iv = int(n)
    if 0 <= iv <= 235959:  # plausible HHMMSS time-of-day
        s = str(iv).zfill(6)
        return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:])
    return n


# Global variables for progress bar and lock
Progress = None
# Centralised log directory — resolves to /deployment/logs inside containers
# (bound mount) or <repo>/logs for local/manual runs. See toolbox.get_log_dir().
LOG_DIR = get_log_dir()

LOGGER = get_logger('06_genInputs', LOG_DIR)

# argparse configuration
parser = argparse.ArgumentParser(description='Generate model inputs from coregistered scans')
parser.add_argument('--load_dir', type=str, default='/FL_system/data/coreg/', help='Directory to load scans from')
parser.add_argument('--save_dir', type=str, default='/FL_system/data/inputs/', help='Directory to save model inputs')
parser.add_argument('--test', nargs='?', type=int, const=40, help='Run in test mode, randomly sample N sessions (default: 40)')
parser.add_argument('--parallel', action='store_true', help='Enable multiprocessing')
parser.add_argument(
    '--ids_file', type=str, default=None,
    help='CSV/txt file containing one ID per line. If provided, only process sessions whose name appears in this file.'
)
args = parser.parse_args()

LOAD_DIR = args.load_dir
SAVE_DIR = args.save_dir
DEBUG = 0
TEST = args.test is not None
N_TEST = args.test if TEST else 40
PARALLEL = args.parallel
PROGRESS = False
# This script is for generating the numpy files utilized for model training
# Performs the calculation of the slope 1 (enhancement) for each scan
# Performs the calculation of the slope 2 (washout) for each scan
# Normalizes samples by dividing by 95th percentile of T1_01_01
def progress_wrapper(item, target, progress_queue, *args, **kwargs):
    result = target(item, *args, **kwargs)
    progress_queue.put((None, f'Processing'))
    return result

def run_with_progress(target: Callable[..., Any], items: List[Any], Parallel: bool=True, *args, **kwargs) -> List[Any]:
    """Run a function with a progress bar"""
    # Initialize using a manager to allow for shared progress queue
    progress_queue = Queue()
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
    if Parallel:
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(target, item, *args, **kwargs) for item in items]
            results = [future.result() for future in futures]
    else:
        results = [target(item) for item in items]

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
    while True:
        item = queue.get()
        if item is None:
            break
        index, status = item
        progress_bar.update(index, status)

        queue.task_done()

def generate_slopes(SessionID):
    # Should generate 2 slopes
    # Slope 1 - between 00 and 01
    # Slope 2 - between 01 and 0X
    if os.path.exists(SAVE_DIR + f'/{SessionID}'):
        LOGGER.warning(f'{SessionID} | Directory already exists')
        # Check for files in the directory
        if len(os.listdir(SAVE_DIR + f'/{SessionID}')) < 3:
            LOGGER.warning(f'{SessionID} | Directory does not have necessary files, reprocessing')
            os.rmdir(SAVE_DIR + f'/{SessionID}')
        else:
            LOGGER.debug(f'{SessionID} | Directory exists and has necessary files, skipping')
            return
    else:
        LOGGER.debug(f'{SessionID} | Creating saving directory for inputs')
    os.mkdir(SAVE_DIR + f'/{SessionID}')

    LOGGER.debug(f'Generating slopes for session: {SessionID}')
    
    Fils = glob.glob(f'{LOAD_DIR}/{SessionID}/*.nii')
    Fils.sort()
    LOGGER.debug(f'{SessionID} | Files | {Fils} ')
    Data = Data_table[Data_table['SessionID'] == SessionID]
    if np.min([len(Data), len(Fils)]) < 3:
        LOGGER.warning(f'{SessionID} | Skipping session due to insufficient number of scans (<3)')
        return
    
    if len(Data) != len(Fils):
        LOGGER.warning(f'{SessionID} | Different number of files and detected times')
        LOGGER.warning(f'{SessionID} | Analyzing timing spreadsheet to remove non-fat saturated (assumption!)')
        Data = Data[Data['Series_desc'].str.contains('FS', na=False)].reset_index(drop=True)
    if not len(Data) == len(Fils):
        LOGGER.error(f'{SessionID} | ERROR: different sizes cannot be fixed through Fat saturation')
        return
    Major = Data['Major'] # Major is the order of the scans
    sorting = np.argsort(Major) # Sorting the scans
    #LOGGER.debug(f'{SessionID} | Sorting values| {sorting.values}')
    #LOGGER.debug(f'{SessionID} | Trigger Time | {Data["TriTime"].values}')
    #LOGGER.debug(f'{SessionID} | Scan Duration | {Data["ScanDur"].values}')
    
    # Build a numeric time vector (seconds, relative) for every scan, index 0 = pre.
    # Two independent clocks are available per scan:
    #   TriTime  — Siemens raw-ms trigger time (0, 87259, 174108, ...) for posts;
    #              the pre often reports 'Unknown' and ScanDur is a bytes blob.
    #   AcqTime  — HHMMSS time-of-day, present (and ordered) even when TriTime is not.
    # Strategy:
    #   1. Prefer TriTime for the posts when they are all numeric (the true
    #      trigger times). Resolve the pre via TriTime, else ScanDur, else the
    #      AcqTime delta to the first post.
    #   2. If the posts' TriTimes are missing OR all identical (no spread usable
    #      for the regression), fall back to AcqTime for the whole session.
    #   3. If neither clock yields a usable, ordered set, skip the session.
    raw_tri = [Data['TriTime'].iloc[ii] for ii in sorting]
    raw_dur = [Data['ScanDur'].iloc[ii] for ii in sorting]
    raw_acq = [Data['AcqTime'].iloc[ii] for ii in sorting]
    n = len(raw_tri)

    tri_s = []
    for t in raw_tri:
        c = _clean_timing(t)
        if c == 'Unknown':
            tri_s.append(None)
        else:
            try:
                tri_s.append(float(c) / 1000.0)  # ms -> s
            except (ValueError, TypeError):
                tri_s.append(None)
    acq_s = [_acq_seconds(a) for a in raw_acq]
    dur_clean = [_clean_timing(d) for d in raw_dur]

    tri_posts_usable = all(tri_s[i] is not None for i in range(1, n)) and n > 2 \
        and len(set(round(x, 6) for x in tri_s[1:])) > 1
    acq_usable = all(acq_s[i] is not None for i in range(n))

    Times = None
    if tri_posts_usable:
        Times = list(tri_s)
        if Times[0] is None:
            if dur_clean[0] != 'Unknown':
                try:
                    Times[0] = float(Times[1]) - float(dur_clean[0]) / 1000.0
                except (ValueError, TypeError):
                    Times[0] = None
        if Times[0] is None and (acq_s[0] is not None and acq_s[1] is not None):
            Times[0] = float(acq_s[0]) - float(acq_s[1])
        if any(Times[i] is None for i in range(n)):
            Times = None
        else:
            LOGGER.debug(f'{SessionID} | times from TriTime (pre resolved) | {Times}')
    elif acq_usable:
        Times = [float(acq_s[i] - acq_s[0]) for i in range(n)]
        LOGGER.info(f'{SessionID} | TriTime not usable for posts; using AcqTime deltas | {Times}')
    else:
        LOGGER.error(
            f'{SessionID} | not enough timing info to calculate slopes '
            f'(TriTime posts={tri_s[1:]}, AcqTime posts={acq_s[1:]}, ScanDur[0]={dur_clean[0]})'
        )
        return

    LOGGER.debug(f'{SessionID} | Times | {Times}')
    
    # Load the 01 scan
    img = nib.load(Fils[0])
    data0 = img.get_fdata()
    data0[np.isnan(data0)] = 0
    p95 = float(np.percentile(data0,95))
    LOGGER.debug(f'{SessionID} | 95% | {p95}')

    header = img.header.copy()
    header['datatype'] = 16 # 32-bit float
    header['scl_slope'] = 1
    header['bitpix'] = 32
    header['cal_max'] = 0
    header['cal_min'] = 0
    
    # Create a new NIfTI image with the same affine, but with the data type, slope, and intercept set explicitly
    #new_img = nib.Nifti1Image(data0, img.affine)
    #new_img.header['datatype'] = 16
    #new_img.header['scl_slope'] = 1
    #new_img.header['bitpix'] = 32
    #new_img.header['cal_max'] = 0
    #new_img.header['cal_min'] = 0

    # Building time matrix same shape as loaded data
    T = np.zeros_like(data0, dtype=np.float32)
    T = np.expand_dims(T, axis=-1)
    T = np.repeat(T, len(Times), axis=-1)
    for ii,jj in enumerate(Times):
        T[:,:,:,ii] = jj
    
    # Loading all image data into single matrix
    D = np.zeros_like(data0, dtype=np.float32)
    D = np.expand_dims(D, axis=-1)
    D = np.repeat(D, len(Times), axis=-1)
    for ii,jj in enumerate(Fils):
        img = nib.load(jj)
        data0 = img.get_fdata().astype(np.float32)
        data0[np.isnan(data0)] = 0
        D[:,:,:,ii] = data0
    D[np.isnan(D)] = 0

    ###################################
    # Calculating slope 1 (enhancement)
    LOGGER.debug(f'{SessionID} | Starting slope 1 calculation')
    Tmean = np.repeat(np.expand_dims(np.mean(T[:,:,:,0:2], axis=3), axis=-1), 2, axis=-1).astype(np.float32)
    Dmean = np.repeat(np.expand_dims(np.mean(D[:,:,:,0:2], axis=3), axis=-1), 2, axis=-1).astype(np.float32)
    denom1 = np.sum(np.square((T[:,:,:,0:2] - Tmean)), axis=3)
    slope1 = np.divide(
        np.sum((T[:,:,:,0:2] - Tmean) * (D[:,:,:,0:2] - Dmean), axis=3),
        denom1,
        out=np.zeros_like(denom1, dtype=np.float32),
        where=denom1 != 0
    ).astype(np.float32)
    slope1 = slope1 / p95

    header['glmax'] = np.max(slope1)
    header['glmin'] = np.min(slope1)
    header['descrip'] = 'pre slp img'

    LOGGER.debug(f'{SessionID} | Slope 1 shape: {slope1.shape}')
    LOGGER.debug(f'{SessionID} | Header shape: {header.get_data_shape()}')

    nib.save(nib.Nifti1Image(slope1.astype('float32'), img.affine, header), SAVE_DIR + f'/{SessionID}/slope1.nii')
    LOGGER.debug(f'{SessionID} | Saved slope 1')

    ###################################
    # Calculating slope 2 (washout)
    LOGGER.debug(f'{SessionID} | Starting slope 2 calculation')
    Tmean = np.repeat(np.expand_dims(np.mean(T[:,:,:,1:], axis=3), axis=-1), len(Times)-1, axis=-1).astype(np.float32)
    Dmean = np.repeat(np.expand_dims(np.mean(D[:,:,:,1:], axis=3), axis=-1), len(Times)-1, axis=-1).astype(np.float32)
    denom2 = np.sum(np.square((T[:,:,:,1:] - Tmean)), axis=3)
    slope2 = np.divide(
        np.sum((T[:,:,:,1:] - Tmean) * (D[:,:,:,1:] - Dmean), axis=3),
        denom2,
        out=np.zeros_like(denom2, dtype=np.float32),
        where=denom2 != 0
    ).astype(np.float32)
    slope2 = slope2 / p95

    header['glmax'] = np.max(slope2)
    header['glmin'] = np.min(slope2)
    header['descrip'] = 'post slp img'

    LOGGER.debug(f'{SessionID} | Slope 2 shape: {slope2.shape}')
    LOGGER.debug(f'{SessionID} | Header shape: {header.get_data_shape()}')

    nib.save(nib.Nifti1Image(slope2.astype('float32'), img.affine, header), SAVE_DIR + f'/{SessionID}/slope2.nii')
    LOGGER.debug(f'{SessionID} | Saved slope 2')

    ###################################
    # Creating post-contrast image
    LOGGER.debug(f'{SessionID} | Starting post contrast scan')
    img = nib.load(Fils[1])
    data1 = img.get_fdata().astype(np.float32)
    data1[np.isnan(data1)] = 0
    post = data1/p95

    LOGGER.debug(f'{SessionID} | Post contrast shape: {post.shape}')
    LOGGER.debug(f'{SessionID} | Header shape: {header.get_data_shape()}')

    nib.save(nib.Nifti1Image(post.astype('float32'), img.affine, img.header), SAVE_DIR + f'/{SessionID}/post.nii')
    LOGGER.debug(f'{SessionID} | Saved post contrast scan')

    ###################################



if __name__ == '__main__':
    try:
        Data_table = pd.read_csv('/FL_system/data/Data_table_timing.csv')
    except:
        LOGGER.error('MISSING CRITICAL FILE | "data_table_timing.csv"')
        exit()
     
    # Load IDs to filter by (if --ids_file provided)
    ids_to_process = None
    if args.ids_file is not None:
        with open(args.ids_file, 'r') as f:
            ids_to_process = set(line.strip() for line in f if line.strip())
        LOGGER.info(f'Loaded {len(ids_to_process)} IDs from {args.ids_file}')

    session = np.unique(Data_table['SessionID'])
    Dirs = os.listdir(f'{LOAD_DIR}/')
    if ids_to_process is not None:
        Dirs = [d for d in Dirs if d in ids_to_process]
        LOGGER.info(f'Filtered to {len(Dirs)} directories matching IDs')
    if TEST:
        session = random.sample(list(session), min(N_TEST, len(session)))
        Dirs = random.sample(Dirs, min(N_TEST, len(Dirs)))
    session = Dirs
    N = len(Dirs)
    k = 0
    
    if N != len(session):
        LOGGER.warning(f'Mismatch number of sessions and input directories | {len(session)} {N}')

    # Check if inputs have already been generated
    if os.path.exists(SAVE_DIR):
        print('Inputs already generated')
        #print('To reprocess data, please remove /data/inputs')
        #exit()
    else:
        # Create directory for saving inputs
        os.mkdir(SAVE_DIR)


    run_with_progress(generate_slopes, session, Parallel=False)
