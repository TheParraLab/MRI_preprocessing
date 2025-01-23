import os
import pydicom as pyd
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Callable, List, Any
from multiprocessing import Queue, Manager, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import subprocess
import threading
from toolbox import ProgressBar, get_logger
# Global variables for progress bar and lock
Progress = None
manager = Manager()
progress_queue = manager.Queue()
LOGGER = get_logger('06_genInputs', '/FL_system/data/logs/')

LOAD_DIR = '/FL_system/data/coreg/'
SAVE_DIR = '/FL_system/data/inputs/'
DEBUG = 0
TEST = True
N_TEST = 10
PARALLAL = True
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
    LOGGER.debug(f'Generating slopes for session: {SessionID}')
    
    Fils = glob.glob(f'{LOAD_DIR}/{SessionID}/*.nii')
    Fils.sort()
    LOGGER.debug(f'{SessionID} | Files | {Fils} ')
    Data = Data_table[Data_table['SessionID'] == i]
    assert np.min(len(Data), len(Fils)) >= 3, 'Minimim number of samples is not met, unable to determine slopes with <3 scans'
    if len(Data) != len(Fils):
        LOGGER.warning(f'{SessionID} | Different number of files and detected times')
        LOGGER.warning(f'{SessionID} | Analyzing timing spreadsheet to remove non-fat saturated (assumption!)')
        Data = Data[Data['Series_desc'].str.contains('FS', na=False)].reset_index(drop=True)
    assert len(Data) == len(Fils), 'ERROR: different sizes cannot be fixed through Fat saturation'

    Major = Data['Major']
    sorting = np.argsort(Major)
    Times = [Data['TriTime'][ii] for ii in sorting] #Loading Times in ms
    # Converting to seconds
    Times = [t/1000 for t in Times]
    LOGGER.debug(f'{SessionID} | Times | {Times}')
    
    # Load the 01 scan
    img = nib.load(Fils[0])
    data0 = img.get_fdata()
    data0[np.isnan(data0)] = 0
    p95 = float(np.percentile(data0,95))
    LOGGER.debug(f'{SessionID} | 95% | {p95}')

    # Create a new NIfTI image with the same affine, but with the data type, slope, and intercept set explicitly
    new_img = nib.Nifti1Image(data0, img.affine)
    new_img.header['datatype'] = 16
    new_img.header['scl_slope'] = 1
    new_img.header['bitpix'] = 32
    new_img.header['cal_max'] = 0
    new_img.header['cal_min'] = 0

    # Building time matrix same shape as loaded data
    T = np.zeros_like(data0)
    T = np.expand_dims(T, axis=-1)
    T = np.repeat(T, len(Times), axis=-1)
    for ii,jj in enumerate(Times):
        T[:,:,:,ii] = jj
    
    # Loading all image data into single matrix
    D = np.zeros_like(data0)
    D = np.expand_dims(D, axis=-1)
    D = np.repeat(D, n, axis=-1)
    for ii,jj in enumerate(Fils):
        img = nib.load(jj)
        data0 = img.get_fdata()
        data0[np.isnan(data0)] = 0
        D[:,:,:,ii] = data0
    D[np.isnan(D)] = 0

    ###################################
    # Calculating slope 1 (enhancement)
    LOGGER.debug(f'{SessionID} | Starting slope 1 calculation')
    Tmean = np.repeat(np.expand_dims(np.mean(T[:,:,:,0:2], axis=3), axis=-1), 2, axis=-1)
    Dmean = np.repeat(np.expand_dims(np.mean(D[:,:,:,0:2], axis=3), axis=-1), 2, axis=-1)
    slope1 = np.divide(
        np.sum((T[:,:,:,0:2] - Tmean) * (D[:,:,:,0:2] - Dmean), axis=3),
        np.sum(np.square((T[:,:,:,0:2] - Tmean)), axis=3)
    ).astype(np.float32)
    slope1 = slope1 / p95

    nib.save(nib.Nifti1Image(slope1.astype('float32'), img.affine), SAVE_DIR + f'/{SessionID}/slope1.nii')
    LOGGER.debug(f'{SessionID} | Saved slope 1')

    ###################################
    # Calculating slope 2 (washout)
    LOGGER.debug(f'{SessionID} | Starting slope 2 calculation')
    Tmean = np.repeat(np.expand_dims(np.mean(T[:,:,:,1:], axis=3), axis=-1), n-1, axis=-1)
    Dmean = np.repeat(np.expand_dims(np.mean(D[:,:,:,1:], axis=3), axis=-1), n-1, axis=-1)
    slope2 = np.divide(
        np.sum((T[:,:,:,1:] - Tmean) * (D[:,:,:,1:] - Dmean), axis=3),
        np.sum(np.square((T[:,:,:,1:] - Tmean)), axis=3)
    ).astype(np.float32)
    slope2 = slope2 / p95

    nib.save(nib.Nifti1Image(slope2.astype('float32'), img.affine), SAVE_DIR + f'/{SessionID}/slope2.nii')
    LOGGER.debug(f'{SessionID} | Saved slope 2')

    ###################################
    # Creating post-contrast image
    LOGGER.debug(f'{SessionID} | Starting post contrast scan')
    img = nib.load(Fils[1])
    data1 = img.get_fdata()
    data1[np.isnan(data1)] = 0
    post = data1/p95

    nib.save(nib.Nifti1Image(post.astype('float32'), img.affine), SAVE_DIR + f'/{SessionID}/post.nii')
    LOGGER.debug(f'{SessionID} | Saved post contrast scan')

    ###################################








if __name__ == '__main__':
    try:
        Data_table = pd.read_csv('/FL_system/data/Data_table_timing.csv')
    except:
        LOGGER.error('MISSING CRITICAL FILE | "data_table_timing.csv"')
        exit()
     
    session = np.unique(Data_table['SessionID'])

    Dirs = os.listdir(f'{LOAD_DIR}/')
    N = len(Dirs)
    k = 0
    
    if N != len(session):
        LOGGER.warning('Mismatch number of sessions and input directories')

    # Check if inputs have already been generated
    if os.path.exists(SAVE_DIR):
        print('Inputs already generated')
        print('To reprocess data, please remove /data/inputs')
        exit()
    else:
        # Create directory for saving inputs
        os.mkdir(SAVE_DIR)


    run_with_progress(generate_slopes, session, Parallel=True)
