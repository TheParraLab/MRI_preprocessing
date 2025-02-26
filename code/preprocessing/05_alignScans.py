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
LOGGER = get_logger('05_alignScans', '/FL_system/data/logs/')

# Define necessary directories
LOAD_DIR = '/FL_system/data/RAS/'
SAVE_DIR = '/FL_system/data/coreg/'
DEBUG = 0
TEST = True
N_TEST = 40
PARALLAL = True
PROGRESS = False

#### Preprocessing | Step 5: Align Scans ####
# This script aims to coregister all session-specific scans to the 01_01 scan
# 
# This 
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
        with ProcessPoolExecutor(max_workers=cpu_count()//2) as executor:
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

def align(Dir):
    # This function coregisters all scans in the input directory to the 01_01 scan
    # It saves the coregistered scans in the output directory
    Fils = glob.glob(f'{Dir}/*_RAS.nii')
    Fils.sort()
    LOGGER.info(f'Processing {Dir}')
    if not os.path.exists(f'{SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}'):
        os.mkdir(f'{SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}')
        LOGGER.debug(f'Created directory: {SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}')
    # Coregister all subsequent scans
    for ii in Fils[2:]:
        # Perform coregistration to 01_01
        dest = f'{SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}{os.sep}{ii.split(os.sep)[-1]}'
        dest = dest.replace('.nii','')
        print(dest)
        aff_save = (os.sep).join(dest.split(os.sep)[:-1])
        #os.system(f'reg_aladin -ref {Fils[1]} -flo {ii} -aff {dest}_aff.txt')
        #os.system(f'reg_f3d -ref {Fils[1]} -flo {ii} -res {dest}.nii -aff {dest}_aff.txt -be 0.1')
        #os.system(f'rm {dest}_aff.txt')
        
        os.system(f'reg_f3d -ref {Fils[1]} -flo {ii} -res {dest}.nii -be 0.1')

        LOGGER.info(f'Coregistered: {ii}')
    # Coregister the first scan
    dest = f'{SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}{os.sep}{Fils[0].split(os.sep)[-1]}'
    dest = dest.replace('.nii','')
    #os.system(f'reg_aladin -ref {Fils[1]} -flo {Fils[0]} -aff {dest}_aff.txt')
    #os.system(f'reg_f3d -ref {Fils[1]} -flo {Fils[0]} -res {dest}.nii -aff {dest}_aff.txt -be 0.1')
    #os.system(f'rm {dest}_aff.txt')
    
    os.system(f'reg_f3d -ref {Fils[1]} -flo {Fils[0]} -res {dest}.nii -be 0.1')
    
    LOGGER.info(f'Coregistered: {Fils[0]}')

    # Copy reference to coregistered samples
    os.system(f'cp {Fils[1]} {SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}{os.sep}{Fils[1].split(os.sep)[-1]}')
    LOGGER.info(f'Copied: {Fils[1]}')

    return 'completed'

if __name__ == '__main__':
    LOGGER.info('Starting alignScans: Step 05')
    LOGGER.info(f'LOAD_DIR: {LOAD_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLAL: {PARALLAL}')
    if TEST:
        LOGGER.info(f'Running in test mode: {TEST}')
        LOGGER.info(f'Number of test sessions: {N_TEST}')

    if os.path.exists(SAVE_DIR):
        if len(os.listdir(SAVE_DIR)) > 0:
            LOGGER.error('Coregistration already complete.')
            LOGGER.error('To reprocess data, please remove /FL_system/data/coreg/ or remove its contents')
            exit()
        else:
            LOGGER.warning('Coregistration directory is empty. Proceeding with coregistration...')
    else:
        os.mkdir(SAVE_DIR)

    # Load the data
    Dirs = glob.glob('/FL_system/data/RAS/*')
    if TEST:
        Dirs = Dirs[:N_TEST]
    
    # Run the coregistration
    run_with_progress(align, Dirs, Parallel=PARALLAL)
