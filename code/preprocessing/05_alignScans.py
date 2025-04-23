import os
import argparse
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
#BASE_PATH = '/FL_system'
BASE_PATH = '/home/nleotta000/Projects/'
# Global variables for progress bar and lock
Progress = None
manager = Manager()
progress_queue = manager.Queue()

# Define command line arguments
parser = argparse.ArgumentParser(description='Align scans to the first post scan')
parser.add_argument('--load_dir', type=str, default=f'{BASE_PATH}/data/RAS/', help='Directory to load scans from')
parser.add_argument('--save_dir', type=str, default=f'{BASE_PATH}/data/coreg/', help='Directory to save aligned scans')
parser.add_argument('--multi', '-m', action='store_true', help='Use multiprocessing')
parser.add_argument('--dir_idx', type=int, help='Index of the folder to process from dirs_to_process.txt')
parser.add_argument('--dir_list', type=str, default='dirs_to_process.txt', help='Path to the directory list file')
args = parser.parse_args()

LOGGER = get_logger('05_alignScans', f'{BASE_PATH}/data/logs/')

# Define necessary directories
LOAD_DIR = args.load_dir
SAVE_DIR = args.save_dir
DEBUG = 0
TEST = False
N_TEST = 40
PARALLAL = args.multi
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
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()//2))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
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

    # Make sure Dir is a string
    assert isinstance(Dir, str), f'Dir should be a string, but got {type(Dir)}'
    LOGGER.info(Dir[-1])
    if Dir[-1] == os.sep:
        LOGGER.warning(f'Directory {Dir} has a trailing slash. Removing it.')
        Dir = Dir[:-1]
    
    Fils = glob.glob(f'{Dir}/*_RAS.nii')
    Fils.sort()
    LOGGER.info(f'Processing {Dir}')
    if not os.path.exists(f'{SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}'):
        os.mkdir(f'{SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}')
        LOGGER.debug(f'Created directory: {SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}')
    # Coregister all subsequent scans
    LOGGER.debug(f'Utilizing {Fils[1]} as reference for coregistration')
    for ii in Fils[2:]:
        # Perform coregistration to 01_01
        LOGGER.info(f'Current DIRECTORY: {Dir}')
        LOGGER.info(f'Current ID: {Dir.split(os.sep)[-1]}')
        LOGGER.info(f'Current seperator: {os.sep}')
        LOGGER.info(f'Split Directory: {Dir.split(os.sep)}')
        dest = f'{SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}{os.sep}{ii.split(os.sep)[-1]}'
        dest = dest.replace('.nii','')
        LOGGER.info(f'Coregistering scan to save to {dest}')
        aff_save = (os.sep).join(dest.split(os.sep)[:-1])
        #os.system(f'reg_aladin -ref {Fils[1]} -flo {ii} -aff {dest}_aff.txt')
        #os.system(f'reg_f3d -ref {Fils[1]} -flo {ii} -res {dest}.nii -aff {dest}_aff.txt -be 0.1')
        #os.system(f'rm {dest}_aff.txt')
        try:
            subprocess.run(['reg_f3d', '-ref', Fils[1], '-flo', ii, '-res', f'{dest}.nii', '-be', '0.1'], check=True)
        except subprocess.CalledProcessError as e:
            LOGGER.error(f'Error during coregistration: {e}')

        LOGGER.info(f'Coregistered: {ii}')
    # Coregister the first scan
    dest = f'{SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}{os.sep}{Fils[0].split(os.sep)[-1]}'
    dest = dest.replace('.nii','')
    #os.system(f'reg_aladin -ref {Fils[1]} -flo {Fils[0]} -aff {dest}_aff.txt')
    #os.system(f'reg_f3d -ref {Fils[1]} -flo {Fils[0]} -res {dest}.nii -aff {dest}_aff.txt -be 0.1')
    #os.system(f'rm {dest}_aff.txt')
    try:
        subprocess.run(['reg_f3d', '-ref', Fils[1], '-flo', Fils[0], '-res', f'{dest}.nii', '-be', '0.1'], check=True)
    except subprocess.CalledProcessError as e:
        LOGGER.error(f'Error during coregistration: {e}')
    
    LOGGER.info(f'Coregistered: {Fils[0]}')

    # Copy reference to coregistered samples
    subprocess.run(['cp', Fils[1], f'{SAVE_DIR}{os.sep}{Dir.split(os.sep)[-1]}{os.sep}{Fils[1].split(os.sep)[-1]}'], check=True)
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

    if args.dir_idx is not None:
        with open(args.dir_list, 'r') as f:
            Dirs = f.readlines()
        Dirs = [x.strip() for x in Dirs]
        if args.dir_idx >= len(Dirs):
            LOGGER.error(f'Directory index {args.dir_idx} is out of range. Please provide a valid index.')
            exit()
        else:
            Dir = [Dirs[args.dir_idx]]
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
            LOGGER.warning(f'Created directory: {SAVE_DIR}')
        LOGGER.info(f'Processing single directory: {Dir}')
        align(Dir[0])
        exit()

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
    Dirs = glob.glob(f'{BASE_PATH}/data/RAS/*')
    if TEST:
        Dirs = Dirs[:N_TEST]
    
    # Run the coregistration
    run_with_progress(align, Dirs, Parallel=PARALLAL)
