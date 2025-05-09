import os
import pydicom as pyd
import glob
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Manager, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from typing import Callable, List, Any
from functools import partial
# Custom Imports
from toolbox import ProgressBar, get_logger

# Define command line arguments
parser = argparse.ArgumentParser(description='Convert Nifti files to RAS orientation')
parser.add_argument('--scan_dir', type=str, required=False, help='Directory containing scans to process')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the output')
parser.add_argument('--dir_idx', type=int, required=True, help='Index of the directory to process')
parser.add_argument('--dir_list', type=str, default='list.txt', help='List of directories to process')
parser.add_argument('--multi', '-m', nargs='?', const=cpu_count()-1, type=int, help='Run with multiprocessing enabled, using provided number of cpus (default: max-1)')
parser.add_argument('-p', '--profile', action='store_true', help='Run with profiler enabled')
args = parser.parse_args()

# Global variables for progress bar
Progress = None
manager = Manager()
progress_queue = manager.Queue()

# Other global variables
LOAD_DIR = args.scan_dir #'/FL_system/data/nifti/'
SAVE_DIR = args.save_dir #'/FL_system/data/RAS/'
TEST = False
N_TEST = 40
PARALLEL = args.multi is not None # If True, the script will run with multiprocessing enabled
PROFILE = args.profile # If True, the script will run with the profiler enabled
PROGRESS = False

LOGGER = get_logger('04_saveRAS', f'{SAVE_DIR}/logs/')

debug = 0
#### Preprocessing | Step 4: Save RAS Nifti Files ####
# This script is for taking the semi-processed nifti files and saving them into RAS
#
# This script utilizes the nibabel library to convert the nifti files to RAS orientation
# It requires the nifti files to be present in the LOAD_DIR directory, this is produced in the previous step
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

def RAS_convert(dir: str, save_path=SAVE_DIR):
    # This function converts all nifti files in the input directory to RAS orientation
    # It saves the RAS files in the output directory
    Fils = glob.glob(f'{dir}/*.nii')
    LOGGER.debug(f'Found {len(Fils)} files in {dir}')
    Fils.sort()
    save_path = os.path.join(save_path, dir.split(os.sep)[-1])
    if not os.path.exists(f'{save_path}'):
        LOGGER.debug(f'Creating directory: {save_path}')
        os.mkdir(f'{save_path}')
    for ii in Fils:
        if ii.endswith('00a.nii'):
            LOGGER.debug(f'{ii} | found 00a.nii, attempting to isolate FS sample...')
            json_00 = json.load(open(f'{dir}/00.json'))
            json_00a = json.load(open(f'{dir}/00a.json'))
            LOGGER.debug(f'{dir} | 00_desc: {json_00["SeriesDescription"]}')
            LOGGER.debug(f'{dir} | 00a_desc: {json_00a["SeriesDescription"]}')  
            if 'FS' in json_00['SeriesDescription']:
                LOGGER.debug(f'{dir} | Found FS in 00')
                Fils.remove(f'{dir}/00a.nii')
            elif 'FS' in json_00a['SeriesDescription']:
                LOGGER.debug(f'{dir} | Found FS in 00a')
                Fils.remove(f'{dir}/00.nii')
            else:
                LOGGER.error(f'{dir} | No FS found in 00 or 00a')
                return
    for ii in Fils:
        LOGGER.debug(f'Processing: {ii}')

        img = nib.load(ii)
        data = img.get_fdata()
        aff = img.affine

        # Get the orientations
        new_ornt = nib.orientations.axcodes2ornt('RAS')
        old_ornt = nib.orientations.io_orientation(aff)

        # Get the transformation matrix to convert to the new orientation
        transform = nib.orientations.ornt_transform(old_ornt, new_ornt)

        # Apply the transformation to the data and the affine
        ras_data = nib.orientations.apply_orientation(data, transform)
        
        # Create a square transformation matrix
        square_transform = np.eye(3)
        for _ , (axis, direction) in enumerate(transform):
            square_transform[int(axis), :] = 0
            square_transform[int(axis), int(axis)] = direction

        # Apply the transformation to the rotation and scaling part of the affine
        ras_affine = aff.copy()
        ras_affine[:3, :3] = np.dot(aff[:3, :3], np.linalg.inv(square_transform))

        # Update the translation part of the affine
        ras_affine[:3, 3] = (np.array(ras_data.shape) - 1) / 2.0 - np.dot(ras_affine[:3, :3], (np.array(data.shape) - 1) / 2.0)
    
        # Create a new Nifti1Image with the RAS data and updated affine
        ras_img = nib.Nifti1Image(ras_data, ras_affine)
        #save_path = ii.replace('/data/nifti', '/data/RAS')
        if ii.endswith('00a.nii'):
            ii = ii.replace('00a.nii', '00.nii')
        ii = ii.replace('.nii', '_RAS.nii')
        nib.save(ras_img,os.path.join(save_path,ii.split(os.sep)[-1]))
        LOGGER.debug(f'{ii} | Saving: {os.path.join(save_path,ii.split(os.sep)[-1])}')

    return 'completed'

if __name__ == '__main__':
    LOGGER.info('Starting saveRAS: Step 04')
    LOGGER.info(f'LOAD_DIR: {LOAD_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if TEST:
        LOGGER.info(f'TEST: {TEST}')
        LOGGER.info(f'N_TEST: {N_TEST}')

    if not os.path.exists(SAVE_DIR):
        try:
            os.mkdir(SAVE_DIR)
            LOGGER.info(f'Created directory: {SAVE_DIR}')
        except Exception as e:
            LOGGER.error(f'Error creating directory {SAVE_DIR}: {e}')
    
    # If not running on an HPC
    if args.dir_idx is None:
        Dirs = glob.glob(f'{LOAD_DIR}*')
        if TEST:
            Dirs = Dirs[:N_TEST]
        run_with_progress(RAS_convert, Dirs, Parallel=PARALLEL)
    else:
        assert os.path.exists(args.dir_list), f'Directory list file {args.dir_list} does not exist'
        # Save to temporary directory
        with open(args.dir_list, 'rb') as f:
            Dirs = pickle.load(f)
        Dir = Dirs[args.dir_idx]
        # Make Dir list if not
        if type(Dir) == str:
            LOGGER.debug(f'Converting Dir to list: {Dir}')
            Dir = [Dir]
        LOGGER.info(f'Processing index {args.dir_idx} of {len(Dirs)}: {Dir}')
        run_with_progress(RAS_convert, Dir, Parallel=PARALLEL, save_path=SAVE_DIR)

        if args.dir_idx == len(Dirs) - 1:
            LOGGER.info('Last script, compiling results')


    #Data_table = pd.read_csv('/FL_system/data/Data_table.csv')
    #Data_table_timing = pd.read_csv('/FL_system/data/Data_table_timing.csv')
    


    LOGGER.info('Completed saveRAS: Step 04')
    LOGGER.info('All files saved to RAS directory')
    LOGGER.info('Exiting saveRAS: Step 04')
