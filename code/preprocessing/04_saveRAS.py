import os
import pydicom as pyd
import glob
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Manager, cpu_count, Lock
from typing import Callable, List, Any
from functools import partial
# Custom Imports
from toolbox import run_function, get_logger

# Define command line arguments
parser = argparse.ArgumentParser(description='Convert Nifti files to RAS orientation')
parser.add_argument('--scan_dir', type=str, required=False, help='Directory containing scans to process')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the output')
parser.add_argument('--dir_idx', type=int, required=False, help='Index of the directory to process')
parser.add_argument('--dir_list', type=str, default='list.txt', help='List of directories to process')
parser.add_argument('--multi', '-m', nargs='?', const=cpu_count()-1, type=int, help='Run with multiprocessing enabled, using provided number of cpus (default: max-1)')
parser.add_argument('-p', '--profile', action='store_true', help='Run with profiler enabled')
parser.add_argument('--test', nargs='?', type=int, const=10, help='Number of test directories to process')
args = parser.parse_args()

# TODO: Need to update stop_flag into an external file saved into the data/logs/.flag directory
# TODO: Validate the run_function command works as a replacement for run_with_progress
# TODO: Change progress updating to instead save a list of completed operations instead of logging to-be-completed

# Global variables for progress bar
#Progress = None
manager = Manager()
disk_space_lock = Lock()
stop_flag = manager.Event()
#progress_queue = manager.Queue()

# Other global variables
LOAD_DIR = args.scan_dir #'/FL_system/data/nifti/'
SAVE_DIR = args.save_dir #'/FL_system/data/RAS/'
TEST = args.test is not None # If True, the script will run with a limited number of directories
if TEST:
    N_TEST = args.test
PARALLEL = args.multi is not None # If True, the script will run with multiprocessing enabled
PROFILE = args.profile # If True, the script will run with the profiler enabled
DISK_SPACE_THRESHOLD = 10 * 1024 * 1024 * 1024  # 100 GB
#PROGRESS = False

LOGGER = get_logger('04_saveRAS', f'/FL_system/data/logs/')

#### Preprocessing | Step 4: Save RAS Nifti Files ####
# This script is for taking the semi-processed nifti files and saving them into RAS
#
# This script utilizes the nibabel library to convert the nifti files to RAS orientation
# It requires the nifti files to be present in the LOAD_DIR directory, this is produced in the previous step

def check_source_files(source_path: str) -> bool:
    """Check if the source path file exists contains files."""
    return (len(glob.glob(f'{source_path}/*')) > 0) or (len(glob.glob(f'{source_path}/*/*')) > 0)

def check_disk_space(directory: str) -> bool:
    """Check if there is enough disk space available."""
    statvfs = os.statvfs(directory)
    available_space = statvfs.f_frsize * statvfs.f_bavail
    if available_space < DISK_SPACE_THRESHOLD*2:
        LOGGER.debug(f'Available space: {available_space}')
    return available_space > DISK_SPACE_THRESHOLD

def save_progress(data, filename):
    """Save progress to a file."""
    LOGGER.info(f'Saving progress to {filename}')
    if os.path.exists(f'{SAVE_DIR}{filename}'):
        os.remove(f'{SAVE_DIR}{filename}')
    with open(f'{SAVE_DIR}{filename}', 'wb') as f:
        pickle.dump(data, f)

def load_progress(filename):
    """Load progress from a file."""
    if os.path.exists(f'{SAVE_DIR}{filename}'):
        LOGGER.info(f'Loading progress from {filename}')
        with open(f'{SAVE_DIR}{filename}', 'rb') as f:
            return pickle.load(f)
    return None

def RAS_convert(dir: str, save_path=SAVE_DIR):
    # This function converts all nifti files in the input directory to RAS orientation
    # It saves the RAS files in the output directory

    if stop_flag.is_set():
        LOGGER.info('stop flag is set, exiting')
        return
    
    with disk_space_lock:
        if not check_disk_space(SAVE_DIR):
            if not stop_flag.is_set():
                LOGGER.warning('Disk space is running low.  Pausing...')
                stop_flag.set()
                LOGGER.warning('Stop flag set')
            return
        if not check_source_files(dir):
            if not stop_flag.is_set():
                LOGGER.warning(f'No files found in {dir}')
                stop_flag.set()
                LOGGER.warning('Stop flag set')
            return
        
    Fils = glob.glob(f'{dir}/*.nii')
    LOGGER.debug(f'{dir} | Found {len(Fils)} files')
    Fils.sort()
    Fils = [os.path.split(ii)[-1] for ii in Fils]
    save_path = os.path.join(save_path, dir.split(os.sep)[-1])
    if not os.path.exists(f'{save_path}'):
        LOGGER.debug(f'Creating directory: {save_path}')
        os.mkdir(f'{save_path}')
    Fils_out = glob.glob(f'{save_path}/*_RAS.nii') or []
    Fils_out = [os.path.split(ii)[-1] for ii in Fils_out]
    Fils_out = [ii.replace('_RAS.nii', '.nii') for ii in Fils_out]
    LOGGER.debug(f'{dir} | Found {len(Fils_out)} files in {save_path}')
    for ii in Fils:
        LOGGER.debug(f'{dir} | Processing: {os.path.join(dir, ii)}')
        if ii.endswith('00a.nii'):
            LOGGER.debug(f'{dir} | found 00a.nii, attempting to isolate FS sample...')
            json_00 = json.load(open(f'{dir}/00.json'))
            json_00a = json.load(open(f'{dir}/00a.json'))
            LOGGER.debug(f'{dir} | 00_desc: {json_00["SeriesDescription"]}')
            LOGGER.debug(f'{dir} | 00a_desc: {json_00a["SeriesDescription"]}')  
            if 'FS' in json_00['SeriesDescription']:
                LOGGER.debug(f'{dir} | Found FS in 00')
                Fils.remove(f'00a.nii')
            elif 'FS' in json_00a['SeriesDescription']:
                LOGGER.debug(f'{dir} | Found FS in 00a')
                Fils.remove(f'00.nii')
            else:
                LOGGER.error(f'{dir} | No FS found in 00 or 00a')
                return
    for ii in Fils:
        LOGGER.debug(f'{dir} | Processing: {os.path.join(dir, ii)}')
        LOGGER.debug(f'{dir} | Checking if {ii} is in {Fils_out}')

        if ii in Fils_out:
            LOGGER.warning(f'{dir} | {ii} | Already processed, skipping')
            continue
        LOGGER.debug(f'{dir} | {ii} | Not processed, converting to RAS')
        # Load the Nifti file
        img = nib.load(os.path.join(dir, ii))
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
        nib.save(ras_img,os.path.join(save_path,ii))
        LOGGER.debug(f'{dir} | Saving: {os.path.join(save_path,ii)}')

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
        #run_with_progress(RAS_convert, Dirs, Parallel=PARALLEL)
        run_function(LOGGER, RAS_convert, Dirs, Parallel=PARALLEL, save_path=SAVE_DIR, P_type='process', stop_flag=stop_flag)
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
        #run_with_progress(RAS_convert, Dir, Parallel=PARALLEL, save_path=SAVE_DIR)
        run_function(LOGGER, RAS_convert, Dirs, Parallel=PARALLEL, save_path=SAVE_DIR, P_type='process', stop_flag=stop_flag)

    #if stop_flag.is_set():
    #    LOGGER.warning('Execution completed with stop flag set')
    #    LOGGER.warning('Some files may not have been processed')
    #    LOGGER.warning('Saving pending tasks to a progress file')
    #    save_progress(Dirs, 'pending_tasks.pkl')

    #Data_table = pd.read_csv('/FL_system/data/Data_table.csv')
    #Data_table_timing = pd.read_csv('/FL_system/data/Data_table_timing.csv')
    


    LOGGER.info('Completed saveRAS: Step 04')
    LOGGER.info('All files saved to RAS directory')
    LOGGER.info('Exiting saveRAS: Step 04')
