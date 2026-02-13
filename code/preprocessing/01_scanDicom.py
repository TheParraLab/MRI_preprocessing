# Standard imports
import os
import time
import argparse
import subprocess
import pickle
import random
from pathlib import Path
import json
# Third-party imports
import pydicom as pyd
import pandas as pd
# Function imports
from multiprocessing import cpu_count, Event
import threading
# Custom imports
from toolbox import get_logger, run_function
from DICOM import DICOMextract


# Define command line arguments
parser = argparse.ArgumentParser(description='Extract DICOM data to build Data_table.csv')
parser.add_argument('--test', nargs='?', const=100, type=int, help='Run in test mode with an optional number of dicom directories to scan (default: 100)')
parser.add_argument('--multi', '-m', nargs='?', const=cpu_count()-1, type=int, help='Run with multiprocessing enabled, using provided number of cpus (default: max-1)')
parser.add_argument('-p', '--profile', action='store_true', help='Run with profiler enabled')
parser.add_argument('--save_dir', nargs='?', default='/FL_system/data/', type=str, help='Location to save the constructed Data_table.csv (default: /FL_system/data/)')
parser.add_argument('--scan_dir', nargs='?', default='/FL_system/data/raw/', type=str, help='Location to recursively scan for dicom files (default: /FL_system/data/raw/)')
parser.add_argument('--dir_idx', type=int, help='Index of the folder to process from dirs_to_process.txt')
parser.add_argument('--dir_list', type=str, default='dirs_to_process.txt', help='Path to the directory list file')
parser.add_argument('--sample-pct', type=float, default=0.0, help='Percent of .dcm files to sample per directory (0 = full scan)')
parser.add_argument('--sample-seed', type=int, default=None, help='Optional random seed for sampling reproducibility')
parser.add_argument('--checkpoint-dir', type=str, default=None, help='Directory to store checkpoint files (default: <SAVE_DIR>/checkpoints/)')
parser.add_argument('--resume', action='store_true', help='Resume from available checkpoints if present')
args = parser.parse_args()

# Apply cli arguments
SAVE_DIR = args.save_dir
SCAN_DIR = args.scan_dir
TEST = args.test is not None # If True, the script will run in test mode
N_TEST = args.test if TEST else 100 # Number of dicom directories to scan if TEST is True
PARALLEL = args.multi is not None # If True, the script will run with multiprocessing enabled
N_CPUS = args.multi if PARALLEL else cpu_count()-1 # Number of cpus to use if PARALLEL is True
PROFILE = args.profile # If True, the script will run with the profiler enabled
SAMPLE_PCT = args.sample_pct
SAMPLE_SEED = args.sample_seed
if SAMPLE_SEED is not None:
    random.seed(SAMPLE_SEED)
# Checkpointing settings
CHECKPOINT_DIR = args.checkpoint_dir
RESUME = args.resume
# Profiler imports
if PROFILE:
    import yappi
    import pstats
    import io

# Generate logger
LOGGER = get_logger('01_scanDicom', f'{SAVE_DIR}/logs/')

def _ensure_checkpoint_dir():
    """Return checkpoint directory path, creating it if necessary."""
    global CHECKPOINT_DIR
    if CHECKPOINT_DIR is None:
        CHECKPOINT_DIR = os.path.join(SAVE_DIR, 'checkpoints/')
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    except Exception:
        # If we can't create the checkpoint dir, fallback to SAVE_DIR
        CHECKPOINT_DIR = SAVE_DIR
    return CHECKPOINT_DIR

def save_checkpoint(name: str, obj):
    """Atomically save checkpoint object to PICKLE file in checkpoint dir.

    name: base name without extension (e.g., 'dirs', 'dicom_files', 'info')
    """
    d = _ensure_checkpoint_dir()
    tmp_path = os.path.join(d, f'.{name}.tmp')
    final_path = os.path.join(d, f'{name}.pkl')
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, final_path)
        LOGGER.info(f'Wrote checkpoint: {final_path}')
    except Exception as e:
        LOGGER.error(f'Failed to write checkpoint {final_path}: {e}')

def load_checkpoint(name: str):
    """Load checkpoint object if it exists, else return None."""
    d = CHECKPOINT_DIR or os.path.join(SAVE_DIR, 'checkpoints/')
    path = os.path.join(d, f'{name}.pkl')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        LOGGER.info(f'Loaded checkpoint: {path}')
        return obj
    except Exception as e:
        LOGGER.error(f'Failed to load checkpoint {path}: {e}')
        return None

#### Preprocessing | Step 1: Extract DICOM data ####
# This script scans the input directory for dicom files and extracts necessary header information
#
# The extracted information is saved to {SAVE_DIR}/Data_table.csv

def _has_dcm_magic(path: str) -> bool:
    """Cheap check for DICOM preamble + 'DICM' marker at offset 128."""
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False

def _dir_contains_mr(item):
    """Worker to check one directory for at least one MR DICOM.
    `item` is a tuple (dirpath, filenames_list).
    Returns dirpath if MR found, else None.
    """
    dirpath, filenames = item
    # cooperative cancellation: if another worker already found enough dirs
    # Filter candidate names with .dcm extension first
    candidates = [fn for fn in filenames if fn.lower().endswith('.dcm')]
    if not candidates:
        return None

    # Optional deterministic sampling hook if desired:
    # rng = random.Random(SAMPLE_SEED) if SAMPLE_SEED is not None else random
    # if SAMPLE_PCT and SAMPLE_PCT > 0:
    #     k = max(1, int(len(candidates) * (SAMPLE_PCT / 100.0)))
    #     if k < len(candidates):
    #         candidates = rng.sample(candidates, k)

    # First pass: check magic bytes to avoid pydicom overhead
    likely = []
    fallback = []
    for fn in candidates:
        p = os.path.join(dirpath, fn)
        if _has_dcm_magic(p):
            likely.append(p)
        else:
            fallback.append(p)

    # Check likely files with pydicom until we find MR
    for p in likely:
        try:
            dcm = pyd.dcmread(p, stop_before_pixels=True, force=False)
            if getattr(dcm, 'Modality', None) == 'MR':
                # If shared counter provided, increment and set stop flag when threshold reached
                    return dirpath
        except Exception:
            # ignore and continue
            continue

    # Fallback: some valid files may not have the DICM magic; check fallback list
    for p in fallback:
        try:
            dcm = pyd.dcmread(p, stop_before_pixels=True, force=False)
            if getattr(dcm, 'Modality', None) == 'MR':
                return dirpath
        except Exception:
            continue

    return None


# ...existing code...


#############################
## Main functions
#############################
def extractDicom(f: str):
    """
    Extracts DICOM information from a file.

    Args:
        f (str): Path to the DICOM file.

    Returns:
        dict: A dictionary containing extracted DICOM information.
    """

    try:
        LOGGER.debug(f'Extracting information for file: {f}')
        extract = DICOMextract(f) # Initialize the DICOMextract class

        # Extract the necessary information from the DICOM file
        result = {
            'PATH': f,
            'Orientation': extract.Orientation(),
            'ID': extract.ID(),
            'Accession': extract.Accession(),
            'Name': extract.Name(),
            'DATE': extract.Date(),
            'DOB': extract.DOB(),
            'Series_desc': extract.Desc(),
            'Modality': extract.Modality(),
            'AcqTime': extract.Acq(),
            'SrsTime': extract.Srs(),
            'ConTime': extract.Con(),
            'StuTime': extract.Stu(),
            'TriTime': extract.Tri(),
            'InjTime': extract.Inj(),
            'ScanDur': extract.ScanDur(),
            'Lat': extract.LR(),
            'NumSlices': extract.NumSlices(),
            'Thickness': extract.Thickness(),
            'BreastSize': extract.BreastSize(),
            'DWI': extract.DWI(),
            'Type': extract.Type(),
            'Series': extract.Series()
        }
        LOGGER.debug(f'Completed extraction for file: {f}')
        return result
    except Exception as e:
        LOGGER.error(f'Error extracting information for file: {f} | {e}')
        return None

def find_all_dicom_dirs(directory, N_test=None):
    """Faster parallel implementation using your run_function wrapper.

    Scans the filesystem in batches and submits directory-check work to
    `run_function`. This lets us stop early once we've found N_test
    directories containing MR DICOMs (if N_test is provided).

    Returns a list of directory paths that contain at least one MR DICOM.
    """
    dir_items = []
    dicom_dirs = []
    batch_size = 256  # number of directories to submit per batch

    # Walk the tree and process in batches so we can stop early when N_test met
    walker = os.walk(directory, followlinks=False)
    for root, _, files in walker:
        if any(f.lower().endswith('.dcm') for f in files):
            dir_items.append((root, files))

        # If batch full, process it
        if len(dir_items) >= batch_size:
            # Submit the batch for parallel checking
            results = run_function(LOGGER, _dir_contains_mr, dir_items, Parallel=True, P_type='thread', N_CPUS=N_CPUS)
            found = [r for r in results if r]
            dicom_dirs.extend(found)
            dir_items = []
            # stop early if we've accumulated enough directories
            if N_test is not None and len(dicom_dirs) >= N_test:
                dicom_dirs = dicom_dirs[:N_test]
                LOGGER.info(f'Found {len(dicom_dirs)} directories containing DICOM files (stopping early)')
                return dicom_dirs

    # Process any remaining dirs
    if dir_items:
        results = run_function(LOGGER, _dir_contains_mr, dir_items, Parallel=True, P_type='thread', N_CPUS=N_CPUS)
        found = [r for r in results if r]
        dicom_dirs.extend(found)

    if not dicom_dirs:
        LOGGER.warning(f'No directories containing DICOM files found in {directory}')
    else:
        LOGGER.info(f'Found {len(dicom_dirs)} directories containing DICOM files')

    if N_test is not None:
        return dicom_dirs[:N_test]
    return dicom_dirs

def findDicom(directory):
    """
    Scan a directory for DICOM files and determine directory formatting through SeriesNumber.
    For each observed series, only the first file is logged.

    Args:
        directory (str): The directory to scan for DICOM files.

    Returns:
        List[str]: A list of paths to the DICOM files found in the directory.
    """

    dicom_files = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Efficiently get candidate filenames with .dcm extension
        dcm_candidates = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_candidates:
            continue

        # Decide whether to sample by percentage
        if SAMPLE_PCT and SAMPLE_PCT > 0 and len(dcm_candidates) > 1:
            # Use a local RNG for deterministic sampling when SAMPLE_SEED is set
            rng = random.Random(SAMPLE_SEED) if SAMPLE_SEED is not None else random
            k = max(1, int(len(dcm_candidates) * (SAMPLE_PCT / 100.0)))
            # If k >= len, just scan all
            if k >= len(dcm_candidates):
                sample_list = dcm_candidates
                fallback_allowed = False
            else:
                sample_list = rng.sample(dcm_candidates, k)
                fallback_allowed = True
        else:
            sample_list = dcm_candidates
            fallback_allowed = False

        found_series = {}

        # Try sampled candidates first
        for fname in sample_list:
            path = os.path.join(root, fname)
            try:
                data = pyd.dcmread(path, stop_before_pixels=True, force=True)
            except Exception as e:
                LOGGER.debug(f'Skipping unreadable/non-DICOM file: {path} | {e}')
                continue
            series = getattr(data, 'SeriesNumber', None)
            if series is not None and series not in found_series:
                found_series[series] = path

        # If sampling was used and results are ambiguous (none or multiple series), fall back to full scan
        if fallback_allowed and (len(found_series) == 0 or len(found_series) > 1):
            LOGGER.debug(f'Ambiguous sampling in {root} (found series={list(found_series.keys())}), falling back to full scan')
            full_found = {}
            for fname in dcm_candidates:
                path = os.path.join(root, fname)
                try:
                    data = pyd.dcmread(path, stop_before_pixels=True, force=True)
                except Exception:
                    continue
                series = getattr(data, 'SeriesNumber', None)
                if series is not None and series not in full_found:
                    full_found[series] = path
            found_series = full_found

        # Record the first file for each detected series
        for series, path in found_series.items():
            dicom_files.append(path)

        LOGGER.debug(f'{root} contains series {sorted(found_series.keys())} | {len(found_series)} series found')

    return dicom_files
#############################
## Main script
#############################
def main(out_name='Data_table.csv', SAVE_DIR='', SCAN_DIR=''):
    # Validate input directories
    assert os.path.exists(SCAN_DIR), f'SCAN_DIR {SCAN_DIR} does not exist. Please provide a valid directory.'

    # Create the save directory if it does not exist
    if not os.path.exists(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR)
            LOGGER.info(f'Created directory {SAVE_DIR}')
        except Exception as e:
            LOGGER.error(f'Error creating directory {SAVE_DIR}: {e}')

    # Print the current configuration
    LOGGER.info('Starting scanDicom: Step 01')
    LOGGER.info(f'SCAN_DIR: {SCAN_DIR}')
    LOGGER.info(f'SAVE_DIR: {SAVE_DIR}')
    LOGGER.info(f'PARALLEL: {PARALLEL}')
    if PROFILE:
        LOGGER.info(f'Profiling is enabled')

    # Check if the output already exists
    if out_name in os.listdir(SAVE_DIR):
        LOGGER.error(f'{out_name} already exists. Skipping step 01')
        LOGGER.error(f'To re-run this step, delete the existing {out_name} file')
        exit()

    # Finding main directory and subdirectories
    LOGGER.info('Finding all directories containing DICOM files')
    if TEST:
        LOGGER.info(f'Running in test mode with a maximum of {N_TEST} directories')
    # Try to resume from checkpoint if requested
    dicom_dirs = None
    if RESUME:
        try:
            _ensure_checkpoint_dir()
            dicom_dirs = load_checkpoint('dirs')
        except Exception:
            dicom_dirs = None

    if dicom_dirs is None:
        dicom_dirs = find_all_dicom_dirs(SCAN_DIR, N_test=N_TEST if TEST else None)
        try:
            save_checkpoint('dirs', dicom_dirs)
        except Exception:
            pass
    #if TEST:
        #dicom_dirs = dicom_dirs[:N_TEST]
        #LOGGER.info(f'Running in test mode with {N_TEST} directories')

    # Scan the directories for dicom files
    LOGGER.info('Analyzing DICOM directories')
    # Attempt to resume dicom_files from checkpoint
    dicom_files = None
    if RESUME:
        try:
            dicom_files = load_checkpoint('dicom_files')
        except Exception:
            dicom_files = None

    if dicom_files is None:
        dicom_files = run_function(LOGGER, findDicom, dicom_dirs, Parallel=PARALLEL, P_type='thread')
        dicom_files = [f for sublist in dicom_files for f in sublist] # Flatten the list of lists
        try:
            save_checkpoint('dicom_files', dicom_files)
        except Exception:
            pass
    LOGGER.info(f'Found {len(dicom_files)} dicom files in the input directory')
    # Extract the dicom information
    LOGGER.info('Extracting information from dicom files')
    # Attempt to resume INFO from checkpoint
    INFO = None
    if RESUME:
        try:
            INFO = load_checkpoint('info')
        except Exception:
            INFO = None

    if INFO is None:
        INFO = run_function(LOGGER, extractDicom, dicom_files, Parallel=PARALLEL, P_type='thread')
        try:
            save_checkpoint('info', INFO)
        except Exception:
            pass

    Data_table = pd.DataFrame(INFO) # Convert the extracted information to a pandas dataframe
    # Write Data_table to CSV atomically
    out_path = os.path.join(SAVE_DIR, out_name)
    tmp_out = out_path + '.tmp'
    try:
        Data_table.to_csv(tmp_out, index=False)
        os.replace(tmp_out, out_path)
    except Exception as e:
        LOGGER.error(f'Failed to write output CSV {out_path}: {e}')
    LOGGER.info(f'DICOM information extraction completed and saved to {out_name}')


if __name__ == '__main__':
    # Start the profiler if enabled
    if PROFILE:
        LOGGER.info('Profiling enabled')
        yappi.start()
        LOGGER.info('Starting main function')

    # Create the save directory when necessary
    if not os.path.exists(SAVE_DIR):
        # Use try-except to handle directory creation, in case parallel processes try to create the same directory
        try:
            os.makedirs(SAVE_DIR)
            LOGGER.info(f'Created directory {SAVE_DIR}')
        except Exception as e:
            LOGGER.error(f'Error creating directory {SAVE_DIR}: {e}')
    
    # If not running on an HPC
    if args.dir_idx is None:
        main(SCAN_DIR=SCAN_DIR, SAVE_DIR=SAVE_DIR)
    # If running on an HPC
    else:
        assert os.path.exists(args.dir_list), f'Directory list file {args.dir_list} does not exist'
        # Save to temporary directory
        SAVE_DIR = os.path.join(SAVE_DIR, 'tmp/')
        with open(args.dir_list, 'rb') as f:
            Dirs = pickle.load(f)
        Dir = Dirs[args.dir_idx]
        SCAN_DIR = Dir # Set the scan directory to the one specified by the index
        LOGGER.info(f'Processing single directory: {args.dir_idx}')
        main(out_name=f'Data_table_{args.dir_idx}.csv', SCAN_DIR=SCAN_DIR, SAVE_DIR=SAVE_DIR)

        if args.dir_idx == len(Dirs) - 1:
            LOGGER.info('Last script, compiling results')
            Tables = []
            while len(Tables) < len(Dirs):
                LOGGER.info('Waiting for all tables to be compiled')
                time.sleep(5)
                Tables = os.listdir(SAVE_DIR)
                Tables = [table for table in Tables if table.endswith('.csv')]
            LOGGER.info('All tables present, compiling...')
            Data_table = pd.DataFrame()
            for table in Tables:
                LOGGER.info(f'Compiling {table}')
                Data_table = pd.concat([Data_table, pd.read_csv(f'{SAVE_DIR}{table}')], ignore_index=True)
            SAVE_DIR = SAVE_DIR.replace('tmp/', '')
            Data_table.to_csv(f'{SAVE_DIR}Data_table.csv', index=False)
            LOGGER.info(f'Compiled results saved to {SAVE_DIR}Data_table.csv')
            try:
                subprocess.run(['rm', '-r', f'{SAVE_DIR}tmp/'], check=True)
                LOGGER.info(f'Deleted temporary directory {SAVE_DIR}tmp/')
            except Exception as e:
                LOGGER.error(f'Error deleting temporary directory {SAVE_DIR}tmp/: {e}')

    # Finalize the profiler if enabled
    if PROFILE:
        LOGGER.info('Main function completed')
        yappi.stop()
        profile_output_path = 'step01_profile.yappi'
        LOGGER.info(f'Writing profile results to {profile_output_path}')
        yappi.get_func_stats().save(profile_output_path, type='pstat')
        LOGGER.info(f'Profile results saved to {profile_output_path}')
    exit()