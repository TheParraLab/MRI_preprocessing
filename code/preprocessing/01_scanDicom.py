"""
DICOM Scanning and Extraction Script
===================================

This script scans a directory for DICOM files, extracts metadata from their headers,
and saves the information to a CSV file. It supports parallel processing,
checkpointing, and execution on HPC environments.

The script performs the following steps:
1.  Recursively finds all directories containing DICOM files (specifically MRI).
2.  Scans each directory to identify DICOM series, optionally sampling files.
3.  Extracts detailed header information from a representative file for each series.
4.  Compiles the extracted information into a pandas DataFrame.
5.  Saves the DataFrame to a CSV file.

Usage:
    python 01_scanDicom.py --scan_dir /path/to/raw/data --save_dir /path/to/output

Arguments:
    --scan_dir (str): Path to the directory containing raw DICOM files.
    --save_dir (str): Path to the directory where the output CSV will be saved.
    --test (int): Run in test mode with a limited number of directories.
    --multi (int): Number of CPUs to use for parallel processing.
    --profile: Enable profiling with yappi.
    --dir_idx (int): Index of the directory to process (for HPC array jobs).
    --dir_list (str): Path to the list of directories (for HPC array jobs).
    --sample-pct (float): Percentage of files to sample per directory (0 = full scan).
    --sample-seed (int): Random seed for sampling.
    --checkpoint-dir (str): Directory for storing checkpoints.
    --resume: Resume from existing checkpoints.

Dependencies:
    - pydicom
    - pandas
    - toolbox (custom)
    - DICOM (custom)
"""

# Standard imports
import os
import time
import argparse
import subprocess
import pickle
import random
from typing import List, Dict, Any, Optional

# Third-party imports
import pydicom as pyd
import pandas as pd

# Function imports
from multiprocessing import cpu_count
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
parser.add_argument('--dir_idx', type=int, help='Index of the folder to process from dirs_to_process.txt (for HPC array jobs)')
parser.add_argument('--dir_list', type=str, default='dirs_to_process.pkl', help='Path to the directory list file (for HPC array jobs)')
parser.add_argument('--sample-pct', type=float, default=0.0, help='Percent of .dcm files to sample per directory (0 = full scan)')
parser.add_argument('--sample-seed', type=int, default=None, help='Optional random seed for sampling reproducibility')
parser.add_argument('--checkpoint-dir', type=str, default=None, help='Directory to store checkpoint files (default: <SAVE_DIR>/checkpoints/)')
parser.add_argument('--profile-dir', type=str, default=None, help='Directory to store profiling output (default: <SAVE_DIR>/)')
parser.add_argument('--resume', action='store_true', help='Resume from available checkpoints if present')
args = parser.parse_args()

# Apply cli arguments
SAVE_DIR = args.save_dir
SCAN_DIR = args.scan_dir
TEST = args.test is not None # If True, the script will run in test mode
N_TEST = args.test if TEST else 100 # Number of dicom directories to scan if TEST is True
PARALLEL = args.multi is not None # If True, the script will run with multiprocessing enabled
PROFILE = args.profile # If True, the script will run with the profiler enabled
SAMPLE_PCT = args.sample_pct
SAMPLE_SEED = args.sample_seed
if SAMPLE_SEED is not None:
    random.seed(SAMPLE_SEED)

# Checkpointing settings
CHECKPOINT_DIR = args.checkpoint_dir
PROFILE_DIR = args.profile_dir
RESUME = args.resume

# Profiler imports
if PROFILE:
    import yappi

# Generate logger
# Note: get_logger might attempt to create directories. Ensure SAVE_DIR is writable or mocked in tests.
LOGGER = get_logger('01_scanDicom', f'{SAVE_DIR}/logs/')

def _ensure_checkpoint_dir() -> str:
    """
    Ensure the checkpoint directory exists.

    This function checks if the global CHECKPOINT_DIR is set. If not, it defaults to
    os.path.join(SAVE_DIR, 'checkpoints/'). It then attempts to create the directory.
    If creation fails, it falls back to using SAVE_DIR.

    Returns:
        str: The path to the checkpoint directory.
    """
    global CHECKPOINT_DIR
    if CHECKPOINT_DIR is None:
        CHECKPOINT_DIR = os.path.join(SAVE_DIR, 'checkpoints/')
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    except Exception:
        # If we can't create the checkpoint dir, fallback to SAVE_DIR
        CHECKPOINT_DIR = SAVE_DIR
    return CHECKPOINT_DIR

def _ensure_profile_dir() -> str:
    """
    Ensure the profile directory exists.

    This function checks if the global PROFILE_DIR is set. If not, it defaults to
    os.path.join(SAVE_DIR, 'profiles/'). It then attempts to create the directory.
    If creation fails, it falls back to the current working directory.

    Returns:
        str: The path to the profile directory.
    """
    global PROFILE_DIR
    if PROFILE_DIR is None:
        PROFILE_DIR = os.path.join(SAVE_DIR, 'profiles/')
    try:
        os.makedirs(PROFILE_DIR, exist_ok=True)
    except Exception:
        PROFILE_DIR = os.getcwd()
    return PROFILE_DIR

def save_checkpoint(name: str, obj: Any) -> None:
    """
    Atomically save a checkpoint object to a PICKLE file.

    The object is first written to a temporary file, which is then renamed to the
    final destination to ensure atomicity.

    Args:
        name (str): The base name of the checkpoint file (without extension).
                    Examples: 'dirs', 'dicom_files', 'info'.
        obj (Any): The Python object to serialize and save.

    Returns:
        None
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

def load_checkpoint(name: str) -> Optional[Any]:
    """
    Load a checkpoint object if it exists.

    Args:
        name (str): The base name of the checkpoint file (without extension).

    Returns:
        Optional[Any]: The loaded object if the checkpoint file exists and can be read,
                       otherwise None.
    """
    d = _ensure_checkpoint_dir()
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
    """
    Perform a fast check for a DICOM preamble and 'DICM' magic marker.

    Args:
        path (str): File path to check.

    Returns:
        bool: True if 'DICM' is found at offset 128, False otherwise.
    """
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False

#############################
## Main functions
#############################
def extractDicom(f: str) -> Optional[Dict[str, Any]]:
    """
    Extract DICOM information from a specific file path.

    This function utilizes the `DICOMextract` class to parse the DICOM header
    and retrieve specific fields such as Patient ID, Study Date, Modality, etc.

    TODO: Edge cases to consider: what if `DICOMextract` succeeds in initialization
          but certain critical fields are missing or corrupted? Currently, `UNKNOWN`
          is returned by methods in DICOMextract, but consider handling entirely unreadable
          files more gracefully.

    Args:
        f (str): Path to the DICOM file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing extracted DICOM information, including keys:
            - PATH, Orientation, ID, Accession, Name, DATE, DOB, Series_desc,
            - Modality, AcqTime, SrsTime, ConTime, StuTime, TriTime, InjTime,
            - ScanDur, Lat, NumSlices, Thickness, BreastSize, DWI, Type, Series.
            Returns None if extraction fails completely.
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

def find_all_dicom_dirs(directory: str, N_test: Optional[int] = None) -> List[str]:
    """
    Find all directories containing MRI DICOM files (.dcm) within the given root directory.

    Traverses the directory tree and checks for files ending with '.dcm'.
    Reads the 'Modality' tag from headers to explicitly filter for 'MR' (MRI scans).

    TODO: The use of `os.walk` here may be a performance bottleneck for heavily nested
          or networked file systems. Consider `os.scandir()` or parallelized tree walking
          for increased throughput.

    Args:
        directory (str): The root directory to search.
        N_test (Optional[int]): If provided, limits the search to the first N_test directories found.
                                Useful for quick testing.

    Returns:
        List[str]: A list of directory paths containing valid MRI DICOM files.
    """
    dicom_dirs = []
    N_found = 0
    for root, _, files in os.walk(directory, followlinks=False):
        has_mri = False
        candidates = [fn for fn in files if fn.lower().endswith('.dcm')]
        for fn in candidates:
            file_path = os.path.join(root, fn)
            if not _has_dcm_magic(file_path):
                continue
            try:
                dcm = pyd.dcmread(file_path, stop_before_pixels=True, force=False)
                if hasattr(dcm, 'Modality') and dcm.Modality == 'MR':
                    has_mri = True
                    break
            except Exception:
                continue
            
        if has_mri: 
            dicom_dirs.append(root)
            N_found += 1
            if N_test is not None and N_found >= N_test:
                break
            LOGGER.debug(f'Found DICOM files for MRI in {root}')

    if not dicom_dirs:
        LOGGER.warning(f'No directories containing DICOM files found in {directory}')
    else:
        LOGGER.info(f'Found {len(dicom_dirs)} directories containing DICOM files')

    if N_test is not None:
        return dicom_dirs[:N_test]
    return dicom_dirs

def findDicom(directory: str) -> List[str]:
    """
    Scan a directory for DICOM files and select one representative file per series.

    This function identifies all DICOM series within a directory. It can optionally
    sample a percentage of files to speed up the process if the directory contains
    many files. For each unique 'SeriesNumber' found, it returns the path to the
    first file encountered.

    TODO: Edge case - If 'SeriesNumber' is missing or ambiguous, a fallback to other
          unique identifiers (like 'SeriesInstanceUID') should be implemented to avoid
          erroneously merging distinct series.

    Args:
        directory (str): The directory to scan for DICOM files.

    Returns:
        List[str]: A list of paths to the selected representative DICOM files.
    """

    dicom_files = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Efficiently get candidate filenames with .dcm extension
        dcm_candidates = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_candidates:
            continue

        # Decide whether to sample by percentage to improve performance on large directories
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

        # Pre-filter: split candidates by magic bytes for fast rejection
        likely = [fn for fn in sample_list if _has_dcm_magic(os.path.join(root, fn))]
        fallback_cands = [fn for fn in sample_list if fn not in likely]

        # Try likely candidates first
        for fname in likely:
            path = os.path.join(root, fname)
            try:
                data = pyd.dcmread(path, stop_before_pixels=True, force=False)
            except Exception:
                continue
            series = getattr(data, 'SeriesNumber', None)
            if series is not None and series not in found_series:
                found_series[series] = path

        # If no series found among likely files, try fallback candidates
        if not found_series and fallback_cands:
            for fname in fallback_cands:
                path = os.path.join(root, fname)
                try:
                    data = pyd.dcmread(path, stop_before_pixels=True, force=False)
                except Exception:
                    continue
                series = getattr(data, 'SeriesNumber', None)
                if series is not None and series not in found_series:
                    found_series[series] = path

        # If sampling was used and results are incomplete, fall back to full scan
        if fallback_allowed and len(found_series) == 0:
            full_found = {}
            full_likely = [fn for fn in dcm_candidates if _has_dcm_magic(os.path.join(root, fn))]
            full_fallback = [fn for fn in dcm_candidates if fn not in full_likely]
            for fname in full_likely:
                path = os.path.join(root, fname)
                try:
                    data = pyd.dcmread(path, stop_before_pixels=True, force=False)
                except Exception:
                    continue
                series = getattr(data, 'SeriesNumber', None)
                if series is not None and series not in full_found:
                    full_found[series] = path
            if not full_found:
                for fname in full_fallback:
                    path = os.path.join(root, fname)
                    try:
                        data = pyd.dcmread(path, stop_before_pixels=True, force=False)
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
def main(out_name: str = 'Data_table.csv', SAVE_DIR: str = '', SCAN_DIR: str = '') -> None:
    """
    Main execution logic for scanning and extracting DICOM data.

    Orchestrates the pipeline process:
    1. Validates input directories.
    2. Finds directories containing DICOM files (resumes from checkpoint if needed).
    3. Scans directories to find representative files per series.
    4. Extracts DICOM header information in parallel.
    5. Saves the extracted metadata to a CSV file.

    Args:
        out_name (str): Name of the output CSV file (default: 'Data_table.csv').
        SAVE_DIR (str): Directory where the output file and checkpoints will be saved.
        SCAN_DIR (str): Directory to scan for DICOM files.

    Returns:
        None
    """
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

    # Check if the output already exists to avoid redundant processing
    if os.path.exists(os.path.join(SAVE_DIR, out_name)):
        LOGGER.error(f'{out_name} already exists. Skipping step 01')
        LOGGER.error(f'To re-run this step, delete the existing {out_name} file')
        return

    # Finding main directory and subdirectories
    LOGGER.info('Finding all directories containing DICOM files')
    if TEST:
        LOGGER.info(f'Running in test mode with a maximum of {N_TEST} directories')

    # Try to resume finding directories from checkpoint if requested
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

    # Scan the directories for dicom files
    LOGGER.info('Analyzing DICOM directories')

    # Attempt to resume finding representative files from checkpoint
    dicom_files = None
    if RESUME:
        try:
            dicom_files = load_checkpoint('dicom_files')
        except Exception:
            dicom_files = None

    if dicom_files is None:
        # Run finding files in parallel or sequentially
        dicom_files = run_function(LOGGER, findDicom, dicom_dirs, Parallel=PARALLEL, P_type='thread')
        dicom_files = [f for sublist in dicom_files for f in sublist] # Flatten the list of lists
        try:
            save_checkpoint('dicom_files', dicom_files)
        except Exception:
            pass
    LOGGER.info(f'Found {len(dicom_files)} dicom files in the input directory')

    # Extract the dicom information
    LOGGER.info('Extracting information from dicom files')

    # Attempt to resume extracted info from checkpoint
    INFO = None
    if RESUME:
        try:
            INFO = load_checkpoint('info')
        except Exception:
            INFO = None

    if INFO is None:
        # Run extraction in parallel or sequentially
        INFO = run_function(LOGGER, extractDicom, dicom_files, Parallel=PARALLEL, P_type='thread')
        try:
            save_checkpoint('info', INFO)
        except Exception:
            pass

    Data_table = pd.DataFrame(INFO) # Convert the extracted information to a pandas dataframe

    # Write Data_table to CSV atomically to prevent partial writes
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

    # Check if running in single directory mode (HPC array job)
    if args.dir_idx is None:
        # Normal execution
        main(SCAN_DIR=SCAN_DIR, SAVE_DIR=SAVE_DIR)

    # If running on an HPC with array jobs
    else:
        # In HPC mode, we process a single directory from a list
        assert os.path.exists(args.dir_list), f'Directory list file {args.dir_list} does not exist'

        # Save to temporary directory to avoid conflicts
        tmp_save_dir = os.path.join(SAVE_DIR, 'tmp/')
        os.makedirs(tmp_save_dir, exist_ok=True)

        # Load the list of directories
        with open(args.dir_list, 'rb') as f:
            Dirs = pickle.load(f)

        # Select the directory based on index
        Dir = Dirs[args.dir_idx]
        SCAN_DIR = Dir # Set the scan directory to the one specified by the index
        LOGGER.info(f'Processing single directory: {args.dir_idx}')

        # Run main for this specific directory
        main(out_name=f'Data_table_{args.dir_idx}.csv', SCAN_DIR=SCAN_DIR, SAVE_DIR=tmp_save_dir)

        # If this is the last job in the array, compile all results
        # Note: This simple check assumes the last index finishes last, which isn't guaranteed in all schedulers.
        # A more robust solution would be a separate compilation job.
        if args.dir_idx == len(Dirs) - 1:
            LOGGER.info('Last script, compiling results')
            Tables = []
            # Wait for all other jobs to finish (checking for file existence)
            while len(Tables) < len(Dirs):
                LOGGER.info('Waiting for all tables to be compiled')
                time.sleep(5)
                Tables = [t for t in os.listdir(tmp_save_dir) if t.endswith('.csv')]

            LOGGER.info('All tables present, compiling...')
            tables_to_concat = [pd.read_csv(os.path.join(tmp_save_dir, t)) for t in Tables]
            Data_table = pd.concat(tables_to_concat, ignore_index=True)

            Data_table.to_csv(os.path.join(SAVE_DIR, 'Data_table.csv'), index=False)
            LOGGER.info(f'Compiled results saved to {SAVE_DIR}Data_table.csv')

            # Clean up tmp directory
            try:
                subprocess.run(['rm', '-r', tmp_save_dir], check=True)
                LOGGER.info(f'Deleted temporary directory {tmp_save_dir}')
            except Exception as e:
                LOGGER.error(f'Error deleting temporary directory {tmp_save_dir}: {e}')

    # Finalize the profiler if enabled
    if PROFILE:
        LOGGER.info('Main function completed')
        yappi.stop()
        profile_output_path = os.path.join(_ensure_profile_dir(), 'step01_profile.yappi')
        LOGGER.info(f'Writing profile results to {profile_output_path}')
        yappi.get_func_stats().save(profile_output_path, type='pstat')
        LOGGER.info(f'Profile results saved to {profile_output_path}')
