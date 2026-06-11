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
    --profile-dir (str): Directory for storing profiling output.
    --resume: Resume from available checkpoints if present.

Dependencies:
    - pydicom
    - pandas
    - toolbox (custom)
    - DICOM (custom)
"""

# Standard imports
from dataclasses import dataclass
import os
import time
import argparse
import subprocess
import pickle
import random
from functools import partial
from typing import List, Dict, Any, Optional
import logging

# Third-party imports
import pydicom as pyd
import pandas as pd

# Function imports
from multiprocessing import cpu_count
# Custom imports
from toolbox import get_logger, run_function
from DICOM import DICOMextract


@dataclass
class ScanConfig:
    """All runtime configuration for 01_scanDicom."""
    save_dir: str = '/FL_system/data/'
    scan_dir: str = '/FL_system/data/raw/'
    test: Optional[int] = None
    n_test: int = 100
    parallel: bool = False
    n_cpus: int = 0
    profile: bool = False
    sample_pct: float = 0.0
    sample_seed: Optional[int] = None
    checkpoint_dir: Optional[str] = None
    profile_dir: Optional[str] = None
    resume: bool = False
    dir_idx: Optional[int] = None
    dir_list: str = 'dirs_to_process.pkl'


def build_config() -> ScanConfig:
    """Parse CLI arguments and return a ScanConfig instance."""
    parser = argparse.ArgumentParser(description='Extract DICOM data to build Data_table.csv')
    parser.add_argument('--test', nargs='?', const=100, type=int,
                        help='Run in test mode with an optional number of dicom directories to scan (default: 100)')
    parser.add_argument('--multi', '-m', nargs='?', const=max(1, cpu_count()-1), type=int,
                        help='Run with multiprocessing enabled, using provided number of cpus (default: max-1)')
    parser.add_argument('-p', '--profile', action='store_true',
                        help='Run with profiler enabled')
    parser.add_argument('--save_dir', nargs='?', default='/FL_system/data/', type=str,
                        help='Location to save the constructed Data_table.csv (default: /FL_system/data/)')
    parser.add_argument('--scan_dir', nargs='?', default='/FL_system/data/raw/', type=str,
                        help='Location to recursively scan for dicom files (default: /FL_system/data/raw/)')
    parser.add_argument('--dir_idx', type=int,
                        help='Index of the folder to process from dirs_to_process.pkl (for HPC array jobs)')
    parser.add_argument('--dir_list', type=str, default='dirs_to_process.pkl',
                        help='Path to the directory list file (for HPC array jobs)')
    parser.add_argument('--sample-pct', type=float, default=0.0,
                        help='Percent of .dcm files to sample per directory (0 = full scan)')
    parser.add_argument('--sample-seed', type=int, default=None,
                        help='Optional random seed for sampling reproducibility')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to store checkpoint files (default: <SAVE_DIR>/checkpoints/)')
    parser.add_argument('--profile-dir', type=str, default=None,
                        help='Directory to store profiling output (default: <SAVE_DIR>/profiles/)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from available checkpoints if present')
    args = parser.parse_args()

    cfg = ScanConfig(
        save_dir=args.save_dir,
        scan_dir=args.scan_dir,
        test=args.test,
        n_test=args.test if args.test is not None else 100,
        parallel=args.multi is not None,
        n_cpus=args.multi if args.multi is not None else cpu_count() - 1,
        profile=args.profile,
        sample_pct=args.sample_pct,
        sample_seed=args.sample_seed,
        checkpoint_dir=args.checkpoint_dir,
        profile_dir=args.profile_dir,
        resume=args.resume,
        dir_idx=args.dir_idx,
        dir_list=args.dir_list,
    )

    if cfg.sample_seed is not None:
        random.seed(cfg.sample_seed)
    return cfg


# ---------------------------------------------------------------------------
# Logger helper — created once from cfg.save_dir
# ---------------------------------------------------------------------------

def create_logger(cfg: ScanConfig) -> logging.Logger:
    return get_logger('01_scanDicom', f'{cfg.save_dir}/logs/')


# ---------------------------------------------------------------------------
# Checkpoint helpers (use mutable cfg.checkpoint_dir and cfg.save_dir)
# ---------------------------------------------------------------------------

def _ensure_checkpoint_dir(cfg: ScanConfig) -> str:
    if cfg.checkpoint_dir is None:
        cfg.checkpoint_dir = os.path.join(cfg.save_dir, 'checkpoints/')
    try:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    except Exception:
        cfg.checkpoint_dir = cfg.save_dir
    return cfg.checkpoint_dir


def _ensure_profile_dir(cfg: ScanConfig) -> str:
    if cfg.profile_dir is None:
        cfg.profile_dir = os.path.join(cfg.save_dir, 'profiles/')
    try:
        os.makedirs(cfg.profile_dir, exist_ok=True)
    except Exception:
        cfg.profile_dir = os.getcwd()
    return cfg.profile_dir


def save_checkpoint(cfg: ScanConfig, logger: logging.Logger, name: str, obj: Any) -> None:
    d = _ensure_checkpoint_dir(cfg)
    tmp_path = os.path.join(d, f'.{name}.tmp')
    final_path = os.path.join(d, f'{name}.pkl')
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, final_path)
        logger.info(f'Wrote checkpoint: {final_path}')
    except Exception as e:
        logger.error(f'Failed to write checkpoint {final_path}: {e}')


def load_checkpoint(cfg: ScanConfig, logger: logging.Logger, name: str) -> Optional[Any]:
    d = _ensure_checkpoint_dir(cfg)
    path = os.path.join(d, f'{name}.pkl')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f'Loaded checkpoint: {path}')
        return obj
    except Exception as e:
        logger.error(f'Failed to load checkpoint {path}: {e}')
        return None


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _has_dcm_magic(path: str) -> bool:
    """Check for DICM magic marker at offset 128."""
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def _extractDicom_impl(f: str, slice_counts: Dict[str, int] = None) -> Optional[Dict[str, Any]]:
    """Extract DICOM information from a specific file path."""
    logger = logging.getLogger('01_scanDicom')
    try:
        logger.debug(f'Extracting information for file: {f}')
        directory = os.path.dirname(f)
        num_slices = None
        if slice_counts is not None:
            num_slices = slice_counts.get(directory)
        extract = DICOMextract(f, num_slices=num_slices)

        result = {
            'PATH': f,
            'Orientation': extract.Orientation(),
            'ID': extract.ID(),
            'Accession': extract.Accession(),
            'Name': extract.Name(),
            'DATE': extract.Date(),
            'DOB': extract.DOB(),
            'Series_desc': extract.Desc(),
            'Study_desc': extract.StudyDesc(),
            'Anatomic_region': extract.AnatomicRegion(),
            'Modality': extract.Modality(),
            'Part': extract.Part(),
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
        logger.debug(f'Completed extraction for file: {f}')
        return result
    except Exception as e:
        logger.error(f'Error extracting information for file: {f} | {e}')
        return None


def _find_all_dicom_dirs_impl(cfg: ScanConfig, logger: logging.Logger, directory: str,
                                n_test: Optional[int] = None) -> List[str]:
    """Find all directories containing MRI DICOM files."""
    dicom_dirs = []
    n_found = 0
    for root, _, files in os.walk(directory, followlinks=False):
        has_mri = False
        candidates = [fn for fn in files if fn.lower().endswith('.dcm')]
        for fn in candidates:
            file_path = os.path.join(root, fn)
            try:
                dcm = pyd.dcmread(file_path, stop_before_pixels=True, force=False)
                if hasattr(dcm, 'Modality') and dcm.Modality == 'MR':
                    has_mri = True
                    break
            except Exception:
                continue

        if has_mri:
            dicom_dirs.append(root)
            n_found += 1
            if n_test is not None and n_found >= n_test:
                break
            logger.debug(f'Found DICOM files for MRI in {root}')

    if not dicom_dirs:
        logger.warning(f'No directories containing DICOM files found in {directory}')
    else:
        logger.info(f'Found {len(dicom_dirs)} directories containing DICOM files')

    if n_test is not None:
        return dicom_dirs[:n_test]
    return dicom_dirs


def _find_and_select_impl(directory: str,
                          n_test: Optional[int],
                          sample_pct: float,
                          sample_seed: Optional[int],
                          logger: logging.Logger
                          ) -> tuple:
    """Single-pass directory discovery + representative selection.

    Walks the directory tree once. Each .dcm file that is read is checked for
    MR modality (directory confirmation) and used as a series representative
    in the same call, halving the number of ``pyd.dcmread`` invocations on
    directories that contain both MR slices and representative files.

    Returns:
        (mr_dirs, dicom_files, slice_counts)
    """
    mr_dirs = []
    dicom_files = []
    slice_counts = {}
    n_found = 0

    for root, dirs, files in os.walk(directory, followlinks=False):
        dcm_candidates = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_candidates:
            continue

        # Record slice count for every directory with .dcm files
        slice_counts[root] = len(dcm_candidates)

        # ------ Decide whether to sample ----------------------------------
        if sample_pct and sample_pct > 0 and len(dcm_candidates) > 1:
            rng = random.Random(sample_seed) if sample_seed is not None else random
            k = max(1, int(len(dcm_candidates) * (sample_pct / 100.0)))
            if k >= len(dcm_candidates):
                scan_list = dcm_candidates
                fallback_allowed = False
            else:
                scan_list = rng.sample(dcm_candidates, k)
                fallback_allowed = True
        else:
            scan_list = dcm_candidates
            fallback_allowed = False

        # ------ Primary scan: rely on pyd.dcmread() to reject non-DICOM --
        is_mr = False
        found_series = {}

        for fname in scan_list:
            path = os.path.join(root, fname)
            try:
                data = pyd.dcmread(path, stop_before_pixels=True, force=False)
            except Exception:
                continue
            if not is_mr and hasattr(data, 'Modality') and data.Modality == 'MR':
                is_mr = True
            series = getattr(data, 'SeriesNumber', None)
            if series is not None and series not in found_series:
                found_series[series] = path

        # ------ Sampling fallback: full rescan if nothing found ------------
        if fallback_allowed and len(found_series) == 0:
            full_found = {}
            for fname in dcm_candidates:
                path = os.path.join(root, fname)
                try:
                    data = pyd.dcmread(path, stop_before_pixels=True, force=False)
                except Exception:
                    continue
                if not is_mr and hasattr(data, 'Modality') and data.Modality == 'MR':
                    is_mr = True
                series = getattr(data, 'SeriesNumber', None)
                if series is not None and series not in full_found:
                    full_found[series] = path
            found_series = full_found

        # ------ Record MR directories ------------------------------------
        if is_mr:
            mr_dirs.append(root)
            n_found += 1
            for series, path in found_series.items():
                dicom_files.append(path)
            logger.debug(
                f'{root} contains series '
                f'{sorted(found_series.keys())} | '
                f'{len(found_series)} series found'
            )
            if n_test is not None and n_found >= n_test:
                break

    if not mr_dirs:
        logger.warning(
            f'No directories containing DICOM files found in {directory}'
        )
    else:
        logger.info(
            f'Found {len(mr_dirs)} directories containing DICOM files'
        )

    return mr_dirs, dicom_files, slice_counts


def _scan_subdir_worker(subdir: str, sample_pct: float, sample_seed: Optional[int]) -> tuple:
    """Worker for multiprocessing directory scanning.

    Calls `_find_dicom_worker` on its assigned single top-level subdirectory.
    Returns list of (dicom_files, slice_counts).
    """
    return [_find_dicom_worker(subdir, sample_pct, sample_seed)]


def _scan_subdir(topdir: str, min_targets: int = 16):
    """Return a list of disjoint subdirectories that cover the entire tree.
    We gather directories using BFS until we have enough targets. If a directory contains
    files directly, we stop expanding it to keep subtrees disjoint for the os.walk workers."""
    dirs_to_scan = []
    queue = [topdir]

    while queue and (len(dirs_to_scan) + len(queue)) < min_targets:
        curr = queue.pop(0)
        try:
            with os.scandir(curr) as it:
                subdirs = []
                has_files = False
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        subdirs.append(entry.path)
                    elif entry.is_file():
                        if entry.name.lower().endswith('.dcm'):
                            has_files = True

                if has_files:
                    dirs_to_scan.append(curr)
                else:
                    queue.extend(subdirs)
        except Exception:
            pass

    dirs_to_scan.extend(queue)

    if not dirs_to_scan:
        dirs_to_scan = [topdir]

    return dirs_to_scan


def _find_dicom_worker(directory: str, sample_pct: float, sample_seed: Optional[int]) -> tuple:
    """Worker for findDicom — called per directory, accepts only plain args.

    Returns:
        (dicom_files, slice_counts)
    """
    logger = logging.getLogger('01_scanDicom')
    dicom_files = []
    slice_counts = {}

    for root, dirs, files in os.walk(directory, followlinks=False):
        dcm_candidates = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_candidates:
            continue

        # Pre-compute the number of .dcm files in this directory once
        slice_counts[root] = len(dcm_candidates)

        # Decide whether to sample
        if sample_pct and sample_pct > 0 and len(dcm_candidates) > 1:
            rng = random.Random(sample_seed) if sample_seed is not None else random
            k = max(1, int(len(dcm_candidates) * (sample_pct / 100.0)))
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

        # Scan via pyd.dcmread() — exceptions handle non-DICOM files
        for fname in sample_list:
            path = os.path.join(root, fname)
            try:
                data = pyd.dcmread(path, stop_before_pixels=True, force=False)
            except Exception:
                continue
            series = getattr(data, 'SeriesNumber', None)
            if series is not None and series not in found_series:
                found_series[series] = path

        # Sampling fallback: rescan everything if nothing found
        if fallback_allowed and len(found_series) == 0:
            full_found = {}
            for fname in dcm_candidates:
                path = os.path.join(root, fname)
                try:
                    data = pyd.dcmread(path, stop_before_pixels=True, force=False)
                except Exception:
                    continue
                series = getattr(data, 'SeriesNumber', None)
                if series is not None and series not in full_found:
                    full_found[series] = path
            found_series = full_found

        for series, path in found_series.items():
            dicom_files.append(path)

        logger.debug(f'{root} contains series {sorted(found_series.keys())} | {len(found_series)} series found')

    return [dicom_files, slice_counts]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(cfg: ScanConfig, logger: logging.Logger, out_name: str = 'Data_table.csv') -> None:
    """Main execution logic for scanning and extracting DICOM data."""
    scan_dir = cfg.scan_dir
    save_dir = cfg.save_dir

    assert os.path.exists(scan_dir), f'SCAN_DIR {scan_dir} does not exist. Please provide a valid directory.'

    # Create the save directory if it does not exist
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            logger.info(f'Created directory {save_dir}')
        except Exception as e:
            logger.error(f'Error creating directory {save_dir}: {e}')

    # Print the current configuration
    logger.info('Starting scanDicom: Step 01')
    logger.info(f'SCAN_DIR: {scan_dir}')
    logger.info(f'SAVE_DIR: {save_dir}')
    logger.info(f'PARALLEL: {cfg.parallel}')
    if cfg.profile:
        logger.info('Profiling is enabled')

    # Check if the output already exists to avoid redundant processing
    if os.path.exists(os.path.join(save_dir, out_name)):
        logger.error(f'{out_name} already exists. Skipping step 01')
        logger.error(f'To re-run this step, delete the existing {out_name} file')
        return

    # --- Combined scan & representative selection (single walk) ------
    logger.info('Finding all directories containing DICOM files')
    test_mode = cfg.test is not None
    n_test_val = cfg.n_test if test_mode else None
    if test_mode:
        logger.info(f'Running in test mode with a maximum of {cfg.n_test} directories')

    # Attempt to resume from checkpoint
    combined_result = None
    if cfg.resume:
        try:
            combined_result = load_checkpoint(cfg, logger, 'scan_and_select')
        except Exception:
            combined_result = None

    if combined_result is None:
        if cfg.parallel:
            # Parallel scan: we parallelize the walk by getting immediate subdirectories
            # and dispatching multiprocessing workers to walk those subtrees independently.
            target_dirs = _scan_subdir(scan_dir, min_targets=cfg.n_cpus * 4)
            logger.info(f'Found {len(target_dirs)} branch directories to scan in parallel')

            if len(target_dirs) > 1:
                worker_results = run_function(
                    logger, _scan_subdir_worker, target_dirs,
                    Parallel=True, P_type='hybrid', N_CPUS=cfg.n_cpus,
                    sample_pct=cfg.sample_pct, sample_seed=cfg.sample_seed,
                )
            else:
                worker_results = [run_function(
                    logger, _scan_subdir_worker, target_dirs,
                    Parallel=False, P_type='thread', N_CPUS=1,
                    sample_pct=cfg.sample_pct, sample_seed=cfg.sample_seed,
                )]
            # Flatten results from all workers
            dicom_files = [f for sublist in worker_results for files, _ in sublist for f in files]
            slice_counts = {}
            for sublist in worker_results:
                for _, counts in sublist:
                    slice_counts.update(counts)
            mr_dirs = list(set(os.path.dirname(f) for f in dicom_files))

            # Apply --test as an output limiter (never alter execution path)
            if n_test_val is not None:
                dicom_files = dicom_files[:n_test_val]
                slice_counts = {k: v for k, v in slice_counts.items() if k in set(os.path.dirname(f) for f in dicom_files)}
                mr_dirs = list(set(os.path.dirname(f) for f in dicom_files))

        else:
            combined_result = _find_and_select_impl(
                directory=scan_dir,
                n_test=n_test_val,
                sample_pct=cfg.sample_pct,
                sample_seed=cfg.sample_seed,
                logger=logger,
            )
            mr_dirs, dicom_files, slice_counts = combined_result

    # Fallback to two-pass if combined pass found nothing and we haven't already
    if not dicom_files:
        # Try legacy checkpoint for MR dirs only
        if cfg.resume:
            try:
                dicom_dirs = load_checkpoint(cfg, logger, 'dirs')
            except Exception:
                dicom_dirs = None
        if dicom_dirs is None:
            dicom_dirs = _find_all_dicom_dirs_impl(
                cfg, logger, scan_dir, n_test=n_test_val)
        mr_dirs = dicom_dirs
        worker_results = run_function(
            logger, _find_dicom_worker, dicom_dirs,
            Parallel=cfg.parallel, P_type='hybrid', N_CPUS=cfg.n_cpus,
            sample_pct=cfg.sample_pct, sample_seed=cfg.sample_seed,
        )
        dicom_files = [f for files, _ in worker_results for f in files]
        slice_counts = {}
        for _, counts in worker_results:
            slice_counts.update(counts)

    if not slice_counts:
        slice_counts = {}

    logger.info('Analyzing DICOM directories')
    logger.info(f'Found {len(dicom_files)} dicom files in the input directory')

    # Extract the dicom information
    logger.info('Extracting information from dicom files')

    # Attempt to resume extracted info from checkpoint
    info_list = None
    if cfg.resume:
        try:
            info_list = load_checkpoint(cfg, logger, 'info')
        except Exception:
            info_list = None

    if info_list is None:
        extract_partial = partial(_extractDicom_impl, slice_counts=slice_counts)
        info_list = run_function(
            logger, extract_partial, dicom_files,
            Parallel=cfg.parallel, P_type='hybrid', N_CPUS=cfg.n_cpus,
        )
        try:
            save_checkpoint(cfg, logger, 'info', info_list)
        except Exception:
            pass

    Data_table = pd.DataFrame(info_list)

    # Write Data_table to CSV atomically to prevent partial writes
    out_path = os.path.join(save_dir, out_name)
    tmp_out = out_path + '.tmp'
    try:
        Data_table.to_csv(tmp_out, index=False)
        os.replace(tmp_out, out_path)
    except Exception as e:
        logger.error(f'Failed to write output CSV {out_path}: {e}')
    logger.info(f'DICOM information extraction completed and saved to {out_name}')
    # Removing checkpoint files after successful completion
    clear_checkpoint_files = ['dirs', 'dicom_files', 'info', 'scan_and_select']
    for chk in clear_checkpoint_files:
        chk_path = os.path.join(_ensure_checkpoint_dir(cfg), f'{chk}.pkl')
        if os.path.exists(chk_path):
            try:
                os.remove(chk_path)
                logger.info(f'Removed checkpoint file: {chk_path}')
            except Exception as e:
                logger.error(f'Error removing checkpoint file {chk_path}: {e}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cfg = build_config()
    logger = create_logger(cfg)

    # Start the profiler if enabled
    if cfg.profile:
        import yappi
        logger.info('Profiling enabled')
        yappi.start()
        logger.info('Starting main function')

    # Check if running in single directory mode (HPC array job)
    if cfg.dir_idx is None:
        # Normal execution
        main(cfg, logger)

    else:
        assert os.path.exists(cfg.dir_list), f'Directory list file {cfg.dir_list} does not exist'

        tmp_save_dir = os.path.join(cfg.save_dir, 'tmp/')
        os.makedirs(tmp_save_dir, exist_ok=True)

        with open(cfg.dir_list, 'rb') as f:
            dirs = pickle.load(f)

        selected_dir = dirs[cfg.dir_idx]
        cfg.scan_dir = selected_dir
        cfg.save_dir = tmp_save_dir
        logger.info(f'Processing single directory: {cfg.dir_idx}')

        # Run main for this specific directory
        main(cfg, logger, out_name=f'Data_table_{cfg.dir_idx}.csv')

        # If this is the last job in the array, compile all results
        if cfg.dir_idx == len(dirs) - 1:
            logger.info('Last script, compiling results')
            tables = []
            while len(tables) < len(dirs):
                logger.info('Waiting for all tables to be compiled')
                time.sleep(5)
                tables = [t for t in os.listdir(tmp_save_dir) if t.endswith('.csv')]

            logger.info('All tables present, compiling...')
            tables_to_concat = [pd.read_csv(os.path.join(tmp_save_dir, t)) for t in tables]
            combined = pd.concat(tables_to_concat, ignore_index=True)

            final_save_dir = os.path.dirname(tmp_save_dir)
            combined.to_csv(os.path.join(final_save_dir, 'Data_table.csv'), index=False)
            logger.info(f'Compiled results saved to {os.path.join(final_save_dir, "Data_table.csv")}')

            try:
                subprocess.run(['rm', '-r', tmp_save_dir], check=True)
                logger.info(f'Deleted temporary directory {tmp_save_dir}')
            except Exception as e:
                logger.error(f'Error deleting temporary directory {tmp_save_dir}: {e}')

    # Finalize the profiler if enabled
    if cfg.profile:
        logger.info('Main function completed')
        yappi.stop()
        profile_output_path = os.path.join(_ensure_profile_dir(cfg), 'step01_profile.yappi')
        logger.info(f'Writing profile results to {profile_output_path}')
        yappi.get_func_stats().save(profile_output_path, type='pstat')
        logger.info(f'Profile results saved to {profile_output_path}')


# ------ End of file ------