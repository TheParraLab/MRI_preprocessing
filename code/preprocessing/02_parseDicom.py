"""
DICOM Parsing Script
====================

This script filters, splits, and orders DICOM scan data extracted from Step 01.
It isolates the primary sequence of scans, removes derived images, and handles
temporal ordering based on trigger/acquisition times.

Pipeline steps:
    1. Filter scans (remove computed images, isolate primary sequences, handle DISCO scans)
    2. Split scans with multiple post-contrast images in a single directory
    3. Order scans by trigger time within each session
    4. Create symbolic links for temporary file relocations

Usage:
    python 02_parseDicom.py --save_dir /path/to/output [--multi] [--filter-only] [--force] [--profile]

Arguments:
    --save_dir (str): Directory to save output tables and logs.
    --load_table (str): Path to the input Data_table.csv from Step 01.
    --multi (int): Enable multiprocessing with specified CPU count (default: max-1).
    --filter-only: Run only the filtering step, skip ordering.
    --force: Overwrite existing output files without prompting.
    --profile: Enable yappi profiling.
    --dir_idx (int): Index for HPC array jobs.
    --dir_list (str): Path to directory list file for HPC jobs.

Dependencies:
    - pandas, numpy
    - toolbox (custom)
    - DICOM (custom)
"""

# Standard imports
from dataclasses import dataclass, field, replace
import os
import argparse
import time
import re
import random
import json
import pickle
import logging
from multiprocessing import cpu_count
from typing import Any, Optional
from collections import defaultdict
import sys
import shutil
import functools

# Third-party imports
import numpy as np
import pandas as pd
try:
    import yappi
except ImportError:
    yappi = None

# Custom imports
from toolbox import get_logger, run_function
from DICOM import DICOMfilter, DICOMorder, DICOMsplit


@dataclass
class ParseConfig:
    """All runtime configuration for 02_parseDicom."""
    save_dir: str = '/FL_system/data/'
    load_table: str = '/FL_system/data/Data_table.csv'
    dir_list: str = 'dirs_to_process.txt'
    dir_idx: Optional[int] = None
    filter_only: bool = False
    force: bool = False
    parallel: bool = False
    n_cpus: int = 0
    profile: bool = False
    n_test: int = 25
    export_fully_removed: bool = False
    computed_flags: list = field(default_factory=lambda: ['slope', 'sub', 'subtract'])
    description_flags: list = field(default_factory=lambda: ['loc', 'pjn', 'calib'])
    out_name: str = 'Data_table_timing.csv'
    resume: bool = False
    filter_batch_size: int = 10
    target: Optional[str] = None
    test: bool = False


def build_config() -> ParseConfig:
    """Parse CLI arguments and return a ParseConfig instance."""
    parser = argparse.ArgumentParser(description='Parse DICOM data: filter, split, and order scans')
    parser.add_argument('--multi', '-m', nargs='?', const=max(1, cpu_count()-1), type=int,
                        help='Run with multiprocessing enabled (default: max-1 CPUs)')
    parser.add_argument('--save_dir', type=str, default='/FL_system/data/',
                        help='Directory to save the updated tables (default: /FL_system/data/)')
    parser.add_argument('--load_table', type=str, default='/FL_system/data/Data_table.csv',
                        help='Path to the input Data_table.csv (default: /FL_system/data/Data_table.csv)')
    parser.add_argument('--dir_idx', type=int,
                        help='Index of the folder to process from dirs_to_process.txt (for HPC array jobs)')
    parser.add_argument('--dir_list', type=str, default='dirs_to_process.txt',
                        help='Path to the directory list file (for HPC array jobs)')
    parser.add_argument('--filter_only', action='store_true',
                        help='Run only the filtering step without ordering')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files without prompting')
    parser.add_argument('--profile', action='store_true',
                        help='Run with profiler enabled')
    parser.add_argument('--resume', action='store_true',
                        help='Resume filtering from checkpoint if available')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of sessions per batch before saving checkpoint (default: 10)')
    args = parser.parse_args()

    cfg = ParseConfig(
        save_dir=args.save_dir,
        load_table=args.load_table,
        dir_list=args.dir_list,
        dir_idx=args.dir_idx,
        filter_only=args.filter_only,
        force=args.force,
        parallel=args.multi is not None,
        n_cpus=args.multi if args.multi is not None else cpu_count() - 1,
        profile=args.profile,
        resume=args.resume,
        filter_batch_size=args.batch_size,
    )
    return cfg


def create_logger(cfg: ParseConfig) -> logging.Logger:
    """Create logger instance from config."""
    logger = logging.getLogger('02_parseDicom')
    logger.handlers.clear()
    return get_logger('02_parseDicom', f'{cfg.save_dir}/logs/')

# ------ -- --- ----------------------------- ----- ----------------- --- ---
# Utility helpers
# ------ ---------------------------------- --- - -------------- --- --- ---

def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV atomically using tmp + os.replace."""
    tmp_path = path + '.tmp'
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ------ --------------------------- ---- - --------------- --- -- --------
# Filter checkpoint helpers
# ------ ---------------------------------- --- - -------------- --- --- ---

CHECKPOINT_DIR = '.filter_checkpoint'

def _checkpoint_path(cfg: ParseConfig) -> str:
    """Return path to the checkpoint directory."""
    return os.path.join(cfg.save_dir, CHECKPOINT_DIR)


def _save_filter_checkpoint(
    cfg: ParseConfig,
    logger: logging.Logger,
    completed_ids: list,
    results: list,
    removed: list,
) -> None:
    """Save filter progress atomically: completed session IDs, results, removed entries."""
    cp_dir = _checkpoint_path(cfg)
    os.makedirs(cp_dir, exist_ok=True)

    meta_path = os.path.join(cp_dir, 'meta.json.tmp')
    meta = {
        'completed_ids': completed_ids,
        'total_results': len(results),
        'total_removed': len(removed),
    }

    results_path = os.path.join(cp_dir, 'results.pkl.tmp')
    removed_path = os.path.join(cp_dir, 'removed.pkl.tmp')

    try:
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        os.replace(meta_path, os.path.join(cp_dir, 'meta.json'))

        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        os.replace(results_path, os.path.join(cp_dir, 'results.pkl'))

        with open(removed_path, 'wb') as f:
            pickle.dump(removed, f)
        os.replace(removed_path, os.path.join(cp_dir, 'removed.pkl'))

        logger.info(f'Checkpoint saved: {len(completed_ids)} sessions done')
    except Exception as e:
        logger.error(f'Failed to write checkpoint: {e}')


def _load_filter_checkpoint(
    cfg: ParseConfig,
    logger: logging.Logger,
) -> tuple:
    """Load filter checkpoint if available. Returns (completed_ids, results, removed) or (None, None, None)."""
    cp_dir = _checkpoint_path(cfg)
    meta_path = os.path.join(cp_dir, 'meta.json')
    results_path = os.path.join(cp_dir, 'results.pkl')
    removed_path = os.path.join(cp_dir, 'removed.pkl')

    if not all(os.path.exists(p) for p in [meta_path, results_path, removed_path]):
        logger.info('No valid filter checkpoint found')
        return None, None, None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        with open(removed_path, 'rb') as f:
            removed = pickle.load(f)

        logger.info(
            f'Loaded filter checkpoint: {meta["total_results"]} results, '
            f'{meta["total_removed"]} removed entries'
        )
        return meta['completed_ids'], results, removed
    except Exception as e:
        logger.error(f'Failed to load checkpoint: {e}')
        return None, None, None


def _remove_checkpoint(cfg: ParseConfig, logger: logging.Logger) -> None:
    """Clean up the checkpoint directory after successful completion."""
    cp_dir = _checkpoint_path(cfg)
    if os.path.exists(cp_dir):
        try:
            shutil.rmtree(cp_dir)
            logger.info('Removed checkpoint directory')
        except Exception as e:
            logger.error(f'Failed to remove checkpoint directory: {e}')


SPLIT_CHECKPOINT_DIR = '.split_checkpoint'

def _split_checkpoint_path(cfg: ParseConfig) -> str:
    return os.path.join(cfg.save_dir, SPLIT_CHECKPOINT_DIR)


def _save_split_checkpoint(
    cfg: ParseConfig,
    logger: logging.Logger,
    completed_ids: list,
    results: list,
    redirections: list,
) -> None:
    cp_dir = _split_checkpoint_path(cfg)
    os.makedirs(cp_dir, exist_ok=True)

    meta_path = os.path.join(cp_dir, 'meta.json.tmp')
    meta = {
        'completed_ids': completed_ids,
        'total_results': len(results),
        'total_redirections': len(redirections),
    }
    results_path = os.path.join(cp_dir, 'results.pkl.tmp')
    redirect_path = os.path.join(cp_dir, 'redirections.pkl.tmp')

    try:
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        os.replace(meta_path, os.path.join(cp_dir, 'meta.json'))

        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        os.replace(results_path, os.path.join(cp_dir, 'results.pkl'))

        with open(redirect_path, 'wb') as f:
            pickle.dump(redirections, f)
        os.replace(redirect_path, os.path.join(cp_dir, 'redirections.pkl'))

        logger.info(f'Split checkpoint saved: {len(completed_ids)} sessions done')
    except Exception as e:
        logger.error(f'Failed to write split checkpoint: {e}')


def _load_split_checkpoint(
    cfg: ParseConfig,
    logger: logging.Logger,
) -> tuple:
    cp_dir = _split_checkpoint_path(cfg)
    meta_path = os.path.join(cp_dir, 'meta.json')
    results_path = os.path.join(cp_dir, 'results.pkl')
    redirect_path = os.path.join(cp_dir, 'redirections.pkl')

    if not all(os.path.exists(p) for p in [meta_path, results_path, redirect_path]):
        logger.info('No valid split checkpoint found')
        return None, None, None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        with open(redirect_path, 'rb') as f:
            redirections = pickle.load(f)

        logger.info(
            f'Loaded split checkpoint: {meta["total_results"]} results, '
            f'{meta["total_redirections"]} redirections'
        )
        return meta['completed_ids'], results, redirections
    except Exception as e:
        logger.error(f'Failed to load split checkpoint: {e}')
        return None, None, None


def _remove_split_checkpoint(cfg: ParseConfig, logger: logging.Logger) -> None:
    cp_dir = _split_checkpoint_path(cfg)
    if os.path.exists(cp_dir):
        try:
            shutil.rmtree(cp_dir)
            logger.info('Removed split checkpoint directory')
        except Exception as e:
            logger.error(f'Failed to remove split checkpoint directory: {e}')


ORDER_CHECKPOINT_DIR = '.order_checkpoint'

def _order_checkpoint_path(cfg: ParseConfig) -> str:
    return os.path.join(cfg.save_dir, ORDER_CHECKPOINT_DIR)


def _save_order_checkpoint(
    cfg: ParseConfig,
    logger: logging.Logger,
    completed_ids: list,
    results: list,
) -> None:
    cp_dir = _order_checkpoint_path(cfg)
    os.makedirs(cp_dir, exist_ok=True)

    meta_path = os.path.join(cp_dir, 'meta.json.tmp')
    meta = {
        'completed_ids': completed_ids,
        'total_results': len(results),
    }
    results_path = os.path.join(cp_dir, 'results.pkl.tmp')

    try:
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        os.replace(meta_path, os.path.join(cp_dir, 'meta.json'))

        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        os.replace(results_path, os.path.join(cp_dir, 'results.pkl'))

        logger.info(f'Order checkpoint saved: {len(completed_ids)} sessions done')
    except Exception as e:
        logger.error(f'Failed to write order checkpoint: {e}')


def _load_order_checkpoint(
    cfg: ParseConfig,
    logger: logging.Logger,
) -> tuple:
    cp_dir = _order_checkpoint_path(cfg)
    meta_path = os.path.join(cp_dir, 'meta.json')
    results_path = os.path.join(cp_dir, 'results.pkl')

    if not all(os.path.exists(p) for p in [meta_path, results_path]):
        logger.info('No valid order checkpoint found')
        return None, None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        with open(results_path, 'rb') as f:
            results = pickle.load(f)

        logger.info(
            f'Loaded order checkpoint: {meta["total_results"]} results'
        )
        return meta['completed_ids'], results
    except Exception as e:
        logger.error(f'Failed to load order checkpoint: {e}')
        return None, None


def _remove_order_checkpoint(cfg: ParseConfig, logger: logging.Logger) -> None:
    cp_dir = _order_checkpoint_path(cfg)
    if os.path.exists(cp_dir):
        try:
            shutil.rmtree(cp_dir)
            logger.info('Removed order checkpoint directory')
        except Exception as e:
            logger.error(f'Failed to remove order checkpoint directory: {e}')


# ------
# Pipeline workers  (accept plain args for run_function compatibility)
# ------

def _filter_worker(data_subset: pd.DataFrame, save_dir: str, computed_flags: list,
                   description_flags: list, logger: logging.Logger) -> tuple:
    """Worker for filter step — called per session subset."""
    data_subset = data_subset.reset_index(drop=True)
    base, last = os.path.split(save_dir.rstrip('/'))
    tmp_save = os.path.join(base, 'tmp_data') if last == 'tmp' else save_dir
    dicom_filter = DICOMfilter(data_subset, logger=logger, tmp_save=tmp_save)
    dicom_filter.Types(computed_flags)
    dicom_filter.Description(description_flags)

    if len(dicom_filter.dicom_table) < 2:
        dicom_filter.logger.error(f'Not enough scans for {dicom_filter.Session_ID}, removing...')
        dicom_filter.removed['Insufficient_Samples'] = dicom_filter.dicom_table.copy()
        dicom_filter.dicom_table = pd.DataFrame(columns=dicom_filter.dicom_table.columns)
        return dicom_filter.dicom_table, dicom_filter.removed, dicom_filter.temporary_relocations

    disco_pattern = re.compile(r'disco', re.IGNORECASE)
    dicom_filter.dicom_table['IS_DISCO'] = dicom_filter.dicom_table['Series_desc'].str.contains(
        disco_pattern, na=False)

    if dicom_filter.dicom_table['IS_DISCO'].sum() > 0:
        dicom_filter.logger.debug(f'DISCO scans detected | {dicom_filter.Session_ID}')
        dicom_filter.disco_table = dicom_filter.dicom_table.loc[dicom_filter.dicom_table['IS_DISCO'] == True]
        dicom_filter.dicom_table = dicom_filter.dicom_table.loc[dicom_filter.dicom_table['IS_DISCO'] == False]
        if len(dicom_filter.dicom_table) > 2:
            dicom_filter.logger.debug(f'Will attempt to determine steady state sequence | {dicom_filter.Session_ID}')
            if not dicom_filter.isolate_sequence():
                dicom_filter.logger.debug(f'Failed to isolate steady state sequence | {dicom_filter.Session_ID}')
                dicom_filter.logger.debug(f'Attempting to solve with disco | {dicom_filter.Session_ID}')
                dicom_filter.dicom_table = dicom_filter.disco_table
                if not dicom_filter.isolate_sequence():
                    dicom_filter.logger.debug(f'Failed to isolate sequence using DISCO | {dicom_filter.Session_ID}')
                    dicom_filter.removed['Sequence_Failure'] = dicom_filter.dicom_table.copy()
                    dicom_filter.dicom_table = pd.DataFrame(columns=dicom_filter.dicom_table.columns)
                else:
                    dicom_filter.logger.debug(f'Sequence isolated using DISCO | {dicom_filter.Session_ID}')
            else:
                dicom_filter.logger.debug(f'Sequence isolated using steady state information | {dicom_filter.Session_ID}')
        elif len(dicom_filter.disco_table) > 2:
            dicom_filter.logger.debug(
                f'Forced to utilize DISCO, not enough steady state information '
                f'[{len(dicom_filter.dicom_table)}] | {dicom_filter.Session_ID}')
            dicom_filter.dicom_table = dicom_filter.disco_table
            if not dicom_filter.isolate_sequence():
                dicom_filter.logger.debug(f'Failed to isolate sequence using DISCO | {dicom_filter.Session_ID}')
                dicom_filter.removed['Sequence_Failure'] = dicom_filter.dicom_table.copy()
                dicom_filter.dicom_table = pd.DataFrame(columns=dicom_filter.dicom_table.columns)
            else:
                dicom_filter.logger.debug(f'Sequence isolated using DISCO | {dicom_filter.Session_ID}')
        else:
            dicom_filter.logger.error(
                f'Not enough scans to identify sequence [DISCO or SS] | {dicom_filter.Session_ID}')
            dicom_filter.removed['Sequence_Failure'] = pd.concat(
                [dicom_filter.dicom_table, dicom_filter.disco_table])
            dicom_filter.dicom_table = pd.DataFrame(columns=dicom_filter.dicom_table.columns)
    else:
        dicom_filter.logger.debug(f'No DISCO scans detected | {dicom_filter.Session_ID}')
        if dicom_filter.isolate_sequence():
            dicom_filter.logger.debug(
                f'Sequence isolated using steady state information | {dicom_filter.Session_ID}')
        else:
            dicom_filter.logger.debug(
                f'Failed to isolate sequence using steady state information | {dicom_filter.Session_ID}')
            dicom_filter.removed['Sequence_Failure'] = dicom_filter.dicom_table.copy()
            dicom_filter.dicom_table = pd.DataFrame(columns=dicom_filter.dicom_table.columns)

    session_id = data_subset['SessionID'].values[0]
    if len(dicom_filter.dicom_table) == 0:
        logger.error(f'No scans remaining after filtering for {session_id}')

    return dicom_filter.dicom_table, dicom_filter.removed, dicom_filter.temporary_relocations


def _order_worker(data_subset: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Worker for ordering step — called per session subset."""
    data_subset = data_subset.reset_index(drop=True)
    session_id = data_subset['SessionID'].values[0]
    order = DICOMorder(data_subset, logger=logger)
    order.order('TriTime', secondary_param='AcqTime')
    if order.dicom_table.empty:
        logger.error(f'No scans remaining after ordering for {session_id}')
        return order.dicom_table
    order.findPre()
    return order.dicom_table


def _split_worker(data_subset: pd.DataFrame, logger: logging.Logger) -> tuple:
    """Worker for splitting step — called per session subset."""
    data_subset = data_subset.reset_index(drop=True)
    splitter = DICOMsplit(data_subset, logger=logger)
    if splitter.SCAN:
        if splitter.scan_complete:
            splitter.load_scan()
        else:
            splitter.scan_all()
        splitter.sort_scans()
        return splitter.dicom_table, splitter.temporary_relocations
    return data_subset, []


def _save_removal_worker(tup: tuple, save_dir: str) -> None:
    """Worker for saving removal logs — called per category."""
    key, item = tup
    out_path = os.path.join(save_dir, 'removal_log', f'Removed_{key}.csv')
    try:
        item.to_csv(out_path, index=False)
    except Exception:
        pass


def _relocate_worker(commands: list, relocations: list, logger: logging.Logger) -> None:
    """Worker for symlinking temporary file relocations."""
    logger.debug(f'Relocate called with {len(commands)} commands')
    logger.debug(f'Current relocations: {len(relocations)}')
    logger.debug(f'First command: {commands[0] if commands else "None"}')
    if not commands:
        logger.warning('No commands supplied to relocate')
        return
    destinations = list(set(cmd[1] for cmd in commands))
    parent_dirs = list(set(os.path.dirname(d) for d in destinations))
    for dest_dir in parent_dirs:
        os.makedirs(dest_dir, exist_ok=True)
    for command in commands:
        logger.debug(f'Linking {command[0]} to {command[1]}')
        src_path = os.path.abspath(command[0])
        dest_path = command[1]
        if os.path.exists(dest_path) or os.path.islink(dest_path):
            os.remove(dest_path)
        os.symlink(src_path, dest_path)
    # ---------------------------------------------------------------------------
# Aggregation helpers (no globals)
# ---------------------------------------------------------------------------

def _init_data_table(load_table: str, target: Optional[str],
                     logger: logging.Logger) -> tuple:
    """Load and prepare the data table, return (table, removed_dict)."""
    data_table = pd.read_csv(load_table, low_memory=False)
    if target is not None:
        try:
            data_table = data_table[data_table['ID'] == target]
            logger.info(f'Filtering data for target ID: {target}')
        except Exception as e:
            logger.error(f'Error filtering data for target ID {target}: {e}')
            raise
    data_table['SessionID'] = data_table['ID'] + '_' + data_table['DATE'].astype(str)
    removed_tables = defaultdict(pd.DataFrame)
    return data_table, removed_tables


def _aggregate_removed(removed_tables: dict, removed_list: list) -> None:
    """Concatenate per-worker removal dicts into the accumulator."""
    buffer = defaultdict(list)
    for removed_dict in removed_list:
        for key, value in removed_dict.items():
            buffer[key].append(value)
    for key, df_list in buffer.items():
        removed_tables[key] = pd.concat([removed_tables[key], pd.concat(df_list, ignore_index=True)], ignore_index=True)


def _normalize_bool_cols(data_table: pd.DataFrame) -> pd.DataFrame:
    """Normalize Pre_scan and Post_scan columns to proper booleans."""
    data_table.loc[data_table['Pre_scan'].isin([True, 'True', 'true', 1, '1']), 'Pre_scan'] = True
    data_table.loc[data_table['Pre_scan'].isin([False, 'False', 'false', 0, '0']), 'Pre_scan'] = False
    data_table.loc[data_table['Post_scan'].isin([True, 'True', 'true', 1, '1']), 'Post_scan'] = True
    data_table.loc[data_table['Post_scan'].isin([False, 'False', 'false', 0, '0']), 'Post_scan'] = False
    return data_table

#############################
## Main script
#############################

def main(cfg: ParseConfig, logger: logging.Logger) -> None:
    """
    Main orchestration function for parsing DICOM data.

    Sequentially filters, splits, and orders DICOM scan sequences, writing
    intermediate checkpoints and final output CSV.

    Args:
        cfg: ParseConfig dataclass with all runtime parameters.
        logger: Configured logger instance.
    """
    # -- Setup ---------------------------------------------------------------
    os.makedirs(cfg.save_dir, exist_ok=True)

    logger.info('Starting parseDicom: Step 02')
    logger.info(f'SAVE_DIR        : {cfg.save_dir}')
    logger.info(f'COMPUTED_FLAGS  : {cfg.computed_flags}')
    logger.info(f'DESCRIPTION_FLG : {cfg.description_flags}')
    logger.info(f'PARALLEL        : {cfg.parallel}')
    logger.info(f'PROFILE         : {cfg.profile}')
    logger.info(f'FILTER_ONLY     : {cfg.filter_only}')
    logger.info(f'FORCE           : {cfg.force}')
    logger.info(f'TEST            : {cfg.test}')
    logger.info(f'N_TEST          : {cfg.n_test}')
    logger.info(f'EXPORT_FULLY_REMOVED: {cfg.export_fully_removed}')

    # -- Overwrite guard -----------------------------------------------------
    out_path = os.path.join(cfg.save_dir, cfg.out_name)
    if os.path.exists(out_path):
        if cfg.force:
            logger.info(f'{cfg.out_name} already exists -- overwriting (--force)')
        else:
            logger.warning(f'{cfg.out_name} already exists')
            if sys.stdin.isatty() == False:
                logger.warning('Running in non-interactive mode, skipping prompt and exiting to avoid overwrite')
                logger.warning('To force overwrite, use the --force flag.')
                return
            try:
                answer = input('Would you like to reprocess? [Y/n]: ')
            except (EOFError, KeyboardInterrupt):
                logger.warning('No input received, aborting.')
                logger.warning('To force overwrite without prompt, use the --force flag.')
                return
            if answer.lower() != 'y':
                logger.info('Stopping processing.')
                return

    # -- Init data table --------------------------------------------------
    Data_table, removed_tables = _init_data_table(cfg.load_table, cfg.target, logger)
    Iden_uniq = np.unique(Data_table['SessionID'])
    PRE_TABLE = Data_table.copy()

    if cfg.test:
        Iden_uniq = Iden_uniq[:cfg.n_test]
        logger.info(f'Running in test mode with {cfg.n_test} sessions')

    if cfg.parallel:
        logger.debug('Running in parallel mode')

    # -- Filtering step --------------------------------------------------
    filter_path = os.path.join(cfg.save_dir, 'Data_table_filtered.csv')
    temporary_relocation = []

    if not os.path.exists(filter_path):
        logger.info('No filtered table found, starting filtering process')

        # Try to resume from checkpoint
        completed_ids = []
        all_results = []
        all_removed = []

        if cfg.resume:
            completed_ids, all_results, all_removed = _load_filter_checkpoint(cfg, logger)
            if completed_ids is not None:
                logger.info(f'Resuming from checkpoint: {len(completed_ids)} sessions already filtered')
            else:
                cfg.resume = False

        # Build work queue (exclude already-completed sessions if resuming)
        if completed_ids:
            completed_set = set(completed_ids)
            Data_subsets = [
                group.copy()
                for sid, group in Data_table.groupby('SessionID')
                if sid in Iden_uniq and sid not in completed_set
            ]
        else:
            Data_subsets = [group.copy() for _, group in Data_table.groupby('SessionID')]
            random.shuffle(Data_subsets)

        if not Data_subsets:
            logger.info('All sessions already processed or no data to filter')
            Data_table = pd.concat(all_results).reset_index(drop=True) if all_results else pd.DataFrame()
            _aggregate_removed(removed_tables, all_removed)
        else:
            logger.info(f'Processing {len(Data_subsets)} session(s)')

            filter_fn = functools.partial(
                _filter_worker,
                save_dir=cfg.save_dir,
                computed_flags=cfg.computed_flags,
                description_flags=cfg.description_flags,
                logger=logger,
            )

            batch_size = cfg.filter_batch_size
            for batch_start in range(0, len(Data_subsets), batch_size):
                batch = Data_subsets[batch_start:batch_start + batch_size]
                logger.info(
                    f'Filtering batch {batch_start // batch_size + 1}: '
                    f'{batch_start + 1}-{min(batch_start + batch_size, len(Data_subsets))} '
                    f'of {len(Data_subsets)} sessions'
                )

                batch_results, batch_removed, batch_temp_rels = run_function(
                    logger, filter_fn, batch,
                    Parallel=cfg.parallel, P_type='process',
                )

                batch_results = [df for df in batch_results if not df.empty]
                all_results.extend(batch_results)
                all_removed.extend(batch_removed)
                temporary_relocation.extend(batch_temp_rels)

                # Track completed session IDs
                for df in batch_results:
                    for sid in df['SessionID'].unique():
                        completed_ids.append(sid)
                for subset in batch:
                    sid = subset['SessionID'].values[0]
                    if sid not in completed_ids:
                        completed_ids.append(sid)

                # Save checkpoint after each batch
                _save_filter_checkpoint(cfg, logger, completed_ids, all_results, all_removed)

            # Final assembly
            results = [df for df in all_results if not df.empty]
            Data_table = pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()

            _aggregate_removed(removed_tables, all_removed)

            # Clean up checkpoint on success
            _remove_checkpoint(cfg, logger)

    else:
        logger.info('Filtered table found, loading filtered data')
        Data_table = pd.read_csv(filter_path, low_memory=False)

    Iden_uniq_after = Data_table['SessionID'].unique() if not Data_table.empty else []

    Iden_uniq_after_clean = []
    for sid in Iden_uniq_after:
        if sid[-2:] in ('_a', '_b', '_l', '_r'):
            Iden_uniq_after_clean.append(sid[:-2])
        else:
            Iden_uniq_after_clean.append(sid)
    Iden_uniq_after_clean = list(set(Iden_uniq_after_clean))

    logger.info('Filtering Results:')
    logger.info(f'Initial number of unique sessions: {len(Iden_uniq)}')
    logger.info(f'Final number of unique sessions  : {len(Iden_uniq_after_clean)}')
    logger.info(f'Final number of sessions (w/ lat): {len(Iden_uniq_after)}')
    logger.info(f'Removed sessions                 : {len(Iden_uniq) - len(Iden_uniq_after_clean)}')

    for key, value in removed_tables.items():
        logger.info(f'=== {key} ===')
        rem_id = value['SessionID'].unique()
        gone_id = set(rem_id) - set(Iden_uniq_after_clean)
        logger.info(f'  Sessions missing from output: {len(gone_id)}')
        logger.info(f'  Scans removed               : {len(value)}')

    Data_table = _normalize_bool_cols(Data_table)

    if not os.path.exists(filter_path):
        logger.info(f'Saving filtered data to {filter_path}')
        _atomic_write_csv(Data_table, filter_path)

        os.makedirs(os.path.join(cfg.save_dir, 'removal_log'), exist_ok=True)
        save_fn = functools.partial(_save_removal_worker, save_dir=cfg.save_dir)
        run_function(logger, save_fn, list(removed_tables.items()),
                    Parallel=cfg.parallel, P_type='process')

        if cfg.export_fully_removed:
            logger.info('Compiling fully removed sessions...')
            iden_uniq_after_set = set(Iden_uniq_after)
            fully_removed_list = [
                PRE_TABLE[PRE_TABLE['SessionID'] == sid]
                for sid in Iden_uniq if sid not in iden_uniq_after_set
            ]
            if fully_removed_list:
                fully_removed = pd.concat(fully_removed_list, ignore_index=True)
                fully_path = os.path.join(cfg.save_dir, 'removal_log', 'Removed_fully.csv')
                fully_removed.to_csv(fully_path, index=False)
                logger.info(f'Saved fully removed sessions to {fully_path}')
        else:
            logger.info('Export of fully removed sessions skipped.')

    if cfg.filter_only:
        logger.info('Filter only mode enabled. Exiting after filtering step.')
        return

    # -- Splitting step --------------------------------------------------
    split_path = os.path.join(cfg.save_dir, 'Data_table_split.csv')
    temporary_relocation = []

    if not os.path.exists(split_path):
        logger.info('No split table found, starting splitting process')

        split_subsets = [
            group.copy() for sid, group in Data_table.groupby('SessionID')
            if sid in Iden_uniq_after
        ]

        # Try to resume from checkpoint
        split_completed_ids = []
        all_split_results = []
        all_split_redirections = []

        if cfg.resume:
            split_completed_ids, all_split_results, all_split_redirections = \
                _load_split_checkpoint(cfg, logger)
            if split_completed_ids is not None:
                logger.info(f'Resuming split checkpoint: {len(split_completed_ids)} sessions already split')
            else:
                cfg.resume = False

        if split_completed_ids:
            completed_set = set(split_completed_ids)
            split_subsets = [
                group.copy()
                for sid, group in Data_table.groupby('SessionID')
                if sid in Iden_uniq_after and sid not in completed_set
            ]

        if not split_subsets:
            logger.info('All sessions already split or no data to split')
            if all_split_results:
                Data_table = pd.concat([df for df in all_split_results if not df.empty]).reset_index(drop=True)
                temporary_relocation = list(all_split_redirections)
                Iden_uniq_after = Data_table['SessionID'].unique()
        else:
            logger.info(f'Splitting {len(split_subsets)} session(s)')

            split_fn = functools.partial(_split_worker, logger=logger)

            for batch_start in range(0, len(split_subsets), cfg.filter_batch_size):
                batch = split_subsets[batch_start:batch_start + cfg.filter_batch_size]
                logger.info(
                    f'Splitting batch {(batch_start // cfg.filter_batch_size) + 1}: '
                    f'{batch_start + 1}-{min(batch_start + cfg.filter_batch_size, len(split_subsets))} '
                    f'of {len(split_subsets)} sessions'
                )

                batch_results, batch_redirects = run_function(
                    logger, split_fn, batch,
                    Parallel=cfg.parallel, P_type='process',
                )

                batch_results = [df for df in batch_results if not df.empty]
                all_split_results.extend(batch_results)
                all_split_redirections.extend(batch_redirects)

                for df in batch_results:
                    for sid in df['SessionID'].unique():
                        split_completed_ids.append(sid)
                for subset in batch:
                    sid = subset['SessionID'].values[0]
                    if sid not in split_completed_ids:
                        split_completed_ids.append(sid)

                _save_split_checkpoint(cfg, logger, split_completed_ids,
                                      all_split_results, all_split_redirections)

            results = [df for df in all_split_results if not df.empty]
            Data_table = pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()
            temporary_relocation = list(all_split_redirections)
            Iden_uniq_after = Data_table['SessionID'].unique()

            _remove_split_checkpoint(cfg, logger)

        logger.info(f'Updated scans after splitting            : {len(Data_table)}')
        logger.info(f'Updated sessions after splitting          : {len(Iden_uniq_after)}')
        logger.info(f'Temporary relocations after splitting     : {len(temporary_relocation)}')
        logger.debug(f'Temp relocations example [first 3]: {temporary_relocation[:3]}')

        _atomic_write_csv(Data_table, split_path)
    else:
        logger.info('Split table found, loading split data')
        Data_table = pd.read_csv(split_path, low_memory=False)
        temporary_relocation = []
        logger.info('Temporary relocation list is empty (symlinks created on the fly)')

    # -- Ordering step ---------------------------------------------------
    Data_subsets = [group.copy() for _, group in Data_table.groupby('SessionID')]
    order_input_ids = [subset['SessionID'].iloc[0] for subset in Data_subsets]

    if not os.path.exists(out_path):
        logger.info('No ordered table found, starting ordering process')
        order_fn = functools.partial(_order_worker, logger=logger)

        if cfg.resume:
            completed_ids, order_results = _load_order_checkpoint(cfg, logger)
            if completed_ids is not None and order_results is not None:
                remaining = [item for item in zip(order_input_ids, Data_subsets)
                             if item[0] not in completed_ids]
                logger.info(
                    f'Resuming order from checkpoint: '
                    f'{len(completed_ids)} done, {len(remaining)} remaining'
                )
                Data_subsets = [item[1] for item in remaining]
                order_input_ids = [item[0] for item in remaining]
                if not Data_subsets:
                    Data_table = pd.concat(order_results).reset_index(drop=True)
                    logger.info('All ordering already completed from checkpoint')
            else:
                order_results = []
                completed_ids = []
        else:
            order_results = []
            completed_ids = []

        if Data_subsets:
            order_input = list(zip(order_input_ids, Data_subsets))
            batch_size = getattr(cfg, 'filter_batch_size', 10)
            for start in range(0, len(order_input), batch_size):
                end = min(start + batch_size, len(order_input))
                batch = [item[1] for item in order_input[start:end]]
                batch_ids = [item[0] for item in order_input[start:end]]

                new_results = run_function(
                    logger, order_fn, batch,
                    Parallel=cfg.parallel, P_type='process',
                )
                order_results.extend(new_results)
                completed_ids.extend(batch_ids)

                _save_order_checkpoint(cfg, logger, completed_ids, order_results)

            Data_table = pd.concat(order_results).reset_index(drop=True)

            logger.info('Ordering complete')
            logger.info(f'Final sessions: {len(Data_table["SessionID"].unique())}')
            logger.info(f'Final scans   : {len(Data_table)}')
            logger.info(f'Saving ordered data to {out_path}')
            _atomic_write_csv(Data_table, out_path)
    else:
        logger.info('Ordered table found, loading ordered data')
        Data_table = pd.read_csv(out_path, low_memory=False)

    # -- Symlink relocations ------------------------------------------------
    logger.debug(
        f'Creating symlinks for separated post scans. '
        f'Temporary relocations: {len(temporary_relocation)}')
    relocate_fn = functools.partial(_relocate_worker,
                                    relocations=list(temporary_relocation),
                                    logger=logger)
    if temporary_relocation:
        run_function(logger, relocate_fn, list(temporary_relocation),
                     Parallel=False, P_type='process')

    logger.info('Redirection complete')
    _remove_checkpoint(cfg, logger)
    _remove_split_checkpoint(cfg, logger)
    _remove_order_checkpoint(cfg, logger)


if __name__ == '__main__':
    cfg = build_config()
    logger = create_logger(cfg)

    try:
        if cfg.profile:
            yappi.start()

        os.makedirs(cfg.save_dir, exist_ok=True)

        if cfg.dir_idx is None:
            main(cfg, logger)
        else:
            cfg.parallel = False
            assert os.path.exists(cfg.dir_list), \
                f'Directory list file {cfg.dir_list} does not exist'
            save_dir_worker = os.path.join(cfg.save_dir, 'tmp/')
            cfg = replace(cfg, save_dir=save_dir_worker)
            logger = create_logger(cfg)

            with open(cfg.dir_list, 'rb') as f:
                items = pickle.load(f)
            target = items[cfg.dir_idx].strip()
            logger.info(f'Processing single directory: {cfg.dir_idx}')
            cfg = replace(cfg, target=target,
                                            out_name=f'Data_table_timing_{cfg.dir_idx}.csv')

            main(cfg, logger)

            if cfg.dir_idx == len(items) - 1:
                logger.info('Last script, compiling results')
                while True:
                    tables = [t for t in os.listdir(save_dir_worker) if t.endswith('.csv')]
                    if len(tables) >= len(items):
                        break
                    logger.info('Waiting for all tables to be compiled')
                    time.sleep(5)

                logger.info('All tables present, compiling...')
                frames = []
                for table in tables:
                    logger.info(f'Compiling {table}')
                    try:
                        frames.append(pd.read_csv(os.path.join(save_dir_worker, table)))
                    except pd.errors.EmptyDataError:
                        logger.error(f'{table} is empty, skipping')
                        continue
                    except Exception as e:
                        logger.error(f'Error compiling {table}: {e}')
                        break
                combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

                final_dir = os.path.dirname(save_dir_worker.rstrip('/'))
                combined.to_csv(os.path.join(final_dir, 'Data_table_timing.csv'), index=False)
                logger.info(f'Compiled results saved to {final_dir}')
                try:
                    shutil.rmtree(save_dir_worker)
                    logger.info(f'Deleted temporary directory {save_dir_worker}')
                except Exception as e:
                    logger.error(f'Error deleting {save_dir_worker}: {e}')

    finally:
        if cfg.profile:
            yappi.stop()
            profile_path = 'step02_profile.yappi'
            logger.info(f'Writing profile results to {profile_path}')
            yappi.get_func_stats().save(profile_path, type='pstat')
            logger.info(f'Profile results saved to {profile_path}')

    sys.exit(0)