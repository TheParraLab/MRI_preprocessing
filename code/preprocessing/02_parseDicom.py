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
import subprocess
import re
import random
import pickle
import logging
from multiprocessing import cpu_count
from typing import Any, Optional
from functools import partial
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


# ------ --------------------------- ---- - --------------- --- -- --------- -
# Checkpoint helpers
# ------ ---------------------------------- --- - -------------- --- --- ---

def save_progress(cfg: ParseConfig, logger: logging.Logger, data: list, filename: str) -> None:
    """Save progress atomically using tmp + os.replace."""
    logger.info(f'Saving progress to {filename}')
    tmp_path = os.path.join(cfg.save_dir, f'.{filename}.tmp')
    final_path = os.path.join(cfg.save_dir, filename)
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp_path, final_path)
    except Exception as e:
        logger.error(f'Failed to write progress {final_path}: {e}')


def load_progress(cfg: ParseConfig, logger: logging.Logger, filename: str) -> Optional[Any]:
    """Load progress checkpoint if it exists."""
    path = os.path.join(cfg.save_dir, filename)
    if not os.path.exists(path):
        return None
    logger.info(f'Loading progress from {filename}')
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f'Failed to load progress {path}: {e}')
    return None


# ---------------------------------------------------------------------------
# Pipeline workers  (accept plain args for run_function compatibility)
# ---------------------------------------------------------------------------

def _filter_worker(data_subset: pd.DataFrame, save_dir: str, computed_flags: list,
                   description_flags: list, logger: logging.Logger) -> tuple:
    """Worker for filter step — called per session subset."""
    data_subset = data_subset.reset_index(drop=True)
    tmp_save = save_dir.replace('tmp/', 'tmp_data/')
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
    for removed_dict in removed_list:
        for key, value in removed_dict.items():
            removed_tables[key] = pd.concat([removed_tables[key], value], ignore_index=True)


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
            try:
                answer = input('Would you like to reprocess? [Y/n]: ')
            except (EOFError, KeyboardInterrupt):
                logger.warning('No input received, aborting.')
                return
            if answer.lower() != 'y':
                logger.info('Stopping processing.')
                return

    # -- Load progress checkpoint or init from scratch ------------------------
    progress = load_progress(cfg, logger, 'parseDicom_progress.pkl')
    temporary_relocation = list(progress) if progress else []

    if progress:
        logger.info(f'Progress file found. {len(progress)} items remaining')
        Data_table = None
        removed_tables = defaultdict(pd.DataFrame)
    else:
        Data_table, removed_tables = _init_data_table(cfg.load_table, cfg.target, logger)
        Iden_uniq = np.unique(Data_table['SessionID'])
        PRE_TABLE = Data_table.copy()

        if cfg.test:
            Iden_uniq = Iden_uniq[:cfg.n_test]
            logger.info(f'Running in test mode with {cfg.n_test} sessions')

        if cfg.parallel:
            logger.debug('Running in parallel mode')

        Data_subsets = [group.copy() for _, group in Data_table.groupby('SessionID')]
        random.shuffle(Data_subsets)

        # -- Filtering step --------------------------------------------------
        filter_path = os.path.join(cfg.save_dir, 'Data_table_filtered.csv')
        if not os.path.exists(filter_path):
            logger.info('No filtered table found, starting filtering process')

            filter_fn = functools.partial(
                _filter_worker,
                save_dir=cfg.save_dir,
                computed_flags=cfg.computed_flags,
                description_flags=cfg.description_flags,
                logger=logger,
            )
            results, removed, temp_rels = run_function(
                logger, filter_fn, Data_subsets,
                Parallel=cfg.parallel, P_type='process',
            )

            results = [df for df in results if not df.empty]
            removed = list(removed)
            temp_rels = list(temp_rels)

            Data_table = pd.concat(results).reset_index(drop=True)
            Iden_uniq_after = Data_table['SessionID'].unique()

            Iden_uniq_after_clean = []
            for sid in Iden_uniq_after:
                if sid[-2:] in ('_a', '_b', '_l', '_r'):
                    Iden_uniq_after_clean.append(sid[:-2])
                else:
                    Iden_uniq_after_clean.append(sid)
            Iden_uniq_after_clean = list(set(Iden_uniq_after_clean))

            _aggregate_removed(removed_tables, removed)

            logger.info('Filtering Results:')
            logger.info(f'Initial number of unique sessions: {len(Iden_uniq)}')
            logger.info(f'Final number of unique sessions  : {len(Iden_uniq_after_clean)}')
            logger.info(f'Final number of sessions (w/ lat ): {len(Iden_uniq_after)}')
            logger.info(f'Removed sessions                 : {len(Iden_uniq) - len(Iden_uniq_after_clean)}')

            for key, value in removed_tables.items():
                logger.info(f'=== {key} ===')
                rem_id = value['SessionID'].unique()
                gone_id = set(rem_id) - set(Iden_uniq_after_clean)
                logger.info(f'  Sessions missing from output: {len(gone_id)}')
                logger.info(f'  Scans removed               : {len(value)}')

            Data_table = _normalize_bool_cols(Data_table)
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
        else:
            logger.info('Filtered table found, loading filtered data')
            Data_table = pd.read_csv(filter_path, low_memory=False)
            Iden_uniq_after = Data_table['SessionID'].unique()

        if cfg.filter_only:
            logger.info('Filter only mode enabled. Exiting after filtering step.')
            return

        Data_table = _normalize_bool_cols(Data_table)

        # -- Splitting step --------------------------------------------------
        Data_subsets = [
            group.copy() for sid, group in Data_table.groupby('SessionID')
            if sid in Iden_uniq_after
        ]

        split_path = os.path.join(cfg.save_dir, 'Data_table_split.csv')
        if not os.path.exists(split_path):
            logger.info('No split table found, starting splitting process')
            split_fn = functools.partial(_split_worker, logger=logger)
            results, redirections = run_function(
                logger, split_fn, Data_subsets,
                Parallel=cfg.parallel, P_type='process',
            )
            results = [df for df in results if not df.empty]
            Data_table = pd.concat(results).reset_index(drop=True)
            temporary_relocation = list(redirections)
            Iden_uniq_after = Data_table['SessionID'].unique()

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

        if not os.path.exists(out_path):
            logger.info('No ordered table found, starting ordering process')
            order_fn = functools.partial(_order_worker, logger=logger)
            results = run_function(
                logger, order_fn, Data_subsets,
                Parallel=cfg.parallel, P_type='process',
            )
            Data_table = pd.concat(results).reset_index(drop=True)

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
    progress_path = os.path.join(cfg.save_dir, 'parseDicom_progress.pkl')
    if os.path.exists(progress_path):
        logger.info('Removing progress file')
        os.remove(progress_path)


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
                combined = pd.DataFrame()
                for table in tables:
                    logger.info(f'Compiling {table}')
                    try:
                        tmp = pd.read_csv(os.path.join(save_dir_worker, table))
                        combined = pd.concat([combined, tmp], ignore_index=True)
                    except pd.errors.EmptyDataError:
                        logger.error(f'{table} is empty, skipping')
                        continue
                    except Exception as e:
                        logger.error(f'Error compiling {table}: {e}')
                        break

                final_dir = save_dir_worker.replace('tmp/', '')
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