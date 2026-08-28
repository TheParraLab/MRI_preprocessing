import os
import json
import subprocess
import argparse

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code', 'preprocessing')
DEFAULT_SCAN_DIR = '/FL_system/data/nifti/'
DEFAULT_RAS_DIR = '/FL_system/data/RAS/'
DEFAULT_COREG_DIR = '/FL_system/data/coreg/'
DEFAULT_INPUTS_DIR = '/FL_system/data/inputs/'


def load_ids(ids_file: str) -> set:
    """Load IDs from a CSV/txt file (one ID per line)."""
    with open(ids_file, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def check_directory_complete(directory: str, min_files: int = 1) -> bool:
    """Check if a directory exists and has at least min_files contents."""
    if not os.path.isdir(directory):
        return False
    try:
        entries = [e for e in os.listdir(directory) if not e.startswith('.')]
        return len(entries) >= min_files
    except PermissionError:
        return False


def verify_ids_at_step(ids: set, output_dir: str, step_name: str, min_files: int = 1) -> dict:
    """Check which IDs have completed output at a given step."""
    succeeded = []
    failed = []
    missing_on_disk = []
    for id_ in ids:
        out_path = os.path.join(output_dir, id_)
        if check_directory_complete(out_path, min_files):
            succeeded.append(id_)
        else:
            failed.append(id_)
    return {
        'step': step_name,
        'total_ids': len(ids),
        'succeeded_count': len(succeeded),
        'failed_count': len(failed),
        'succeeded_ids': sorted(succeeded),
        'failed_ids': sorted(failed),
    }


def run_verification(ids: set, ras_dir: str, coreg_dir: str, inputs_dir: str, ids_file: str) -> dict:
    """Run ID verification after all steps complete and write summary."""
    step_04 = verify_ids_at_step(ids, ras_dir, '04_saveRAS', min_files=1)
    step_05 = verify_ids_at_step(ids, coreg_dir, '05_alignScans', min_files=1)
    step_06 = verify_ids_at_step(ids, inputs_dir, '06_genInputs', min_files=3)

    summary = {
        'ids_file': ids_file,
        'total_input_ids': len(ids),
        'step_04_saveRAS': step_04,
        'step_05_alignScans': step_05,
        'step_06_genInputs': step_06,
        'final_pipeline_ids': sorted(set(step_06['succeeded_ids'])),
    }

    print('\n--- Final ID Verification ---')
    for step in ['step_04_saveRAS', 'step_05_alignScans', 'step_06_genInputs']:
        s = summary[step]
        print(f"  {s['step']}: {s['succeeded_count']} succeeded, {s['failed_count']} failed/missing out of {s['total_ids']}")

    print(f"\n  Final IDs through entire pipeline: {len(summary['final_pipeline_ids'])}")

    if summary['step_06']['failed_count'] > 0:
        fail_path = os.path.join(os.path.dirname(ids_file), 'failed_ids_after_step_06.txt')
        with open(fail_path, 'w') as f:
            for id_ in summary['step_06']['failed_ids']:
                f.write(f'{id_}\n')
        print(f'  Failed IDs written to: {fail_path}')

    out_path = os.path.join(os.path.dirname(ids_file), 'processing_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Full summary written to: {out_path}')

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run preprocessing steps 04-06 sequentially.'
    )
    parser.add_argument(
        '-n', '--test', type=int,
        help='Number of sessions to randomly sample (per step). Omit to process all.'
    )
    parser.add_argument('-m', '--multi', action='store_true',
                        help='Enable multiprocessing for all steps.')
    parser.add_argument('--scan_dir', type=str, default=DEFAULT_SCAN_DIR,
                        help=f'Directory containing NIfTI scans (default: {DEFAULT_SCAN_DIR})')
    parser.add_argument('--ras_dir', type=str, default=DEFAULT_RAS_DIR,
                        help=f'Directory for RAS-converted output (default: {DEFAULT_RAS_DIR})')
    parser.add_argument('--coreg_dir', type=str, default=DEFAULT_COREG_DIR,
                        help=f'Directory for coregistered output (default: {DEFAULT_COREG_DIR})')
    parser.add_argument('--inputs_dir', type=str, default=DEFAULT_INPUTS_DIR,
                        help=f'Directory for model inputs (default: {DEFAULT_INPUTS_DIR})')
    parser.add_argument('--ids_file', type=str, default=None,
                        help='CSV/txt file containing one ID per line. Only processes directories whose name appears in this file.')

    args = parser.parse_args()

    ids_to_process = None
    if args.ids_file is not None:
        ids_to_process = load_ids(args.ids_file)
        print(f'Loaded {len(ids_to_process)} IDs from {args.ids_file}')

    base_args = []
    if args.test is not None:
        base_args += ['--test', str(args.test)]
    if args.multi:
        base_args.append('--multi')
    if args.ids_file is not None:
        base_args += ['--ids_file', args.ids_file]

    steps = [
        (
            '04_saveRAS.py',
            [
                '--scan_dir', args.scan_dir,
                '--save_dir', args.ras_dir,
            ] + base_args,
        ),
        (
            '05_alignScans.py',
            [
                '--load_dir', args.ras_dir,
                '--save_dir', args.coreg_dir,
            ] + base_args,
        ),
        (
            '06_genInputs.py',
            [
                '--load_dir', args.coreg_dir,
                '--save_dir', args.inputs_dir,
            ] + base_args,
        ),
    ]

    all_completed = True
    for name, cmd_args in steps:
        script = os.path.join(SCRIPT_DIR, name)
        print(f'\n--- Running step ({name}) ---')
        result = subprocess.run(['python', script] + cmd_args)
        if result.returncode != 0:
            print(f'Step {name} failed (exit code {result.returncode}). Stopping.')
            all_completed = False
            break
        print(f'Step {name} completed successfully.')

    if all_completed and ids_to_process is not None:
        run_verification(ids_to_process, args.ras_dir, args.coreg_dir, args.inputs_dir, args.ids_file)


if __name__ == '__main__':
    main()
