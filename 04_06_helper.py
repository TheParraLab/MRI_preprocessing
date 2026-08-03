import os
import subprocess
import argparse

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code', 'preprocessing')
DEFAULT_SCAN_DIR = '/FL_system/data/nifti/'
DEFAULT_RAS_DIR = '/FL_system/data/RAS/'
DEFAULT_COREG_DIR = '/FL_system/data/coreg/'
DEFAULT_INPUTS_DIR = '/FL_system/data/inputs/'


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

    args = parser.parse_args()

    base_args = []
    if args.test is not None:
        base_args += ['--test', str(args.test)]
    if args.multi:
        base_args.append('--multi')

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

    for name, cmd_args in steps:
        script = os.path.join(SCRIPT_DIR, name)
        print(f'\n--- Running step ({name}) ---')
        result = subprocess.run(['python', script] + cmd_args)
        if result.returncode != 0:
            print(f'Step {name} failed (exit code {result.returncode}). Stopping.')
            break
        print(f'Step {name} completed successfully.')

    else:
        print('\nAll steps completed.')


if __name__ == '__main__':
    main()
