"""
Scan a directory tree, compute MD5 checksums for every file, and save
the results as a JSON file in scan_results/.

Organizes files by parent session directory. Outputs a JSON file with
metadata (scan directory, start/stop timestamps) and a checksum entry
for each file found.

Usage:
  python scan_dest.py /path/to/scan [options]

Options:
  --skip DIR              Directory name(s) to skip (can be repeated).
  --skip-file FILE        Text file with session names to skip (one per line).
  --output FILE           Output filename (defaults to auto-generated).
"""
import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import md5

parser = ArgumentParser(description="Scan a directory tree and compute MD5 checksums for every file.")
parser.add_argument("scan_dir", help="Root directory to scan. Session subdirectories will be checksummed.")
parser.add_argument("--skip", action="append", default=[], help="Session directory names to skip (can be repeated).")
parser.add_argument("--skip-file", help="Text file with session names to skip (one per line).")
parser.add_argument("--output", help="Output filename in scan_results/ (defaults to auto-generated).")
args = parser.parse_args()

if not os.path.isdir(args.scan_dir):
    print(f"Error: {args.scan_dir} is not a valid directory.", file=sys.stderr)
    sys.exit(1)

skip_list = list(args.skip)
if args.skip_file:
    if not os.path.isfile(args.skip_file):
        print(f"Error: {args.skip_file} not found.", file=sys.stderr)
        sys.exit(1)
    with open(args.skip_file, 'r') as f:
        skip_list.extend(line.strip() for line in f if line.strip())

def file_md5(file_path):
    hash_md5 = md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

start_time = datetime.now(timezone.utc)
skip_set = set(skip_list)
print(f'Scanning directory: {args.scan_dir}')
if skip_set:
    print(f'Skipping sessions: {sorted(skip_set)}')

results = {}

for root, dirs, files in os.walk(args.scan_dir):
    if root == args.scan_dir:
        continue

    session_id = os.path.basename(root)

    if session_id in skip_set:
        print(f'Skipping session: {session_id}')
        continue

    session_files = []

    for file in sorted(files):
        file_path = os.path.join(root, file)
        try:
            session_files.append({
                'file_name': file,
                'md5': file_md5(file_path),
            })
        except (OSError, PermissionError):
            print(f'  Warning: could not read {file_path}, skipping.')

    if session_files:
        results[session_id] = {
            'files': session_files,
        }

stop_time = datetime.now(timezone.utc)

output = {
    'header': {
        'scan_dir': args.scan_dir,
        'start_time': start_time.isoformat(),
        'stop_time': stop_time.isoformat(),
        'skipped': sorted(skip_set),
    },
    'results': results
}

results_dir = os.path.join(os.getcwd(), 'scan_results')
os.makedirs(results_dir, exist_ok=True)

if args.output:
    output_file = args.output
else:
    output_file = 'scan_results_0.json'
    if os.path.exists(os.path.join(results_dir, output_file)):
        N = output_file.split('_')[-1].split('.')[0]
        N = int(N)
        output_file = f'scan_results_{N + 1}.json'

output_path = os.path.join(results_dir, output_file)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
print(f'Saved JSON results to: {output_path}')