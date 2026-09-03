"""
Scan a directory tree, compute checksums for every file, and save the
results as a JSON file in scan_results/.

Organizes files by parent session directory. Outputs a JSON file with
metadata (scan directory, start/stop timestamps, algorithm, file counts)
and a checksum entry for each file found.

Usage:
  python scan_dest.py /path/to/scan [options]

Options:
  --skip DIR              Directory name(s) to skip (can be repeated).
  --skip-file FILE        Text file with session names to skip (one per line).
  --output FILE           Output filename (defaults to auto-generated).
  --hash ALGO             Hash algorithm: sha256 (default) or md5.
  --workers N             Number of worker threads for hashing.
"""
import os
import sys
from argparse import ArgumentParser

import checksum_core as core

parser = ArgumentParser(description="Scan a directory tree and compute checksums for every file.")
parser.add_argument("scan_dir", help="Root directory to scan. Session subdirectories will be checksummed.")
parser.add_argument("--skip", action="append", default=[], help="Session directory names to skip (can be repeated).")
parser.add_argument("--skip-file", help="Text file with session names to skip (one per line).")
parser.add_argument("--output", help="Output filename in scan_results/ (defaults to auto-generated).")
parser.add_argument("--hash", choices=["sha256", "md5"], default=core.DEFAULT_ALGO, help="Hash algorithm.")
parser.add_argument("--workers", type=int, default=None, help="Hashing threads (default: cpu_count; 1 = serial).")
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

print(f'Scanning directory: {args.scan_dir}')
if skip_list:
    print(f'Skipping sessions: {sorted(set(skip_list))}')
print(f'Using hash algorithm: {args.hash}')

try:
    header, results = core.scan_tree(
        args.scan_dir, algo=args.hash, skip=skip_list, n_workers=args.workers
    )
except (OSError, NotADirectoryError) as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

output = {
    "header": header,
    "results": results,
}

results_dir = os.path.join(os.getcwd(), 'scan_results')
os.makedirs(results_dir, exist_ok=True)

if args.output:
    output_path = os.path.join(results_dir, os.path.basename(args.output))
else:
    output_path = core.alloc_scan_name(results_dir, base="scan_results", ext="json")

with open(output_path, 'w', encoding='utf-8') as f:
    import json
    json.dump(output, f, indent=2)

print(f'Finished: {header["n_sessions"]} sessions, {header["n_files"]} files ({args.hash})')
print(f'Saved JSON results to: {output_path}')
