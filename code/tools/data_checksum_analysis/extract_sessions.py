"""
Extract session directory names from a scan result JSON and write them
to a plain-text file (one session ID per line).

Usage:
  python extract_sessions.py <scan_result.json> [-o OUTPUT.txt]
"""
import json
import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description="Extract session IDs from a scan result JSON into a plain-text file.")
parser.add_argument("scan_file", help="Path to the scan result JSON file.")
parser.add_argument("-o", "--output", help="Output text file path. Defaults to <scan_basename>_sessions.txt.")
args = parser.parse_args()

if not os.path.isfile(args.scan_file):
    print(f"Error: {args.scan_file} not found.", file=sys.stderr)
    sys.exit(1)

with open(args.scan_file, 'r') as f:
    data = json.load(f)

sessions = sorted(data["results"].keys())

if not sessions:
    print("No sessions found in scan result.")
    sys.exit(0)

if args.output:
    output_path = args.output
else:
    base = os.path.splitext(os.path.basename(args.scan_file))[0]
    output_path = os.path.join(os.path.dirname(args.scan_file) or ".", f"{base}_sessions.txt")

with open(output_path, 'w') as f:
    f.write("\n".join(sessions))
    f.write("\n")

print(f"Wrote {len(sessions)} session IDs to {output_path}")