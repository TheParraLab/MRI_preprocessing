"""
Digest a comparison report JSON into plain-text session ID lists.

Takes the JSON output from compare_checksum.py and produces three
plain-text files, each containing one session ID per line:
  - sessions_need_transfer.txt
  - sessions_ready_for_deletion.txt
  - sessions_missing_from_primary.txt

Usage:
  python digest_comparison.py <comparison_report.json> [-o OUTPUT_DIR]
"""
import os
import sys
import json
from argparse import ArgumentParser

import checksum_core as core

parser = ArgumentParser(description="Digest a comparison report JSON into plain-text session ID lists.")
parser.add_argument("report", help="Path to the comparison report JSON file (output from compare_checksum.py).")
parser.add_argument("-o", "--outdir", help="Directory to write output files. Defaults to the report's directory.")
args = parser.parse_args()

if not os.path.exists(args.report):
    print(f"Error: {args.report} not found.", file=sys.stderr)
    sys.exit(1)

with open(args.report, 'r') as f:
    data = json.load(f)

report = data["report"]
if args.outdir:
    outdir = args.outdir
else:
    outdir = os.path.dirname(os.path.abspath(args.report))
os.makedirs(outdir, exist_ok=True)

transfer_sessions = sorted({item["session"] for item in report["need_transfer"]})

deletion_sessions = sorted({
    item["path"].replace(os.sep, "/").split("/", 1)[0]
    for item in report["ready_for_deletion"]
})

missing_sessions = sorted({item["session"] for item in report["missing_from_primary"]})

n_transfer = core.write_session_list(os.path.join(outdir, "sessions_need_transfer.txt"), transfer_sessions)
n_deletion = core.write_session_list(os.path.join(outdir, "sessions_ready_for_deletion.txt"), deletion_sessions)
n_missing = core.write_session_list(os.path.join(outdir, "sessions_missing_from_primary.txt"), missing_sessions)

print(f"Digested {args.report} -> {outdir}")
print(f"  sessions_need_transfer.txt: {n_transfer} sessions")
print(f"  sessions_ready_for_deletion.txt: {n_deletion} sessions")
print(f"  sessions_missing_from_primary.txt: {n_missing} sessions")
