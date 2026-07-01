import json
import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description="Digest a comparison report JSON into plain-text session ID lists.")
parser.add_argument("report", help="Path to the comparison report JSON file (output from compare_checksum.py).")
parser.add_argument("-o", "--outdir", help="Directory to write output files. Defaults to comparison_findings/.")
args = parser.parse_args()

if not os.path.exists(args.report):
    print(f"Error: {args.report} not found.", file=sys.stderr)
    sys.exit(1)

with open(args.report, 'r') as f:
    data = json.load(f)

report = data["report"]
outdir = args.outdir if args.outdir else "comparison_findings"

# need_transfer: sessions flagged because at least one file differed
transfer_sessions = sorted(item["session"] for item in report["need_transfer"])

# ready_for_deletion: extract unique sessions from individual file entries
deletion_sessions = sorted(item["path"].split("/", 1)[0] for item in report["ready_for_deletion"])
deletion_sessions = sorted(set(deletion_sessions))

# missing_from_primary: sessions in secondary with no primary counterpart
missing_sessions = sorted(item["session"] for item in report["missing_from_primary"])

files_written = []

with open(os.path.join(outdir, "sessions_need_transfer.txt"), 'w') as f:
    f.write("\n".join(transfer_sessions))
    if transfer_sessions:
        f.write("\n")
    files_written.append(("sessions_need_transfer.txt", len(transfer_sessions)))

with open(os.path.join(outdir, "sessions_ready_for_deletion.txt"), 'w') as f:
    f.write("\n".join(deletion_sessions))
    if deletion_sessions:
        f.write("\n")
    files_written.append(("sessions_ready_for_deletion.txt", len(deletion_sessions)))

with open(os.path.join(outdir, "sessions_missing_from_primary.txt"), 'w') as f:
    f.write("\n".join(missing_sessions))
    if missing_sessions:
        f.write("\n")
    files_written.append(("sessions_missing_from_primary.txt", len(missing_sessions)))

print(f"Digested {args.report}")
for fname, count in files_written:
    print(f"  {fname}: {count} sessions")
