"""
Remove session directories listed in a plain-text file from a target directory.

Reads session IDs (one per line) from a text file and deletes each
corresponding directory from the target path.

Usage:
  python remove_sessions.py <sessions.txt> <target_dir> [--dry-run] [--yes]
"""
import os
import sys
import shutil
from argparse import ArgumentParser

parser = ArgumentParser(description="Remove session directories listed in a plain-text file from a target directory.")
parser.add_argument("session_file", help="Path to the text file containing session IDs (one per line).")
parser.add_argument("target", help="Target base directory (parent of session directories to remove).")
parser.add_argument("--dry-run", action="store_true", help="Print what would be removed without actually removing.")
parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
args = parser.parse_args()

if not os.path.isfile(args.session_file):
    print(f"Error: {args.session_file} not found.", file=sys.stderr)
    sys.exit(1)

if not os.path.isdir(args.target):
    print(f"Error: target directory {args.target} not found.", file=sys.stderr)
    sys.exit(1)

with open(args.session_file, 'r') as f:
    sessions = [line.strip() for line in f if line.strip()]

if not sessions:
    print(f"No session IDs found in {args.session_file}.")
    sys.exit(0)

if not args.dry_run and not args.yes:
    answer = input(f"Remove {len(sessions)} session(s) from {args.target}? Type 'yes' to continue: ")
    if answer.strip().lower() != "yes":
        print("Aborted by user.")
        sys.exit(0)

total = len(sessions)
removed = 0
skipped = 0
errors = 0

mode = "Would remove" if args.dry_run else "Removing"
print(f"{mode} {total} sessions from {args.target}")
print(f"{'='*60}")

for i, session_id in enumerate(sessions, 1):
    target_path = os.path.join(args.target, session_id)

    if not os.path.exists(target_path):
        print(f"[{i}/{total}] SKIP (not found): {session_id}")
        skipped += 1
        continue

    if args.dry_run:
        print(f"[{i}/{total}] DRY-RUN: {session_id}")
        removed += 1
        continue

    try:
        shutil.rmtree(target_path)
        print(f"[{i}/{total}] OK: {session_id}")
        removed += 1
    except Exception as e:
        print(f"[{i}/{total}] ERROR: {session_id} -> {e}", file=sys.stderr)
        errors += 1

print(f"{'='*60}")
print(f"Done: {removed} removed, {skipped} skipped, {errors} errors out of {total} sessions.")
