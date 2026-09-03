"""
Move session directories listed in a plain-text file from one base path to another.

Reads session IDs (one per line) from a text file and moves each
corresponding directory from the source to the destination.

Usage:
  python move_sessions.py <sessions.txt> <source_dir> <dest_dir> [--dry-run] [--yes]
"""
import os
import sys
import shutil
from argparse import ArgumentParser

parser = ArgumentParser(description="Move session directories listed in a plain-text file from one base path to another.")
parser.add_argument("session_file", help="Path to the text file containing session IDs (one per line).")
parser.add_argument("source", help="Source base directory (parent of session directories).")
parser.add_argument("destination", help="Destination base directory to move sessions into.")
parser.add_argument("--dry-run", action="store_true", help="Print what would be moved without actually moving.")
parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
args = parser.parse_args()

if not os.path.isfile(args.session_file):
    print(f"Error: {args.session_file} not found.", file=sys.stderr)
    sys.exit(1)

if not os.path.isdir(args.source):
    print(f"Error: source directory {args.source} not found.", file=sys.stderr)
    sys.exit(1)

with open(args.session_file, 'r') as f:
    sessions = [line.strip() for line in f if line.strip()]

if not sessions:
    print(f"No session IDs found in {args.session_file}.")
    sys.exit(0)

if not args.dry_run and not args.yes:
    answer = input(f"Move {len(sessions)} session(s) from {args.source} to {args.destination}? Type 'yes' to continue: ")
    if answer.strip().lower() != "yes":
        print("Aborted by user.")
        sys.exit(0)

total = len(sessions)
moved = 0
skipped = 0
errors = 0

mode = "Would move" if args.dry_run else "Moving"
print(f"{mode} {total} sessions from {args.source} -> {args.destination}")
print(f"{'='*60}")

for i, session_id in enumerate(sessions, 1):
    src_path = os.path.join(args.source, session_id)
    dst_path = os.path.join(args.destination, session_id)

    if not os.path.exists(src_path):
        print(f"[{i}/{total}] SKIP (not found): {session_id}")
        skipped += 1
        continue

    if args.dry_run:
        print(f"[{i}/{total}] DRY-RUN: {session_id}")
        moved += 1
        continue

    try:
        os.makedirs(args.destination, exist_ok=True)
        shutil.move(src_path, dst_path)
        print(f"[{i}/{total}] OK: {session_id}")
        moved += 1
    except Exception as e:
        print(f"[{i}/{total}] ERROR: {session_id} -> {e}", file=sys.stderr)
        errors += 1

print(f"{'='*60}")
print(f"Done: {moved} moved, {skipped} skipped, {errors} errors out of {total} sessions.")
