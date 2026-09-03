"""
Merge two checksum scan result JSON files into a single comparison file.

Builds a per-file index from both scans and classifies every file as
identical, modified, primary_only, or secondary_only. Writes a merged
JSON with statistics and per-session file listings.

Usage:
  python merge_checksums.py <scan1.json> <scan2.json> [-o OUTPUT.json]
"""
import os
import sys
import json
from argparse import ArgumentParser
from datetime import datetime, timezone

import checksum_core as core

parser = ArgumentParser(description="Merge two checksum scan result JSON files into a single comparison file.")
parser.add_argument("scan1", help="Path to the first scan result JSON file (primary/source).")
parser.add_argument("scan2", help="Path to the second scan result JSON file (secondary/destination).")
parser.add_argument("-o", "--output", help="Output file path. Defaults to merged_comparison.json in the current directory.")
args = parser.parse_args()

start_time = datetime.now(timezone.utc)

for path in (args.scan1, args.scan2):
    if not os.path.isfile(path):
        print(f"Error: {path} not found.", file=sys.stderr)
        sys.exit(1)

try:
    primary_header, primary = core.load_scan(args.scan1)
    secondary_header, secondary = core.load_scan(args.scan2)
except (ValueError, OSError) as e:
    print(f"Error loading scan: {e}", file=sys.stderr)
    sys.exit(1)

n1 = sum(len(d["files"]) for d in primary.values())
n2 = sum(len(d["files"]) for d in secondary.values())
print(f"Loaded scan1: {args.scan1}  ({len(primary)} sessions, {n1} files)")
print(f"Loaded scan2: {args.scan2}  ({len(secondary)} sessions, {n2} files)")

index1 = core.build_index(primary)
index2 = core.build_index(secondary)
all_paths = sorted(set(index1.keys()) | set(index2.keys()))

merged_results = {}
stats = core.summarize_stats(index1, index2)

for path in all_paths:
    p = index1.get(path)
    s = index2.get(path)
    session, file_name = os.path.split(path)
    file_entry = {"file_name": file_name}

    if p is not None and s is not None:
        if core._digest_match(p, s):
            file_entry["md5"] = p.get("digest")
            file_entry["source"] = "both"
        else:
            file_entry["md5"] = p.get("digest")
            file_entry["md5_secondary"] = s.get("digest")
            file_entry["source"] = "both_modified"
    elif p is not None:
        file_entry["md5"] = p.get("digest")
        file_entry["source"] = "primary"
    else:
        file_entry["md5"] = s.get("digest")
        file_entry["source"] = "secondary"

    key = session or ""
    merged_results.setdefault(key, {"files": []})["files"].append(file_entry)

stop_time = datetime.now(timezone.utc)

summary = {k: v for k, v in stats.items() if k != "algorithm_mismatch"}
output = {
    "header": {
        "primary_scan": primary_header,
        "secondary_scan": secondary_header,
        "merged_at": stop_time.isoformat(),
        "summary": summary,
        "algorithm_mismatch": stats.get("algorithm_mismatch", False),
    },
    "results": merged_results,
}

output_path = args.output if args.output else "merged_comparison.json"
out_dir = os.path.dirname(os.path.abspath(output_path))
os.makedirs(out_dir, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

total_files = sum(len(d["files"]) for d in merged_results.values())
elapsed = (stop_time - start_time).total_seconds()
print(f"\nMerged {total_files} files ({len(merged_results)} sessions) -> {output_path}")
print(f"  Identical:      {stats['identical']}")
print(f"  Modified:       {stats['modified']}")
print(f"  Primary only:   {stats['primary_only']}")
print(f"  Secondary only: {stats['secondary_only']}")
if stats.get("algorithm_mismatch"):
    print("  WARNING: some files were compared across different hash algorithms.")
print(f"  Elapsed: {elapsed:.1f}s")
