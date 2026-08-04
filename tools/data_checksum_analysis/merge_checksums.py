"""
Merge two checksum scan result JSON files into a single comparison file.

Builds a per-file index from both scans and classifies every file as
identical, modified, primary_only, or secondary_only. Writes a merged
JSON with statistics and per-session file listings.

Usage:
  python merge_checksums.py <scan1.json> <scan2.json> [-o OUTPUT.json]
"""
import json
import os
from argparse import ArgumentParser
from datetime import datetime, timezone

parser = ArgumentParser(description="Merge two checksum scan result JSON files into a single comparison file.")
parser.add_argument("scan1", help="Path to the first scan result JSON file (primary/source).")
parser.add_argument("scan2", help="Path to the second scan result JSON file (secondary/destination).")
parser.add_argument("-o", "--output", help="Output file path. Defaults to merged_comparison.json in the current directory.")
args = parser.parse_args()

start_time = datetime.now(timezone.utc)

with open(args.scan1, 'r') as f:
    scan1 = json.load(f)
with open(args.scan2, 'r') as f:
    scan2 = json.load(f)

print(f"Loaded scan1: {args.scan1}  ({len(scan1['results'])} sessions, {sum(len(d['files']) for d in scan1['results'].values())} files)")
print(f"Loaded scan2: {args.scan2}  ({len(scan2['results'])} sessions, {sum(len(d['files']) for d in scan2['results'].values())} files)")

index1 = {}
for session, data in scan1['results'].items():
    for f in data['files']:
        key = os.path.join(session, f['file_name'])
        index1[key] = f['md5']

index2 = {}
for session, data in scan2['results'].items():
    for f in data['files']:
        key = os.path.join(session, f['file_name'])
        index2[key] = f['md5']

all_paths = sorted(set(index1.keys()) | set(index2.keys()))

merged_results = {}
stats = {"identical": 0, "modified": 0, "primary_only": 0, "secondary_only": 0}

for path in all_paths:
    md5_1 = index1.get(path)
    md5_2 = index2.get(path)

    session, file_name = os.path.split(path)
    file_entry = {"file_name": file_name}

    if md5_1 is not None and md5_2 is not None:
        if md5_1 == md5_2:
            file_entry["md5"] = md5_1
            file_entry["source"] = "both"
            stats["identical"] += 1
        else:
            file_entry["md5"] = md5_1
            file_entry["md5_secondary"] = md5_2
            file_entry["source"] = "both_modified"
            stats["modified"] += 1
    elif md5_1 is not None:
        file_entry["md5"] = md5_1
        file_entry["source"] = "primary"
        stats["primary_only"] += 1
    else:
        file_entry["md5"] = md5_2
        file_entry["source"] = "secondary"
        stats["secondary_only"] += 1

    if session not in merged_results:
        merged_results[session] = {"files": []}
    merged_results[session]["files"].append(file_entry)

stop_time = datetime.now(timezone.utc)

output = {
    "header": {
        "primary_scan": scan1["header"],
        "secondary_scan": scan2["header"],
        "merged_at": stop_time.isoformat(),
        "summary": stats,
    },
    "results": merged_results,
}

output_path = args.output if args.output else "merged_comparison.json"
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

total_files = sum(len(d["files"]) for d in merged_results.values())
print(f"\nMerged {total_files} files ({len(merged_results)} sessions) -> {output_path}")
print(f"  Identical:      {stats['identical']}")
print(f"  Modified:       {stats['modified']}")
print(f"  Primary only:   {stats['primary_only']}")
print(f"  Secondary only: {stats['secondary_only']}")
print(f"  Elapsed: {(stop_time - start_time).total_seconds():.1f}s")
