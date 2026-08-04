# Data Checksum Analysis Toolkit

Verify data transfers between locations by computing and comparing MD5 checksums for session directories.

## Workflow

The typical workflow is:

1. **Scan source** — compute checksums of the original location.
2. **Scan destination** — compute checksums of the transfer target.
3. **Compare** — identify which sessions are verified, which need re-transfer, and which are missing.
4. **Digest** — convert the comparison report into plain-text session ID lists.
5. **Act** — move or remove sessions based on the digested lists.

## Scripts

| Script | Purpose |
|---|---|
| `scan_dest.py` | Scan a directory tree, compute MD5 for every file, save results as JSON in `scan_results/`. Supports `--skip` (repeatable) and `--skip-file` to exclude sessions. |
| `extract_sessions.py` | Extract session directory names from a scan result JSON into a plain-text file (one ID per line). |
| `compare_checksum.py` | Interactively compare two scan results. Classifies sessions into *need transfer*, *ready for deletion*, and *missing from primary*. Writes report JSON to `comparison_findings/`. |
| `digest_comparison.py` | Convert a comparison report JSON into plain-text files (one session ID per line) in `comparison_findings/`. |
| `merge_checksums.py` | Merge two scan JSON files into a single per-file comparison with statistics. |
| `move_sessions.py` | Move session directories from source to destination based on a plain-text ID list. Supports `--dry-run`. |
| `remove_sessions.py` | Delete session directories from a target path based on a plain-text ID list. Supports `--dry-run`. |

## Usage

```bash
# 1. Scan the source directory
python scan_dest.py /path/to/source --output source_scan.json
# → outputs scan_results/source_scan.json

# 2. Scan the destination directory (skip certain sessions)
python scan_dest.py /path/to/destination --skip tmp --skip backup --output dest_scan.json
# → outputs scan_results/dest_scan.json

# 3. Compare the two scans (interactive)
python compare_checksum.py
# → outputs comparison_findings/comparison_report_*.json

# 4. Digest the report into plain-text session lists
python digest_comparison.py comparison_findings/comparison_report_0_vs_1.json

# 4b. Extract all sessions from a scan (alternative to digesting)
python extract_sessions.py scan_results/scan_results_0.json

# 5a. Move sessions that still need transfer
python move_sessions.py comparison_findings/sessions_need_transfer.txt /path/to/source /path/to/destination --dry-run

# 5b. Remove sessions that are verified
python remove_sessions.py comparison_findings/sessions_ready_for_deletion.txt /path/to/source --dry-run
```

## Directory Structure

```
scan_results/                    # JSON checksum scans
comparison_findings/             # Comparison reports and digested session lists
```
