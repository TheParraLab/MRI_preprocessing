# Data Checksum Analysis Toolkit

Verify data transfers between locations by computing and comparing checksums for session directories. The directories being compared are NOT simultaneously accessible: the source scan JSON is the authoritative *source of truth*, it ships to the destination, the destination does a **bounded** local status check (hashing only the sessions listed in the manifest), the result ships back, and the source deletes / re-sends accordingly.

## Workflow (state-carrier)

1. **Scan at source** — `scan_dest.py` produces a manifest JSON in `scan_results/`.
2. **Ship the manifest** to the destination machine (single small JSON).
3. **Bounded status check at destination** — `confirm_local.py` hashes only the sessions listed in the manifest and emits `confirmed.txt`, `stale.txt`, `absent.txt` and `manifest_status.json`.
4. **Ship the three session lists back** (very small).
5. **At source**: `remove_sessions.py confirmed.txt` for verified sessions; `move_sessions.py stale.txt absent.txt` for re-transfer.
6. **Verify (optional)** — after re-transfer, `confirm_local.py manifest.json local_dir --verify sessions.txt` re-hashes each listed session and emits `verify_report.json`.
7. **Cross-machine compare (optional)** — if you do have both scans, `compare_checksum.py primary.json secondary.json` produces the legacy report JSON + session lists under `comparison_findings/`.

## Scripts

| Script | Purpose |
|---|---|
| `checksum_core.py` | Shared pure-stdlib primitives: `hash_file`, `scan_tree`, `load_scan`, `build_index`, `summarize_stats`, `classify_sessions`, `bounded_status`, `alloc_scan_name`, `write_session_list`. |
| `scan_dest.py` | Scan a directory tree and write a manifest JSON into `scan_results/`. Supports `--skip`, `--skip-file`, `--output`, `--hash {sha256,md5}`, `--workers`. |
| `confirm_local.py` | Bounded local status check against a manifest. Emits `confirmed.txt`/`stale.txt`/`absent.txt`/`manifest_status.json`. `--list-all` adds `unlisted_present.txt`, `--emit-details` inlines per-file details. `--verify sessions.txt` re-hashes and emits `verify_report.json`. |
| `extract_sessions.py` | Extract session IDs from a scan result JSON into a plain-text file (one ID per line). |
| `compare_checksum.py` | Non-interactive: `python compare_checksum.py <primary.json> <secondary.json> [-o OUTDIR]`. Interactive (backward-compat): no positional args, picks from `scan_results/`. Writes report JSON + `sessions_*.txt` lists into the output directory. |
| `digest_comparison.py` | Convert a comparison report JSON into plain-text files (one session ID per line) in `comparison_findings/` (created if missing). |
| `merge_checksums.py` | Merge two scan JSONs into a single per-file comparison with statistics. |
| `move_sessions.py` | Move session directories from source to destination based on a plain-text ID list. Supports `--dry-run`, `--yes`. |
| `remove_sessions.py` | Delete session directories from a target path based on a plain-text ID list. Supports `--dry-run`, `--yes`. |

## Hash algorithm & the algorithm-mismatch caveat

- `scan_dest.py` defaults to `--hash sha256`; pass `--hash md5` for backward-compatible manifests. Each file entry always writes `md5` (equal to the `digest` value), so legacy readers/consumers that read `f['md5']` keep working.
- **Caveat:** a new `sha256` scan of the same file will **NOT** match an old `md5` manifest of that file — digests differ across algorithms. Comparisons must be like-to-like. If both entries carry an `algo` field and they differ, `summarize_stats` counts the pair as `modified` and sets `algorithm_mismatch=True` in the stats dict; `classify_sessions` marks the session as `stale`.

## Exit codes

- `scan_dest.py` — `1` on bad input / invalid directory.
- `confirm_local.py`
  - `0` on success (any actionable state).
  - `1` bounded status check raised.
  - `2` manifest file or local directory not found.
  - `3` `--verify` mode: at least one session FAILED re-hash.
- `compare_checksum.py`
  - `0` clean (no stale, no missing).
  - `1` any stale or missing.
  - `2` file not found / load failure.
- `digest_comparison.py` — `1` on missing report.
- `move_sessions.py` / `remove_sessions.py` — `0` on success or abort; `1` on missing input.
- All scripts create their output directories automatically (`os.makedirs(exist_ok=True)`); none of them will fail if `scan_results/` or `comparison_findings/` is absent.

## Usage

```bash
# 1. Scan the source directory (manifest is the source of truth)
python scan_dest.py /path/to/source --output source_manifest.json

# 2. Ship source_manifest.json to the destination machine and run a bounded check
python confirm_local.py source_manifest.json /path/to/local
# -> source_manifest_dir/confirmed.txt, stale.txt, absent.txt, manifest_status.json

# 2b. Optional: also list local sessions that aren't in the manifest
python confirm_local.py source_manifest.json /path/to/local --list-all

# 2c. Optional: include per-file detail in manifest_status.json
python confirm_local.py source_manifest.json /path/to/local --emit-details

# 3. After re-transfer, verify the moved sessions at the destination
python confirm_local.py source_manifest.json /path/to/local --verify confirmed.txt

# 4. If two scans are available, do a full source-vs-destination compare
python compare_checksum.py scan_results/primary.json scan_results/secondary.json

# 4b. Interactive legacy mode: pick two scans from scan_results/
python compare_checksum.py
# -> comparison_findings/comparison_report_*.json + sessions_*.txt

# 5. Digest a legacy report into plain-text session lists
python digest_comparison.py comparison_findings/comparison_report_x.json

# 6. Move / remove sessions based on the produced lists
python move_sessions.py stale.txt /source /destination --dry-run
python remove_sessions.py confirmed.txt /source --yes
```

## File entry shape (backward compatible)

```json
{
  "file_name": "01.nii.gz",
  "algo": "sha256",
  "digest": "abc123...",
  "size_bytes": 12345,
  "md5": "abc123..."
}
```

`md5` is always equal to `digest` so any older reader keyed on `f['md5']` keeps working.

## Directory structure

```
scan_results/                    # JSON checksum scans (created automatically)
comparison_findings/             # Comparison reports and digested session lists (created automatically)
```

## Follow-ups

- **Hoist `checksum_core.py` to a shared location** (e.g. `code/tools/_shared/checksum_core.py`) once the number of consumers grows beyond this folder. Currently `code/tools/pipeline_review/review_03.py` imports it across via a `sys.path.insert` (review_03.py:32), which works but smells. When restructuring: move the module, bump the import-path line in `scan_dest.py`, `compare_checksum.py`, `digest_comparison.py`, `extract_sessions.py`, `merge_checksums.py`, `confirm_local.py`, `review_03.py`, and the two test files (`code/test/test_checksum_core.py`, `code/test/test_confirm_local.py`). Keep the public API stable so no other changes are needed.
