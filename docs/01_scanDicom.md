# 01_scanDicom — DICOM Inventory & Metadata Extraction

## Purpose

Recursively scans a directory tree for MRI DICOM files, selects one representative file per series, extracts 24 metadata fields via `DICOM.DICOMextract`, and writes the cumulative result to `Data_table.csv`.

This is **step 1 of 6** in the MRI preprocessing pipeline. All subsequent steps (parsing, NIfTI conversion, orientation correction, coregistration, and model input generation) depend on this file as their source of truth for what exists in the scan directory.

---

## What It Does

The script executes four logical stages:

1. **Directory discovery & representative selection** — A single `_find_and_select_impl` walk traverses the tree once. Each `.dcm` read confirms MR modality and registers a series representative, eliminating redundant `pyd.dcmread()` calls vs. a two-pass design. When `--multi` is set, `_scan_subdir` splits the tree into disjoint subtrees proportional to core count × 4 so parallel `os.walk()` calls don't contend over the same filesystem nodes.
2. **Parallel dispatch** — Hybrid workers (`ProcessPoolExecutor` wrapping inner `ThreadPoolExecutor`) walk trees independently via `_find_dicom_worker`. Process-level parallelism avoids GIL contention for CPU-bound `dcmread`; inner threads saturate I/O bandwidth per subtree.
3. **Metadata extraction** — `_extractDicom_impl` instantiates `DICOMextract` per representative file and collects 24 DICOM header fields (PatientID, Orientation, TriggerTime, SeriesDescription, Modality, etc.) into a dict. Any individual failure produces `None` and is skipped — one bad file doesn't abort the run.
4. **Output** — Dict list → `pd.DataFrame` → atomic CSV write (`.tmp` → `os.replace`). On success, any leftover checkpoint files are cleaned up automatically.

---

## Configuration

All runtime parameters flow through the `ScanConfig` dataclass and are set via CLI flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--scan_dir` | path | `/FL_system/data/raw/` | Root directory to recursively scan for DICOM files (`.dcm`) |
| `--save_dir` | path | `/FL_system/data/` | Output directory for `Data_table.csv` and checkpoints |
| `--multi`, `-m` | int | off | Enable hybrid multiprocessing with N CPUs (default: max-1) |
| `--test` | int? | off | Limit scan to at most N directories (default 100); operates at directory boundaries, never mid-session |
| `--sample_pct` | float | 0.0 | Percent of `.dcm` files to read per directory (0 = full scan) |
| `--sample_seed` | int? | None | Seed for reproducible sampling |
| `--resume` | flag | off | Resume from the last checkpoint instead of starting fresh |
| `--checkpoint_dir` | path? | `<save_dir>/checkpoints/` | Override checkpoint storage location (useful if raw data lives on slow/shared storage) |
| `--dir_idx` | int? | None | Index into `dirs_to_process.pkl` for HPC array-job execution |
| `--dir_list` | path | `dirs_to_process.pkl` | Path to pickled directory list used by HPC mode |
| `--profile`, `-p` | flag | off | Run with yappi profiler; outputs `<profile_dir>/step01_profile.yappi` |
| `--profile_dir` | path? | `<save_dir>` | Override profile storage location |

---

## Outputs

| File | Location | Description |
|------|----------|-------------|
| `Data_table.csv` | `<save_dir>/` | Primary output: one row per DICOM series with 24 metadata columns |
| `Data_table_<idx>.csv` | `<save_dir>/tmp/` | Per-worker output in HPC array mode; concatenated automatically into `Data_table.csv` on last-job completion |
| `scan_and_select.pkl` | `<checkpoint_dir>/` | Checkpoint for discovery + selection stage (survives crash, cleaned up on success) |
| `info.pkl` | `<checkpoint_dir>/` | Auxiliary checkpoint preserving directory metadata for resume |

---

## HPC Array Execution

The script supports embarrassingly-parallel array-job submissions. Each worker processes one directory from a pre-generated list:

```bash
# Generate the directory list once (normal run or custom script)
python 01_scanDicom.py --scan_dir /path/to/raw --save_dir /path/to/data

# Submit as an HPC array job (SLURM example)
sbatch <<'EOF'
#SBATCH --array=0-99
python 01_scanDicom.py --dir_idx $SLURM_ARRAY_TASK_ID \
  --save_dir /path/to/data --scan_dir /placeholder
EOF
```

**Compile guard:** Workers signal completion via sentinel markers (`.done.<idx>`) in the temporary output directory. The last-index job (`dir_idx == len(dirs) - 1`) waits for all sentinels with a per-worker timeout of **60 seconds**. An exclusive `fcntl` file lock ensures only one worker runs the CSV concatenation step. If the timeout expires, the compile is skipped and the run exits gracefully — partial worker CSVs remain in `<save_dir>/tmp/` for manual recovery.

---

## Checkpoint & Resume Behavior

Checkpoints are written atomically every 10 batches during parallel extraction. On resume:
- The discovery + selection stage restores from `scan_and_select.pkl` if valid, skipping the full directory walk.
- Extraction resumes from the last successfully checkpointed batch.
- On successful completion all checkpoints are removed automatically.

If `<save_dir>/checkpoints/` cannot be created (e.g., permission issue on shared storage), an error is logged and checkpoints fall back to `<save_dir>/`. This pollutes output with `.pkl` files, but does not corrupt data.

---

## Performance Notes

| Factor | Impact | Recommendation |
|--------|--------|----------------|
| `--multi` flag | **Critical** — without it both scanning and extraction are serial | Always use `--multi` for production runs. Typical 4–16× improvement on multi-core machines. |
| Hybrid parallelism | Correct pairing: processes handle I/O tree walks; threads saturate dcmread bandwidth per subtree | Default is correct for DICOM workloads. |
| `--sample_pct > 0` | Reduces dcmread calls proportionally | Use only for development/testing. Set to `0` (default) for production scans. |
| External checkpoint disk | Checkpoints on slow/shared storage degrade batch-commit speed | Use `--checkpoint_dir` pointing to a fast local scratch mount when available. |

---

## Testing

| Suite | Tests | Scope |
|-------|-------|-------|
| `test/test_scanDicom_unit.py` | 6 | Path resolution, config parsing, logger creation, checkpoint I/O |
| `test/test_scanDicom_full.py` (Groups A–B) | 26 | Full directory scanning + metadata extraction |
| `test/test_scanDicom_integration.py` | 1 | End-to-end pipeline behavior with synthetic data |

**Total:** 33/33 passing.

### Coverage gaps

- **Checkpoint resume (`--resume`)** — requires real `.pkl` files on disk; slow to set up in CI. Risk: low (code path is simple file I/O with try/except everywhere).
- **HPC array-job compile (`--dir_idx`)** — requires scheduler environment and multiple job instances. Risk: low (sentinel-based guard matches step 02, which has been battle-tested).
- **Profiling (`--profile`)** — optional yappi dependency absent in test env. Risk: negligible.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Data_table.csv already exists` error at start | Previous run completed; output still present | Delete `<save_dir>/Data_table.csv` to re-run, or adjust `--save_dir`. |
| Workers timeout during HPC compile | A worker crashed before writing its sentinel marker | Check logs for failed workers, fix root cause (`dcmread` crashes usually), then re-run the specific `--dir_idx`. |
| Checkpoints appear in `<save_dir>/` instead of subdirectory | Permission denied writing `checkpoints/` subdir | Log will note this; run with writable permissions or use `--checkpoint-dir` on a local path. |
