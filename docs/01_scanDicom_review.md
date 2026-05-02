# 01_scanDicom.py Review

**Last updated:** 2026-05-01
**Status:** Clean — implementation is stable; two architectural trade-offs remain (documented below).
**Test coverage:** 33/33 tests pass across unit, full, and integration suites.

---

## Summary

Scans a directory tree for MRI DICOM files, selects one representative file per series, extracts 22 metadata fields via `DICOM.DICOMextract`, and writes the result to `Data_table.csv`. Supports parallel processing, checkpoint/resume, and HPC array-job mode.

**Pipeline stages:**
1. `_find_all_dicom_dirs_impl` — recursive walk, verifies DICM magic bytes + `Modality == 'MR'`
2. `_find_dicom_worker` — per-directory series discovery; magic-byte pre-filter; optional sampling with full-scan fallback
3. `_extractDicom_impl` — instantiates `DICOMextract` and collects 22 fields into a dict
4. DataFrame assembly + atomic CSV write (tmp file + `os.replace`)

Configuration is encapsulated in the `ScanConfig` dataclass. No module-level globals; `cfg` and `logger` flow through the pipeline as arguments.

---

## Remaining Issues

### Hardcoded `/FL_system/` path defaults

`ScanConfig` defaults `scan_dir` to `/FL_system/data/raw/` and `save_dir` to `/FL_system/data/`. Running the script on a different machine without explicit arguments results in confusing file-not-found errors.

**Options:**
- Make `--scan_dir` and `--save_dir` required in argparse
- Default to `os.getcwd()` for a portable fallback
- Leave as-is (production environment is `/FL_system/`)

### Thread-based parallelism for CPU-bound work

Both pipeline stages dispatch via `P_type='thread'` through `toolbox.run_function` (lines 438, 462). pydicom header parsing has CPU cost that could benefit from process-based parallelism on multi-core hardware. This is a performance trade-off, not a bug.

**Recommendation:** Benchmark `thread` vs `process` on representative data volumes; pin the better mode and document the rationale.

---

## Test Coverage

| Suite | Tests | Status |
|---|---|---|
| `test_scanDicom_unit.py` | 6 | Passing |
| `test_scanDicom_full.py` (Group A: detection, Group B: extraction) | 26 | Passing |
| `test_scanDicom_integration.py` | 1 | Passing |
| **Total** | **33** | **33/33** |

### Coverage gaps
- Checkpoint resume (`--resume`) logic
- HPC array-job compilation path (`--dir_idx`)
- Profiling flag (`--profile`) end-to-end
- Concurrent/multi-process execution scenarios
