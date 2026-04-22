# Test Suite Overview

This directory contains the full test suite for the MRI preprocessing pipeline.
Tests are organized by **what** they verify and **how deeply** they exercise the code.

## Quick Start

```bash
# Run all tests
pytest test/ -v

# Run only unit tests (fastest, ~6s)
pytest test/test_scanDicom_unit.py -v

# Run comprehensive/functional tests (~0.5s, creates realistic DICOM files)
pytest test/test_scanDocom_full.py -v

# Run known-result tests (deterministic pipeline verification, ~0.3s)
pytest test/test_synthetic_known_result.py -v

# Run end-to-end integration test
pytest test/test_scanDocom_integration.py -v --integration

# Run only a single group within test_scanDicom_full.py
pytest test/test_scanDocom_full.py -k "Group A" -v
pytest test/test_scanDocom_full.py -k "Group B" -v
pytest test/test_scanDocom_full.py -k "Group C" -v
pytest test/test_scanDocom_full.py -k "Group D" -v

# Run a single test
pytest test/test_scanDocom_unit.py::test_find_all_dicom_dirs_single -v
pytest test/test_synthetic_known_result.py::TestScript01_Compilation::test_synth_csv_row_count -v
```

## Test Directory Structure

```
test/
├── __init__.py                          # Marks this directory as a Python package
├── conftest.py                          # Shared fixtures: synthetic DICOM file generators
├── generate_synthetic_datatable.py      # Deterministic (seed=42) Data_table.csv generator
├── synthetic_Data_table.csv             # Pre-generated expected output (320 rows, 20 sessions)
├── TESTS.md                             # This file
│
├── test_scanDicom_unit.py               # Unit tests for 01_scanDicom.py (6 tests)
├── test_scanDocom_integration.py        # Integration test for 01_scanDicom.py (1 test)
├── test_scanDocom_full.py               # Comprehensive tests for 01 + 02 (24 tests)
├── test_synthetic_known_result.py       # Known-result tests for 01 + 02 (58 tests)
```

## Test Files

### `test_scanDicom_unit.py` (6 tests)
**Coverage:** `01_scanDicom.py` only -- isolated unit tests for each public function.

| # | Test | File Under Test | What It Verifies |
|---|------|-----------------|------------------|
| 1 | `test_find_all_dicom_dirs_single` | `find_all_dicom_dirs()` | 1 MR file in 1 sub-directory is discovered |
| 2 | `test_findDicom_series` | `findDicom()` | One file per MR series; CT excluded |
| 3 | `test_extractDicom_basic` | `extractDicom()` | Returns dict with string `Modality` |
| 4 | `test_find_all_dicom_dirs_ignores_non_mr` | `find_all_dicom_dirs()` | CT+garbage returns 0 MRI dirs |
| 5 | `test_findDicom_handles_unreadable` | `findDicom()` | Skips corrupt files, returns valid MR file |
| 6 | `test_findDicom_sampling_is_deterministic` | `findDicom()` | Fixed seed produces identical results |

**When to use:** Fast feedback during development. Runs in ~1 second. These tests create
minimal synthetic DICOM files (only modality + series_number) and do not exercise the
full `DICOMextract()` class.

---

### `test_scanDicom_integration.py` (1 test)
**Coverage:** `01_scanDicom.py` only -- end-to-end pipeline.

| # | Test | What It Verifies |
|---|------|------------------|
| 1 | `test_end_to_end_small` | `find_all_dicom_dirs()` → `findDicom()` → `extractDicom()` → non-empty DataFrame |

**When to use:** Verify the full chain (file I/O → module loading → DataFrame construction) works
together. This test is tagged `@pytest.mark.integration` so it can be skipped in CI if needed

---

### `test_scanDicom_full.py` (24 tests)
**Coverage:** `01_scanDicom.py` **and** `02_parseDicom.py` -- comprehensive functional tests
using realistic synthetic DICOM files.

**Group A: `01_scanDicom.py` -- DICOM detection (10 tests)**

| # | Test | Scenario |
|---|------|----------|
| A1 | `test_A1_find_all_dicom_dirs_single` | Single MR directory |
| A2 | `test_A2_mixed_dir_only_mr_found` | MR + CT + non-DICOM files |
| A3 | `test_A3_nested_dirs` | Deeply nested directories |
| A4 | `test_A4_missing_series_number_no_crash` | Missing SeriesNumber tag |
| A5 | `test_A5_duplicate_series_returns_one` | 5 files, same series_number → 1 result |
| A6 | `test_A6_corrupt_files` | Good MR + 3 corrupt .dcm files |
| A7 | `test_A7_no_dcm_extension_ignored` | .jpg files ignored |
| A8 | `test_A8_sampling_deterministic` | Random sampling with fixed seed |
| A9 | `test_A9_empty_directory` | Empty directory → empty list |
| A10 | `test_A10_non_mr_modalities_not_returned` | CT, MRNS, US, CR, XA, NM, PT, RX, RTSTRUCT |

**Group B: `01_scanDicom.py` -- Metadata extraction (3 tests)**

| # | Test | Scenario |
|---|------|----------|
| B1 | `test_B1_extractDicom_has_all_keys` | All 22 expected output keys present |
| B2 | `test_B2_T1_vs_T2_modality` | RepetitionTime <780→T1, >=780→T2 (with boundary tests) |
| B3 | `test_B3_unknown_fields_missing_tags` | Missing tags → 'Unknown' |

**Group C: `02_parseDicom.py` -- Sequence isolation (8 tests)**

| # | Test | Scenario |
|---|------|----------|
| C1 | `test_C1_pure_t1_sequence` | All T1 rows preserved |
| C2 | `test_C2_mixed_t1_t2` | T2 removed, 2 T1 remain |
| C3a | `test_C3a_DISCO_steady_state_many` | DISCO removed when >=3 steady-state |
| C3b | `test_C3b_DISCO_few_steady_state` | DISCO kept when <3 steady-state |
| C4 | `test_C4_multiple_sessions` | Unique SessionID per patient+date |
| C5 | `test_C5_pre_post_trigger_time` | Pre/post via TriTime |
| C6 | `test_C6_pre_post_series_desc` | Pre/post via series description |
| C7 | `test_C7_ordering` | Scan ordering by TriTime + AcqTime |
| C8 | `test_C8_slices_consistency_post` | NumSlices preserved |

**Group D: `02_parseDicom.py` -- Edge cases (4 tests)**

| # | Test | Scenario |
|---|------|----------|
| D1 | `test_D1_filter_empty_dataframe` | Empty input → AssertionError |
| D2 | `test_D2_few_scans` | <2 scans handled gracefully |
| D3 | `test_D3_all_computed` | COMPUTED images removed |
| D4 | `test_D4_all_T1` | CT+MR mix → only T1 retained |

**When to use:** Before any PR. Verifies both scripts work correctly under realistic conditions.
Requires realistic DICOM attributes (RepetitionTime, NumSlices, laterality, etc.).

---

### `test_synthetic_known_result.py` (58 tests)
**Coverage:** `01_scanDicom.py` **and** `02_parseDicom.py` -- deterministic known-result testing.

These tests verify **exact, predetermined outputs** from `synthetic_Data_table.csv` (seed=42).

**TestGroup 1: `TestScript01_Compilation` -- 01 scanDicom output schema (10 tests)**

| # | Test | What It Verifies |
|---|------|------------------|
| 1 | `test_synth_csv_row_count` | Exactly 320 rows |
| 2 | `test_synth_csv_has_required_columns` | All 23 columns present |
| 3 | `test_synth_csv_no_null_rows` | No nulls in critical columns |
| 4 | `test_synth_csv_all_modalities_t1_t2_or_unknown` | Only valid modalities |
| 5 | `test_synth_csv_session_composition` | 20 unique sessions |
| 6 | `test_synth_csv_20_sessions` | Exactly 20 sessions |
| 7 | `test_synth_csv_series_desc_variety` | Realistic series descriptions |
| 8 | `test_synth_csv_tri_time_has_numeric_values` | Mix of Unknown + numeric TriTime |
| 9 | `test_synth_csv_has_pre_and_post` | Every session has pre+post contrast |
| 10 | `test_synth_csv_t2_rows_exist` | T2 rows exist (for filter removal verif.) |

**TestGroup 2: `TestScript02_Filtering` -- 02 parseDicom exact counts (38 tests)**

| # | Test | What It Verifies |
|---|------|------------------|
| 1-20 | `test_filter_remaining_row_count` | Exact rows per session after `removeT2()` |
| 21-40 | `test_filter_all_remainig_are_t1` | All remaining are T1 per session |
| 41 | `test_filter_removes_all_t2` | Zero T2 rows remain |
| 42 | `test_filter_total_expected_rows` | Sum equals predicted total |
| 43 | `test_filter_preserves_schema_columns` | All 23 columns preserved |
| 44 | `test_filter_preserves_session_id_col` | SessionID present |
| 45 | `test_filter_removes_correct_session` | removeT2() keeps exactly the T1 count |

**TestGroup 3: `TestSyntheticDataIntegrity` -- synthetic CSV integrity (4 tests)**

| # | Test | What It Verifies |
|---|------|------------------|
| 1 | `test_synthetic_data_is_deterministic` | 320 rows, 20 unique IDs (no drift) |
| 2 | `test_synthetic_data_modality_distribution` | T1 and T2 both present |
| 3 | `test_synthetic_data_has_predefined_series_desc` | ≥8 common keywords present |
| 4 | `test_synthetic_data_has_varied_tri_times` | Mix of Unknown and numeric TriTime |

**How known values are computed:**
1. `generate_synthetic_datatable.py` creates `synthetic_Data_table.csv` (seed=42, 320 rows, 20 sessions)
2. `DICOMfilter.removeT2()` removes every row where Modality is `'T2'` or `'Unknown'`
3. Row counts per session are manually verified and stored in `EXPECTED_SESSIONS`
4. Each test compares actual output against these expected values

**When to use:** Before any data-flow change. Catches drift in either the synthetic generator
or the processing logic with specific, actionable failure messages.

---

## Shared Test Infrastructure

### `conftest.py` -- Fixtures and Helpers

Provides utilities for creating synthetic DICOM files without real patient data.
All generated DICOM files are modern-format compliant (pydicom `write_like_original=False`).

| Helper | Purpose |
|--------|---------|
| `make_minimal_dcm(path, ...)` | Minimal DICOM: modality + series_number + patient_id |
| `make_realistic_mr_dcm(path, ...)` | Realistic MR with all commonly used attributes |
| `make_t1_mr_dcm(path, ...)` | Convenience wrapper: RT=450.0 |
| `make_t2_mr_dcm(path, ...)` | Convenience wrapper: RT=850.0 |
| `make_dwi_mr_dcm(path, ...)` | Convenience wrapper: DWI b-value |
| `create_test_dicom_directory(base, configs)` | Creates a dir with multiple DICOM files |
| `create_test_study_structure(tmp, configs)` | Creates multi-study directory structures |

---

## Synthetic Data

### `generate_synthetic_datatable.py` + `synthetic_Data_table.csv`

`generate_synthetic_datatable.py` produces `synthetic_Data_table.csv` deterministically
(random.seed(42), np.random.seed(42)).

- **320 rows**, **20 sessions**
- Each session has: locator/scout rows, pre-contrast T1, optional non-fat-sat T1, PJN (injection),
  6-12 post-contrast T1, optional MIP, optional T2, optional Dixon water, optional DWI+ADC, optional STIR
- Modality distribution: ~76% T1, ~24% T2
- Laterality distribution: ~88% Unknown, ~6% right, ~6% left, ~1% bilateral

**To regenerate synthetic data:**
```bash
cd test/
python generate_synthetic_datatable.py
```

---

## CI/CD

Tests run automatically on every push/PR via GitHub Actions (`.github/workflows/tests.yml`).

- Runs on Python 3.10, 3.11, 3.12
- Runs all 4 test suites
- Branches: all pushed branches + any PR to main/develop

Manual run:
```bash
pytest test/test_scanDicom_unit.py test/test_scanDocom_full.py test/test_synthetic_known_result.py -v
```
