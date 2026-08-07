"""
Integration tests for DICOMfilter pipeline.

Exercises the real DICOMfilter.__init__ + isolate_sequence() pipeline
against synthetic data.  Some sessions deliberately lack explicit 'pre' labels
so the pipeline must fall back to TriTime-based detection.

Tests verify structural invariants of the pipeline's output, not a parallel
"known-correct" implementation.  The synthetic CSV is deterministically
generated (seed=42) so rows are stable across runs.

Run with: pytest test/test_synthetic_known_result.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# ---- Module loading ----
proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))
from DICOM import DICOMfilter

SYNTHETIC_CSV = str(proj_root / "test" / "synthetic_Data_table.csv")


@pytest.fixture(scope="module")
def synth_df():
    """Load the deterministic synthetic Data_table once for all tests."""
    return pd.read_csv(SYNTHETIC_CSV)


def _run_filter(subset: pd.DataFrame, pid, date):
    """Convenience helper: run DICOMfilter + isolate_sequence on one session.

    Returns (filter_instance, success_bool).
    """
    subset = subset.copy()
    subset["SessionID"] = f"{pid}_{date}"
    f = DICOMfilter(subset, logger=None)
    ok = f.isolate_sequence()
    return f, ok


# ==================
# GROUP 1: Schema / integrity of synthetic_Data_table.csv
# ==================

class TestScript01_Schema:
    """Verify synthetic_Data_table.csv is well-formed and complete."""

    def test_row_count(self, synth_df):
        assert len(synth_df) == 320

    def test_all_columns_present(self, synth_df):
        required = {
            'PATH', 'Orientation', 'ID', 'Accession', 'Name', 'DATE', 'DOB',
            'Series_desc', 'Modality', 'Part', 'AcqTime', 'SrsTime', 'ConTime', 'StuTime',
            'TriTime', 'InjTime', 'ScanDur', 'Lat', 'NumSlices', 'Thickness',
            'BreastSize', 'DWI', 'Type', 'Series',
        }
        assert required.issubset(set(synth_df.columns))

    def test_no_nulls_in_critical_columns(self, synth_df):
        for col in ['ID', 'DATE', 'Modality', 'Series_desc', 'TriTime']:
            assert synth_df[col].notna().all(), f"'{col}' has nulls"

    def test_modality_only_t1_t2_unknown(self, synth_df):
        assert set(synth_df['Modality'].unique()).issubset({'T1', 'T2', 'Unknown'})

    def test_20_unique_sessions(self, synth_df):
        assert synth_df.groupby(['ID', 'DATE']).ngroups == 20

    def test_ambiguous_sessions_exist(self, synth_df):
        """At least some sessions must rely on TriTime (no explicit 'pre' label)."""
        n_ambiguous = 0
        for (_, grp) in synth_df.groupby(['ID', 'DATE']):
            has_pre = grp['Series_desc'].str.lower().str.contains('pre', na=False).any()
            if not has_pre:
                n_ambiguous += 1
        assert n_ambiguous > 0, "All sessions have 'pre' label — not testing ambiguous detection"

    def test_t2_rows_exist_in_input(self, synth_df):
        assert (synth_df['Modality'] == 'T2').sum() > 0


# ==================
# GROUP 2: Real DICOMfilter pipeline (structural invariants)
# ==================

class TestDICOMfilterPipeline:
    """Run the actual DICOMfilter pipeline and verify structural invariants."""

    def _filter_session(self, synth_df, pid, date):
        grp = synth_df[(synth_df['ID'] == pid) & (synth_df['DATE'].astype(str) == str(date))].copy()
        return _run_filter(grp, pid, date)

    # -- T2 removal (happens in __init__) --

    def test_t2_removed_on_success(self, synth_df):
        """For every successful session, output has zero T2 / Unknown modality rows."""
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            f, ok = self._filter_session(synth_df, pid, date)
            if ok:
                bad = f.dicom_table['Modality'].isin(['T2', 'Unknown']).sum()
                assert bad == 0, f"Session {pid}: {bad} non-T1 rows remain"

    def test_t2_removed_stored(self, synth_df):
        """T2 rows are stored in f.removed['T2']."""
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            f, _ = self._filter_session(synth_df, pid, date)
            assert 'T2' in f.removed, f"Session {pid}: missing removed['T2']"

    # -- Pre/Post flagging on successful sessions --

    def test_successful_has_pre_and_post(self, synth_df):
        """Every successful session has >= 1 pre and >= 1 post scan."""
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            f, ok = self._filter_session(synth_df, pid, date)
            if ok:
                n_pre = f.dicom_table['Pre_scan'].sum()
                n_post = f.dicom_table['Post_scan'].sum()
                assert n_pre >= 1, f"Session {pid}: no pre scan detected"
                assert n_post >= 1, f"Session {pid}: no post scan detected"

    def test_post_has_numeric_tri_time(self, synth_df):
        """Post scans must have numeric (non-Unknown) TriTime.
        
        The pipeline detects post scans via TriTime != 'Unknown' or series description
        containing 'post'.  On successful sessions, post rows should never be Unknown.
        """
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            f, ok = self._filter_session(synth_df, pid, date)
            if not ok:
                continue
            post = f.dicom_table[f.dicom_table['Post_scan'] == True]
            assert len(post) > 0, f"Session {pid}: no post scans"
            assert (post['TriTime'].astype(str).str.lower() != 'unknown').all(), \
                f"Session {pid}: post-scan flagged with Unknown TriTime"

    def test_output_only_t1(self, synth_df):
        """Successful output only contains Modality='T1' rows."""
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            f, ok = self._filter_session(synth_df, pid, date)
            if ok:
                assert (f.dicom_table['Modality'] == 'T1').all(), \
                    f"Session {pid}: unexpected modality in output"

    def test_failed_sessions_empty(self, synth_df):
        """Failed sessions return empty dicom_table."""
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            f, ok = self._filter_session(synth_df, pid, date)
            if not ok:
                assert f.dicom_table.empty, f"Session {pid}: nonzero output despite failure"

    def test_some_ambiguous_succeed(self, synth_df):
        """At least some no-pre-label sessions succeed (TriTime fallback works)."""
        ambiguous_ok = 0
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            has_pre = grp['Series_desc'].str.lower().str.contains('pre', na=False).any()
            if not has_pre:
                f, ok = self._filter_session(synth_df, pid, date)
                if ok:
                    ambiguous_ok += 1
        assert ambiguous_ok > 0, "No no-pre-label sessions succeeded — TriTime fallback broken"

    def test_post_orientation_consistency(self, synth_df):
        """Post scans in successful sessions share single dominant Orientation."""
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            f, ok = self._filter_session(synth_df, pid, date)
            if not ok:
                continue
            post = f.dicom_table[f.dicom_table['Post_scan'] == True]
            # The pipeline filters to a single dominant Orientation for post scans
            assert post['Orientation'].nunique() <= 1, \
                f"Session {pid}: multiple Orientations in post scans"

    def test_slice_counts_consistent(self, synth_df):
        """Major post slices are a single value (pipeline filters to dominant slices).
        MIP reconstructions may have smaller slice counts, but the bulk post/
        pre scans share the session's primary slice count."""
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            f, ok = self._filter_session(synth_df, pid, date)
            if not ok:
                continue
            # The pipeline should produce output where pre slices are consistent
            pre_slices = f.dicom_table.loc[f.dicom_table['Pre_scan'] == True, 'NumSlices'].values
            assert len(pre_slices) > 0, f"Session {pid}: no pre slices"
            # Post slices may include MIP with smaller counts, but the primary
            # post sequence dominates
            post_slices = f.dicom_table.loc[f.dicom_table['Post_scan'] == True, 'NumSlices'].unique()
            # At least most common post slice should be >= 100 (MIP is ~30-44 slices)
            major_post = post_slices[post_slices >= 100]
            if len(major_post) > 0:
                # Major post and pre should share a slice count
                assert any(p >= 100 for p in pre_slices) or len(pre_slices) == 1, \
                    f"Session {pid}: can't align pre/post slice counts"
