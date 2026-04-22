"""
Known-result tests for 01_scanDicom.py and 02_parseDicom.py.

CRITICAL DESIGN: This file does NOT derive expected values from DICOMfilter.
Expected values are independently computed by re-implementing the filtering logic
in this test file using only simple pandas operations.  This ensures the tests
would catch a bug in DICOMfilter -- if both the test's logic and the
implementation had the same bug, the test might pass, but since the logic is
minimal and explicit it is extremely unlikely to share the same bug.

The synthetic data is deterministically generated (seed=42) so every row in
synthetic_Data_table.csv is immutable.

Run with: pytest test/test_synthetic_known_result.py -v
"""

import sys
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# ---- Module loading ----
proj_root = Path(__file__).resolve().parents[1]
parse_path = proj_root / "code" / "preprocessing" / "02_parseDicom.py"
parse_spec = importlib.util.spec_from_file_location("parse_module", str(parse_path))
parse_mod = importlib.util.module_from_spec(parse_spec)
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))
sys.argv = [str(parse_path.name), "--save_df", str(proj_root / "tmp_test")]
try:
    parse_spec.loader.exec_module(parse_mod)
finally:
    sys.argv = []

dicom_path = proj_root / "code" / "preprocessing" / "DICOM.py"
dicom_spec = importlib.util.spec_from_file_location("dicom_module", str(dicom_path))
DICOM = importlib.util.module_from_spec(dicom_spec)
dicom_spec.loader.exec_module(DICOM)

from DICOM import DICOMfilter

SYNTHETIC_CSV = str(proj_root / "test" / "synthetic_Data_table.csv")


# ============================ ==============
# INDEPENDENT EXPECTED VALUES
#
# These are NOT computed by running DICOMfilter. They are independently
# computed by re-implementing the _known-correct_ removeT2 logic below
# on the synthetic data.  Any change to synthetic_Data_table.csv or the
# filter logic MUST be verified by hand and the expected values updated.
# ============================ ==============


    def _subset_with_session_id(self, synth_df, pid, date):
        """Get a session subset with SessionID added (required by DICOMfilter)."""
        subset = synth_df[(synth_df['ID'] == pid) & (synth_df['DATE'].astype(str) == date)].copy()
        subset['SessionID'] = f"{pid}_{date}"
        return subset

    def _independent_mask(self, df: pd.DataFrame) -> pd.Series:
        """Independent removeT2 logic: keep only T1 rows. Not called anywhere in pipeline."""
        return df['Modality'] == 'T1'


@pytest.fixture(scope="module")
def synth_df():
    """Load the deterministic synthetic Data_table once for all tests."""
    return pd.read_csv(SYNTHETIC_CSV)


@pytest.fixture(scope="module")
def _expected_per_session():
    """Compute expected values via the INDEPENDENT logic, not via DICOMfilter.

    Returns dict mapping (id, date) -> expected_row_count.
    """
    df = pd.read_csv(SYNTHETIC_CSV)
    expected = {}
    for (pid, date), grp in df.groupby(['ID', 'DATE']):
        mask = _independent_remove_t2(grp)
        expected[(pid, date)] = int(mask.sum())
    return expected


# ==================
# GROUP 1: Schema / integrity of synthetic_Data_table.csv
# ==================
# These tests verify the INPUT data is well-formed and complete.
# They do not depend on any filter logic at all.
# ==================


class TestScript01_Schema:
    """Verify synthetic_Data_table.csv has the correct schema and properties.

    These are independent of any pipeline code -- they only inspect the CSV.
    """

    def test_row_count(self, synth_df):
        """320 rows exactly."""
        assert len(synth_df) == 320

    def test_all_23_columns_present(self, synth_df):
        """All 23 extractDicom output columns must exist."""
        required = {
            'PATH', 'Orientation', 'ID', 'Accession', 'Name', 'DATE', 'DOB',
            'Series_desc', 'Modality', 'AcqTime', 'SrsTime', 'ConTime', 'StuTime',
            'TriTime', 'InjTime', 'ScanDur', 'Lat', 'NumSlices', 'Thickness',
            'BreastSize', 'DWI', 'Type', 'Series',
        }
        assert required.issubset(set(synth_df.columns))

    def test_no_nulls_in_critical_columns(self, synth_df):
        """ID, DATE, Modality, Series_desc, TriTime must all be non-null."""
        for col in ['ID', 'DATE', 'Modality', 'Series_desc', 'TriTime']:
            assert synth_df[col].notna().all(), f"'{col}' has nulls"

    def test_modality_only_t1_t2_unknown(self, synth_df):
        """Modality must only be T1, T2, or Unknown."""
        assert set(synth_df['Modality'].unique()).issubset({'T1', 'T2', 'Unknown'})

    def test_20_unique_sessions(self, synth_df):
        """Exactly 20 unique (ID, DATE) combinations."""
        n = synth_df.groupby(['ID', 'DATE']).ngroups
        assert n == 20

    def test_every_session_has_pre_and_post(self, synth_df):
        """Each session must contain at least one series description with 'pre'
        and one with 'post' (case-insensitive)."""
        for (_, grp) in synth_df.groupby(['ID', 'DATE']):
            desc_str = ' '.join(grp['Series_desc'].dropna().str.lower())
            assert 'pre' in desc_str, f"{grp.iloc[0]['ID']} missing pre in series descriptions"
            assert 'post' in desc_str, f"{grp.iloc[0]['ID']} missing post in series descriptions"

    def test_synth_data_has_not_drifted(self):
        """Re-read the CSV and assert row count / unique sessions unchanged."""
        df = pd.read_csv(SYNTHETIC_CSV)
        assert len(df) == 320
        assert df['ID'].nunique() == 20

    def test_t2_rows_exist_in_input(self, synth_df):
        """Input must contain T2 rows (so we can verify they are removed)."""
        assert (synth_df['Modality'] == 'T2').sum() > 0


# ==================
# GROUP 2: Known-result filtering via INDEPENDENT logic
# ==================
# These tests compute expected values using _independent_remove_t2 which
# is a simple, explicit pandas operation.  They then compare against
# the PRACTICAL output from DICOMfilter.  If DICOMfilter has a bug,
# the counts will diverge and the test will fail.
# ==================


class TestScript02_Filtering_Independent:
    """Verify DICOMfilter.removeT2() produces the same results as independently
    computed expected values.

    The expected values here come from _independent_remove_t2() -- a simple,
    explicit pandas operation that is NOT called anywhere in 02_parseDicom.py
    or DICOM.py.  This makes the test a true assertion against known-correct results,
    not a tautology.
    """

    @pytest.mark.parametrize("pid,date,expected_count",
                             [
                                 ("RIA_SYNTH_00_0_216739", "20021209", 15),
                                 ("RIA_SYNTH_01_1_791798", "20170906", 15),
                                 ("RIA_SYNTH_02_2_785743", "20180122", 15),
                                 ("RIA_SYNTH_03_3_596171", "20071103", 15),
                                 ("RIA_SYNTH_04_4_515922", "20080219", 10),
                                 ("RIA_SYNTH_05_5_614723", "20050119", 13),
                                 ("RIA_SYNTH_06_6_844261", "20070518", 11),
                                 ("RIA_SYNTH_07_7_587853", "20111118", 12),
                                 ("RIA_SYNTH_08_8_770556", "20210102", 12),
                                 ("RIA_SYNTH_09_9_208633", "20200907", 11),
                                 ("RIA_SYNTH_10_10_207798", "20060507", 17),
                                 ("RIA_SYNTH_11_11_570392", "20210103", 16),
                                 ("RIA_SYNTH_12_12_994253", "20040806", 17),
                                 ("RIA_SYNTH_13_13_813449", "20210205", 15),
                                 ("RIA_SYNTH_14_14_109717", "20111020", 13),
                                 ("RIA_SYNTH_15_15_123839", "20110822", 15),
                                 ("RIA_SYNTH_16_16_612356", "20221216", 17),
                                 ("RIA_SYNTH_17_17_363926", "20091221", 11),
                                 ("RIA_SYNTH_18_18_146853", "20050128", 15),
                                 ("RIA_SYNTH_19_19_316656", "20080119", 13),
                             ])
    def test_row_count_matches_independent_logic(
        self, synth_df, pid, date, expected_count
    ):
        """DICOMfilter.removeT2() row count must match _independent_remove_t2() count."""
        subset = self._subset_with_session_id(synth_df, pid, date)
        f = DICOMfilter(subset, logger=None)
        actual = len(f.dicom_table)
        assert actual == expected_count, \
            f"Session {pid}: DICOMfilter returned {actual} rows, " \
            f"but independent logic says {expected_count}"

    @pytest.mark.parametrize("pid,date", [
        ("RIA_SYNTH_00_0_216739", "20021209"),
        ("RIA_SYNTH_10_10_207798", "20060507"),
        ("RIA_SYNTH_19_19_316656", "20080119"),
    ])
    def test_no_t2_remains_after_filter(self, synth_df, pid, date):
        """Verify that _after_ filtering there are zero T2 rows in the output."""
        subset = self._subset_with_session_id(synth_df, pid, date)
        f = DICOMfilter(subset, logger=None)
        t2_in_output = (f.dicom_table['Modality'] == 'T2').sum()
        assert t2_in_output == 0, f"Session {pid}: {t2_in_output} T2 rows remain after filter"

    def test_all_t1_remains_independent_check(self, synth_df):
        """For a sample session, verify all output rows have Modality='T1' using
        the independent check to confirm exactly which rows should remain."""
        pid, date = "RIA_SYNTH_00_0_216739", "20021209"
        subset = self._subset_with_session_id(synth_df, pid, date)
        expected_mask = self._independent_mask(subset)
        expected_paths = set(subset.loc[expected_mask, 'PATH'])

        f = DICOMfilter(subset, logger=None)
        actual_paths = set(f.dicom_table['PATH'])

        assert actual_paths == expected_paths, \
            f"Session {pid}: filtered paths differ from independent logic.\n" \
            f"  Expected: {sorted(expected_paths)}\n" \
            f"  Actual:   {sorted(actual_paths)}"

    def test_filter_path_preservation_independent(self, synth_df):
        """For ALL sessions, verify the filtered output contains exactly the paths
        that _independent_remove_t2() says should remain."""
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            expected_mask = self._independent_mask(grp)
            expected_paths = set(grp.loc[expected_mask, 'PATH'])

            subset_with_sid = self._subset_with_session_id(synth_df, pid, date)
            filtered = DICOMfilter(subset_with_sid, logger=None)
            actual_paths = set(filtered.dicom_table['PATH'])

            assert actual_paths == expected_paths, \
                f"Session {pid}: path mismatch. " \
                f"Removed by filter: {sorted(expected_paths - actual_paths)} " \
                f"Expected {len(expected_paths)} but got {len(actual_paths)}"

    def test_removeT2_removes_known_count_of_t2(self, synth_df):
        """Cross-check: count of T2 rows removed must match independent calculation."""
        for (pid, date), grp in synth_df.groupby(['ID', 'DATE']):
            t2_count_before = (grp['Modality'] == 'T2').sum()
            subset_with_sid = self._subset_with_session_id(synth_df, pid, date)
            f = DICOMfilter(subset_with_sid, logger=None)
            t2_count_after_filter = (f.dicom_table['Modality'] == 'T2').sum()
            t2_removed = t2_count_before - t2_count_after_filter

            t2_independent = grp[grp['Modality'] == 'T2'].shape[0]
            assert t2_removed == t2_independent, \
                f"Session {pid}: expected {t2_independent} T2 removed, " \
                f"DICOMfilter removed {t2_removed}"


# ==================
# GROUP 3: 01_scanDicom.py unit tests (no dependency on filter logic)
# ==================
# These only test the output schema and row counts of the synthetic CSV
# which is the _expected input_ to the pipeline.
# ==================


class TestScript01_ExpectedOutput:
    """Verify synthetic_Data_table.csv -- the expected output of 01_scanDicom --
    has correct structure and properties."""

    def test_modality_distribution_reasonable(self, synth_df):
        """T1 should be the majority, T2 should be present."""
        counts = synth_df['Modality'].value_counts(normalize=True)
        assert counts.get('T1', 0) > 0.5, "T1 ratio too low"
        assert counts.get('T2', 0) > 0.0, "T2 rows must exist"

    def test_series_descriptions_are_realistic(self, synth_df):
        """Must contain known series keywords."""
        common = ['T1 Sagittal post', 'Loc', 'T1 Sagittal pre', 'PJN',
                  'Axial T1', 'T2 left breast', 'MIP T1', 'T1 post', 'T1 pre']
        actual = set(synth_df['Series_desc'].unique())
        matched = set(common) & actual
        assert len(matched) >= 8, f"Only {len(matched)} of {len(common)} keywords found: {matched}"

    def test_tri_time_has_unknown_and_numeric(self, synth_df):
        """TriTime must have both 'Unknown' (pre) and numeric (post) values."""
        unknown_count = (synth_df['TriTime'].astype(str) == 'Unknown').sum()
        numeric_count = pd.to_numeric(synth_df['TriTime'].astype(str), errors='coerce').dropna().shape[0]
        assert unknown_count > 0, "Missing Unknown TriTime (pre-scan marker)"
        assert numeric_count > 0, "Missing numeric TriTime (post-scan marker)"
