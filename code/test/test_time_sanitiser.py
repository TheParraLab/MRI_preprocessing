"""
Regression tests for the time-value sanitiser introduced after the HPC
clinical run crashed with:

    could not convert string to float: "b'\\x16\\xba\\xa2L'"

The offending value is a raw-bytes blob stored as a CSV cell in
`Data_table_timing.csv` (TriTime / ScanDur / AcqTime column).  Two
defence-in-depth layers now protect us:

1.  ``DICOM.DICOMextract._clean_time`` — single-choke-point validation
    applied in ``Acq / Srs / Con / Stu / Tri / Inj`` getters, and a
    matching guard in ``ScanDur``.
2.  ``06_genInputs._clean_timing`` — converts any residual bad value
    (from a legacy CSV written before fix #1 was deployed) to 'Unknown'
    so the existing fallback path handles it.

Run with: pytest code/test/test_time_sanitiser.py -v
"""

import importlib.util
import sys
from pathlib import Path

proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))

# ---- load DICOM module directly (not via 01) so we can test _clean_time ----
_dcm_path = proj_root / "code" / "preprocessing" / "DICOM.py"
_dcm_spec = importlib.util.spec_from_file_location("dicom_under_test", str(_dcm_path))
DICOM = importlib.util.module_from_spec(_dcm_spec)
_dcm_spec.loader.exec_module(DICOM)

# ---- load 06 module for the _clean_timing helper ------------------------------
# 06 calls argparse.parse_args() at import time, which reads sys.argv.
# Swap in a neutral argv for the duration of the exec_module call.
_06_path = proj_root / "code" / "preprocessing" / "06_genInputs.py"
_06_spec = importlib.util.spec_from_file_location("six_under_test", str(_06_path))
six = importlib.util.module_from_spec(_06_spec)
_orig_argv = sys.argv
sys.argv = ["06_genInputs.py"]
try:
    _06_spec.loader.exec_module(six)
finally:
    sys.argv = _orig_argv


# ---------------------------------------------------------------------------
# DICOM.DICOMextract._clean_time
# ---------------------------------------------------------------------------
def _make_extractor_stub():
    """Create a DICOMextract instance with a stubbed .metadata so the
    constructor's dcmread is bypassed."""
    obj = object.__new__(DICOM.DICOMextract)
    obj.debug = 0
    return obj


def test_clean_time_valid_tm_string_passthrough():
    e = _make_extractor_stub()
    assert e._clean_time("123456") == "123456"


def test_clean_time_valid_fractional_tm_string_passthrough():
    e = _make_extractor_stub()
    assert e._clean_time("123456.789") == "123456.789"


def test_clean_time_whitespace_trims_and_strips():
    e = _make_extractor_stub()
    assert e._clean_time("  123456  ") == "123456"


def test_clean_time_none_returns_unknown():
    e = _make_extractor_stub()
    assert e._clean_time(None) == e.UNKNOWN


def test_clean_time_ascii_bytes_decode_success():
    e = _make_extractor_stub()
    # ASCII bytes that would decode cleanly AND satisfy the TM regex
    assert e._clean_time(b"123456") == "123456"


def test_clean_time_binary_bytes_blob_returns_unknown():
    """The exact value that crashed the HPC clinical run:
    b'\\x16\\xba\\xa2L' — raw binary that is not even valid UTF-8."""
    e = _make_extractor_stub()
    result = e._clean_time(b'\x16\xba\xa2L')
    assert result == "Unknown"


def test_clean_time_non_dict_string_returns_unknown():
    e = _make_extractor_stub()
    assert e._clean_time("hello") == e.UNKNOWN


def test_clean_time_float_input_returns_unknown():
    """float inputs are not valid TM strings — we don't silently coerce."""
    e = _make_extractor_stub()
    assert e._clean_time(1234.0) == e.UNKNOWN


def test_clean_time_float_nan_returns_unknown():
    e = _make_extractor_stub()
    assert e._clean_time(float('nan')) == e.UNKNOWN


# ---------------------------------------------------------------------------
# 06_genInputs._clean_timing — the legacy-CSV defence layer
# ---------------------------------------------------------------------------
def test_06_clean_timing_bytes_blob_returns_unknown():
    """Regression: the exact bytes repr that triggered the original crash.
    When pandas wrote a bytes cell to CSV, the cell came back as the *string*
    'b\"\\x16\\xba\\xa2L\"'.  _clean_timing must map it to 'Unknown'."""
    bad = "b'\\x16\\xba\\xa2L'"
    assert six._clean_timing(bad) == "Unknown"


def test_06_clean_timing_native_bytes_returns_unknown():
    # Bytes that pandas would have round-tripped as repr-string, and also
    # the literal bytes object (defensive in case a legacy CSV path passes
    # them through raw).
    assert six._clean_timing(b'\x16\xba\xa2L') == "Unknown"


def test_06_clean_timing_valid_numeric_string_passthrough():
    assert six._clean_timing("123456") == "123456"


def test_06_clean_timing_valid_float_passthrough():
    assert six._clean_timing(1234.5) == 1234.5


def test_06_clean_timing_Unknown_passthrough():
    # 'Unknown' is our sentinel — must survive unchanged so the existing
    # tri_all_unknown / Scan_Duration[0] == 'Unknown' branches fire.
    assert six._clean_timing("Unknown") == "Unknown"


def test_06_clean_timing_nan_returns_unknown():
    import math
    assert six._clean_timing(float('nan')) == "Unknown"


def test_06_clean_timing_none_returns_unknown():
    assert six._clean_timing(None) == "Unknown"


# ---------------------------------------------------------------------------
# Integration: verify the 5 time getters all route through _clean_time
# ---------------------------------------------------------------------------
class _NS:
    pass


def _capture_call(*, attribute):
    """Create a DICOMextract stub whose metadata carries a bad value for the
    given attribute, call the matching getter, and assert it returns Unknown
    (root-cause fix layer)."""
    meta = _NS()
    setattr(meta, attribute, b'\x16\xba\xa2L')  # the exact crashing payload

    e = object.__new__(DICOM.DICOMextract)
    e.debug = 0
    e.metadata = meta
    return e


def test_Tri_returns_unknown_for_binary_bytes():
    e = _capture_call(attribute='TriggerTime')
    assert e.Tri() == "Unknown"


def test_Acq_returns_unknown_for_binary_bytes():
    e = _capture_call(attribute='AcquisitionTime')
    assert e.Acq() == "Unknown"


def test_Srs_returns_unknown_for_binary_bytes():
    e = _capture_call(attribute='SeriesTime')
    assert e.Srs() == "Unknown"


def test_Con_returns_unknown_for_binary_bytes():
    e = _capture_call(attribute='ContentTime')
    assert e.Con() == "Unknown"


def test_Stu_returns_unknown_for_binary_bytes():
    e = _capture_call(attribute='StudyTime')
    assert e.Stu() == "Unknown"


def test_Inj_returns_unknown_for_binary_bytes():
    e = _capture_call(attribute='InjectionTime')
    assert e.Inj() == "Unknown"


def test_ScanDur_returns_unknown_for_binary_bytes():
    """ScanDur used to return the raw .value directly. With the fix,
    bytes b'\\x16\\xba\\xa2L' must come back as 'Unknown'."""

    class _StubMetaWithIndex:
        def __getitem__(self, key):
            class _elem:
                value = b'\x16\xba\xa2L'
            return _elem()

    e = object.__new__(DICOM.DICOMextract)
    e.debug = 0
    e.metadata = _StubMetaWithIndex()
    assert e.ScanDur() == "Unknown"


def test_ScanDur_passthrough_for_valid_float():
    class _StubMeta:
        def __getitem__(self, key):
            class _elem:
                value = 25000
            return _elem()

    e = object.__new__(DICOM.DICOMextract)
    e.debug = 0
    e.metadata = _StubMeta()
    assert e.ScanDur() == 25000.0
