"""
Unit tests for DICOMfilter.detect_fs() — fat-saturation classification from
Series_desc, validated against spellings observed in the production corpus
(data/Data_table.csv, ~1.08M T1 rows / 350k unique descriptions).

Coverage goals
--------------
1. Non-fat-saturated descriptions are ALWAYS classified False (never True or
   NaN): explicit negation with every observed negator/spelling.
2. Fat-saturated descriptions classified False by the old patterns (f/s, T1FS)
   are now positively detected.
3. Conflict resolution: an explicit negation overrides a positive token.
4. False-positive safety: unrelated "no"/"non" usage ("(no delay)", "non-gad")
   and similar-looking tokens (FSPGR/fatigue) never fire the negation rules.
5. Unmarked descriptions remain NaN (the gate — not detect_fs — decides their
   fate, see removeNonFSScans()).

Run with: pytest code/test/test_detect_fs.py -v
"""

import sys
from pathlib import Path

import logging
import numpy as np
import pandas as pd
import pytest

proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))
from DICOM import DICOMfilter


def _classify(descs):
    """Run the real detect_fs() on one synthetic session and return its
    FatSaturated column."""
    df = pd.DataFrame({
        'Series_desc': list(descs),
        'Modality': ['T1'] * len(descs),
        'SessionID': ['S1'] * len(descs),
        'Lat': ['Unknown'] * len(descs),
        'NumSlices': [160] * len(descs),
        'Type': ["['ORIGINAL', 'PRIMARY']"] * len(descs),
    })
    f = DICOMfilter(df)
    assert f.dicom_table.shape[0] == len(descs), "test T1 rows must survive removeT2"
    f.detect_fs()
    return f.dicom_table['FatSaturated'].tolist()


def _one(desc):
    return _classify([desc])[0]


# ---------------------------------------------------------------------------
# 1. Non-fat-saturated — MUST be False (unambiguous, per protocol requirement)
# ---------------------------------------------------------------------------

NEGATIVE_DESCRIPTIONS = [
    # Top-of-corpus spellings ('T1 non fat sat' = ~25.6k rows in production)
    "T1 non fat sat",
    "T1 left breast non fat sat",
    "T1 right breast non fat sat",
    "Axial Non FS",
    "T1 right breast non fs",
    # Negator 'not' — the 557-row misclassification this fix addresses
    "T1 Sagittal not fat sat",
    "T1 not FS post",
    # Hyphen / typo variants observed in production
    "T1 right breast non-fat sat",
    "T1 left non fat satt",
    "T1 right breast non fat ssat",
    "Sag PRE  NON FAT SAT",
    "T1 non fat sat UNI",
    # 'no' + standalone 'fat' (corpus idiom, approved as non-FS)
    "T1 right breast no fat",
    "T1 left breast no fat",
    "T1 left no fat",
    # Legacy standalone forms
    "Axial NNFS post",
    "WOFS axial t1 post",
    "T1 ONLY",
]


@pytest.mark.parametrize("desc", NEGATIVE_DESCRIPTIONS)
def test_negated_desc_is_false(desc):
    assert _one(desc) is False or _one(desc) == False, f"{desc!r} must classify as non-fat-saturated"


# ---------------------------------------------------------------------------
# 2. Conflict resolution — explicit negation wins over a positive token
# ---------------------------------------------------------------------------

def test_negation_wins_over_positive():
    # "Vibrant" is a positive (GE FS family), explicit "Non FS" overrides it
    assert _one("Axial Vibrant Pre Non FS") == False


@pytest.mark.parametrize("desc", ["Sag T1 non f/s post RIGHT", "T1 non fs pre", "without fat sat t1 post"])
def test_negation_with_alt_spellings_is_false(desc):
    # Slash forms and "without" also carry unambiguous non-FS intent
    assert _one(desc) is False or _one(desc) == False, f"{desc!r} must classify as non-fat-saturated"


def test_t1_only_even_with_fs_wording():
    assert _one("T1 only") == False


# ---------------------------------------------------------------------------
# 3. Fat-saturated — positively detected, incl. previously-missed spellings
# ---------------------------------------------------------------------------

POSITIVE_DESCRIPTIONS = [
    # Standard forms (must remain working)
    "Axial T1 FS pre",
    "Ph1/Axial T1 FS post",
    "T1 Sagittal post fat sat",
    "WATER: Ph1/AX, T1 FS, MULTI, DISCO",
    "FAT: Ph2/AX, T1 FS, MULTI, DISCO",
    "OPT AX FS FSPGR",                      # FS word + FSPGR family
    "T1 SPAIR post",
    # f/s spelling (~5.4k production rows previously dropped)
    "Sag T1 f/s post delay RIGHT",
    "Sag T1 f/s RIGHT pre",
    "Axial T1 left breast f/s",
    "T1 F/S post",
    # no-space spelling (previously dropped)
    "WATER: Ph2/WATER AX, T1FS, MULTI DISCO",
    "Ph1/Axial T1FS post",
]


@pytest.mark.parametrize("desc", POSITIVE_DESCRIPTIONS)
def test_positive_desc_is_true(desc):
    assert _one(desc) is True, f"{desc!r} must classify as fat-saturated"


# ---------------------------------------------------------------------------
# 4. Safety — similar-looking text must NOT fire the rules
# ---------------------------------------------------------------------------

def test_no_delay_does_not_negate():
    # "(no delay)" contains 'no' but is not a fat-sat negation; positive FS token stands
    assert _one("Ph2/ Axial T1 FS pre/post (no delay)") is True


def test_non_gad_negative_prefix_is_not_fs_negation():
    # 'non-' appears as an unrelated prefix (non-gadolinium etc.); no fs/fat term follows
    desc = "T1 pre non-gad"
    assert np.isnan(_one(desc)), f"{desc!r} must stay unmarked, not False"


def test_fatty_is_not_a_negation():
    # 'fatigue'/'fatty'-like text without an actual negator must not fire rule 2
    assert np.isnan(_one("T1 Axial (fatigued)"))


def test_fspgr_alone_is_not_fs():
    # FSPGR is a gradient-recalled family name, not fat saturation
    assert np.isnan(_one("OPT AX FSPGR")) or _one("OPT AX FSPGR") is False


# ---------------------------------------------------------------------------
# 5. Unmarked descriptions stay NaN (ambiguous — gate's jurisdiction)
# ---------------------------------------------------------------------------

AMBIGUOUS_DESCRIPTIONS = [
    "T1 Sagittal post",
    "Axial T1",
    "PJN",
    "LOC",
    "Localization",
    "3-Plane Loc",
    "T1 right breast",
    "Axial T1 AP",
    "WATER: Ph1/DISCO",
    "SHORT TEMP RES",          # STIR not guessed from the description
    "T1 post",
]


@pytest.mark.parametrize("desc", AMBIGUOUS_DESCRIPTIONS)
def test_unmarked_desc_is_nan(desc):
    assert np.isnan(_one(desc)), f"{desc!r} must remain unmarked"


# ---------------------------------------------------------------------------
# 6. Gate policy (removeNonFSScans) — switch behaviour on a mixed session
# ---------------------------------------------------------------------------

def _session(descs, monkeypatch, policy=None):
    """Build a session and apply the FS gate under an explicit policy."""
    if policy is not None:
        monkeypatch.setenv('TREAT_UNKNOWN_AS_SATURATED', policy)
    df = pd.DataFrame({
        'Series_desc': list(descs),
        'Modality': ['T1'] * len(descs),
        'SessionID': ['S1'] * len(descs),
        'Lat': ['Unknown'] * len(descs),
        'NumSlices': [160] * len(descs),
        'Type': ["['ORIGINAL', 'PRIMARY']"] * len(descs),
    })
    f = DICOMfilter(df)
    kept = f.removeNonFSScans()
    return f, kept


def _kept_descs(kept):
    return set(kept['Series_desc'].astype(str))


MIXED_SESSION = [
    "Axial T1 FS pre",          # True  - always kept
    "T1 non fat sat",           # False - ALWAYS dropped
    "T1 Sagittal not fat sat",  # False (regression) - ALWAYS dropped
    "T1 Sagittal post",         # NaN (ambiguous)
]


def test_gate_default_is_lenient(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.delenv('TREAT_UNKNOWN_AS_SATURATED', raising=False)
    f, kept = _session(MIXED_SESSION, monkeypatch, policy=None)
    got = _kept_descs(kept)
    assert got == {"Axial T1 FS pre", "T1 Sagittal post"}, got
    # audit line present with counts
    assert any('[FS gate] lenient' in r.message and 'pos=1 neg=2 unknown_kept=1' in r.message for r in caplog.records), \
        [r.message for r in caplog.records if 'FS gate' in r.message]


def test_gate_strict_drops_unmarked(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    f, kept = _session(MIXED_SESSION, monkeypatch, policy='false')
    got = _kept_descs(kept)
    assert got == {"Axial T1 FS pre"}, got
    assert any('[FS gate] strict' in r.message and 'unknown_dropped=1' in r.message for r in caplog.records), \
        [r.message for r in caplog.records if 'FS gate' in r.message]


def test_gate_false_is_unconditional(monkeypatch):
    # Explicit non-FS must never survive, whatever the policy says
    f_l, kept_l = _session(MIXED_SESSION, monkeypatch, policy='true')
    f_s, kept_s = _session(MIXED_SESSION, monkeypatch, policy='false')
    for kept in (kept_l, kept_s):
        assert not kept['Series_desc'].str.contains('non fat sat|not fat sat', case=False).any()


def test_gate_no_marked_rows_warns(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    # Session entirely unmarked under lenient policy: kept, with a warning naming the policy
    f, kept = _session(["T1 Sagittal post", "LOC"], monkeypatch, policy='true')
    assert len(kept) == 2
    assert any('[FS gate] No positively fat-saturated marked scans' in r.message for r in caplog.records), \
        [r.message for r in caplog.records if 'FS gate' in r.message]


def test_policy_flag_parsing(monkeypatch):
    cls = DICOMfilter
    for val, expect in [('true', True), ('TRUE', True), ('1', True), ('yes', True),
                        ('false', False), ('0', False), ('no', False), ('off', False)]:
        monkeypatch.setenv('TREAT_UNKNOWN_AS_SATURATED', val)
        assert cls._treat_unknown_as_saturated() is expect, f"{val!r} parsed as {expect}"
    monkeypatch.delenv('TREAT_UNKNOWN_AS_SATURATED', raising=False)
    assert cls._treat_unknown_as_saturated() is True, "default must be lenient"


# ---------------------------------------------------------------------------
# 7. Shared primitives: scalar classify_fs + vectorised type semantics
# ---------------------------------------------------------------------------

def test_scalar_classify_fs_matches_series():
    from DICOM import classify_fs, classify_fs_series
    descs = ["Axial T1 FS pre", "T1 non fat sat", "Sag T1 f/s RIGHT", "T1FS post",
             "T1 Sagittal not fat sat", "LOC"]
    s = classify_fs_series(pd.Series(descs))
    for d, v in zip(descs, s):
        scalar = classify_fs(d)
        if np.isnan(v):
            assert scalar is None, f"{d!r}: series NaN but scalar {scalar!r}"
        else:
            assert bool(scalar) is bool(v), f"{d!r}: series {v!r} vs scalar {scalar!r}"


def test_scalar_classify_fs_none_for_unmarked():
    from DICOM import classify_fs
    assert classify_fs("Axial T1") is None
    assert classify_fs(None) is None
    assert classify_fs("") is None


def test_classify_fs_series_object_semantics():
    """True/False must stay Python bools and unknowns NaN (not float 1.0/0.0),
    so downstream 'is True' / == False checks work against the raw column."""
    from DICOM import classify_fs_series
    s = classify_fs_series(pd.Series(["T1 FS", "non fs", "LOC"]))
    assert s[0] is True
    assert s[1] is False
    assert np.isnan(s[2])


def test_classify_fs_series_preserves_index():
    from DICOM import classify_fs_series
    idx = [5, 9, 3]
    s = classify_fs_series(pd.Series(["T1 FS", "LOC", "non fs"], index=idx))
    assert s.index.tolist() == idx


# ---------------------------------------------------------------------------
# 8. Ordering: dual-pre sessions must not silently collide Major values
# ---------------------------------------------------------------------------

def _ordered_table(n_pre, logger=None):
    from DICOM import DICOMorder
    n = n_pre + 3
    d = pd.DataFrame({
        'SessionID': ['S1'] * n,
        'TriTime': ['Unknown'] * n_pre + ['20032', '20110', '20200'],
        'AcqTime': [str(x) for x in range(1, n + 1)],
        'Pre_scan': [True] * n_pre + [False] * 3,
        'Post_scan': [False] * n_pre + [True] * 3,
        'Series_desc': [f'pre {i}' for i in range(n_pre)] + ['post a', 'post b', 'post c'],
        'NumSlices': [160] * n,
    })
    o = DICOMorder(d, logger=logger or logging.getLogger('order'))
    return o.order('TriTime', secondary_param='AcqTime')


def test_order_single_pre_has_unique_majors():
    out = _ordered_table(n_pre=1)
    majors = sorted(int(x) for x in out['Major'])
    assert majors == [0, 1, 2, 3]


def test_order_dual_pre_emits_collision_warning(caplog):
    caplog.set_level(logging.WARNING)
    out = _ordered_table(n_pre=2)
    majors = sorted(int(x) for x in out['Major'])
    assert majors.count(0) == 2, "both pre scans must occupy Major 0"
    assert any('share Major' in r.message for r in caplog.records), \
        [r.message for r in caplog.records if r.levelno >= logging.WARNING]
