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
