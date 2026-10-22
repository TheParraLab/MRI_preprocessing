"""Regression tests for ISSUE-002: TriTime==0 baseline handling in detect_post.

The vendor stamps TriggerTime=0 on the pre-contrast baseline in some protocols
and on the first post phase (Ph1) in others. post_trigger_time() must treat
these differently — and only the description disambiguates:

  * pure-pre description + TriTime=0   -> pre-contrast baseline (NOT post)
  * pure-post description + TriTime=0  -> genuine post phase (stays post)
  * hybrid 'pre/post' + TriTime=0      -> post phase (stays post; pre resolves
    to the Unknown-TriTime reference series)

Run with: pytest code/test/test_detectpost_tri0_baseline.py -v
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))

from DICOM import DICOMfilter  # noqa: E402

NUMSLICES = 194


def _mk(pre_desc, ph1_desc='Ph1/Axial T1 FS post', ph2_desc='Ph2/Axial T1 FS post',
        ref_desc='Axial T1', n_post=3):
    """Minimal 01-table shape for one session: a reference series (Unknown
    TriTime), a TriTime=0 series (pre_desc), and n_post numeric post phases."""
    post_descs = [ph1_desc, ph2_desc, 'Ph3/Axial T1 FS post'][:n_post]
    tri = ['Unknown', '0'] + ['132446', '264894', '397341'][:n_post]
    descs = [ref_desc, pre_desc] + post_descs
    n = len(descs)
    return pd.DataFrame({
        'PATH': [f'/x/{i}.dcm' for i in range(n)],
        'Orientation': ['A'] * n,
        'ID': ['TST001'] * n,
        'DATE': ['20240101'] * n,
        'SessionID': ['TST001_20240101'] * n,
        'Series': list(range(1, n + 1)),
        'Series_desc': descs,
        'Modality': ['T1'] * n,
        'TriTime': tri,
        'Lat': ['Unknown'] * n,
        'NumSlices': [NUMSLICES] * n,
        'Type': [['ORIGINAL', 'PRIMARY', 'OTHER']] * n,
        'DWI': [None] * n,
    })


def _run(df):
    filt = DICOMfilter(df.copy(), logger=logging.getLogger('t'), tmp_save='/tmp')
    filt.Types(['slope', 'sub', 'subtract'])
    filt.Description(['loc', 'pjn', 'calib'])
    ok = filt.isolate_sequence()
    return filt, ok


def test_pure_pre_baseline_not_post():
    """ISSUE-002 core: TriTime=0 pure-pre baseline must become the pre, not a
    post phase. Posts must retain the numeric phases."""
    filt, ok = _run(_mk('FAT: Axial T1 PRE FS'))
    assert ok, f'isolate_sequence failed; removed={list(filt.removed.keys())}'
    out = filt.dicom_table
    assert len(out) == 4, f'expected 4 rows (1 pre + 3 post), got {len(out)}'
    pre = out[out['Pre_scan'].astype(str).isin(['True', '1'])]
    assert len(pre) == 1
    assert pre.iloc[0]['Series_desc'] == 'FAT: Axial T1 PRE FS', \
        'the TriTime=0 pure-pre baseline must be the labeled pre'
    assert str(pre.iloc[0]['TriTime']) == '0'
    posts = out[out['Post_scan'].astype(str).isin(['True', '1'])]
    assert len(posts) == 3
    assert all('post' in str(d).lower() for d in posts['Series_desc'].values)


def test_pure_post_desc_tri0_stays_post():
    """Guard against the REJECTED blanket fix: a pure-post description with
    TriTime=0 is a genuine post phase (Ph1) and must stay post."""
    filt, ok = _run(_mk('Ph1/Axial T1 POST FS', ph1_desc='Ph2/Axial T1 FS post',
                       ph2_desc='Ph3/Axial T1 FS post', ref_desc='AXIAL T1 PRE FS',
                       n_post=2))
    assert ok
    out = filt.dicom_table
    posts = out[out['Post_scan'].astype(str).isin(['True', '1'])]
    # all three T1 FS series are post (the TriTime=0 one included)
    assert len(posts) == 3, f'genuine post phase named 0 must stay post; got {len(posts)}'
    assert (posts['Series_desc'] == 'Ph1/Axial T1 POST FS').any()
    assert (posts['TriTime'].astype(str) == '0').any()


def test_hybrid_prep_post_tri0_stays_post():
    """Hybrid 'pre/post' baseline (vendor names the DCE dynamic both ways):
    the TriTime=0 series stays post; pre resolves to the Unknown reference."""
    filt, ok = _run(_mk('Axial T1 FS pre/post', ph1_desc='Ph1/Axial T1 FS pre/post',
                       ph2_desc='Ph2/Axial T1 FS pre/post'))
    assert ok
    out = filt.dicom_table
    pre = out[out['Pre_scan'].astype(str).isin(['True', '1'])]
    assert len(pre) == 1
    assert str(pre.iloc[0]['TriTime']) == 'Unknown', \
        'hybrid baseline must not become the pre'
    posts = out[out['Post_scan'].astype(str).isin(['True', '1'])]
    assert (posts['TriTime'].astype(str) == '0').any(), \
        'hybrid TriTime=0 series must stay in the post set'


def test_ria_ph1_pre_titled_stays_post():
    """RIA protocol quirk: TriTime=0 is stamped on the WHOLE session and the
    DCE dynamic is named 'pre' in the template. Ph1 (TriTime=0) is the bolus
    arrival — a POST-contrast phase — so the phase marker (Ph<digit>) must
    keep it in the post pool. The labeled pre is the Unknown-TriTime
    reference. This pins the user-confirmed domain fact that these sessions
    are post sequences despite the 'pre' descriptions."""
    df = _mk('WATER: Ph1/Axial T1 FS pre',
             ph1_desc='WATER: Ph2/Axial T1 FS pre',
             ph2_desc='WATER: Ph3/Axial T1 FS pre',
             ref_desc='WATER: Axial T1 FS pre', n_post=2)
    # numeric phases get real post-contrast trigger times (RIA stamps 0 only
    # on the first dynamic + the baseline)
    df.loc[df['Series_desc'] == 'WATER: Ph2/Axial T1 FS pre', 'TriTime'] = '124343'
    df.loc[df['Series_desc'] == 'WATER: Ph3/Axial T1 FS pre', 'TriTime'] = '248666'
    filt, ok = _run(df)
    assert ok
    out = filt.dicom_table
    pre = out[out['Pre_scan'].astype(str).isin(['True', '1'])]
    assert len(pre) == 1
    assert pre.iloc[0]['Series_desc'] == 'WATER: Axial T1 FS pre', \
        'the Unknown-TriTime reference must be the labeled pre'
    posts = out[out['Post_scan'].astype(str).isin(['True', '1'])]
    assert len(posts) == 3, f'all three Ph phases must stay post; got {len(posts)}'
    assert (posts['Series_desc'].values == 'WATER: Ph1/Axial T1 FS pre').any(), \
        'Ph1 (TriTime=0) is the bolus phase — must stay post despite the pre title'


def test_ria_numbered_post_dynamic_bolus_stays_post():
    """ISSUE-002 single-pre invariant, RIA template variant (b): the session
    has NO Ph marker but DOES contain a numbered post dynamic
    ('Axial T1 post 1..N'). The TriTime=0 'Axial T1 pre' series is then
    phase-0 of that dynamic (the bolus) and must stay post — the genuine
    pre is the Unknown-TriTime reference. Without the session-level
    numbered-post guard this series is demoted and the session ends up with
    TWO pre-labeled scans, violating the single-pre invariant."""
    n = 5
    df = pd.DataFrame({
        'PATH': [f'/x/{i}.dcm' for i in range(n)],
        'Orientation': ['A'] * n,
        'ID': ['TST002'] * n,
        'DATE': ['20240101'] * n,
        'SessionID': ['TST002_20240101'] * n,
        'Series': [1, 7, 801, 802, 803],
        'Series_desc': ['Axial T1', 'Axial T1 FS pre', 'Axial T1 pre',
                        'Axial T1 post 1', 'Axial T1 post 2'],
        'Modality': ['T1'] * n,
        'TriTime': ['Unknown', 'Unknown', '0', '132446', '264894'],
        'Lat': ['Unknown'] * n,
        'NumSlices': [NUMSLICES] * n,
        'Type': [['ORIGINAL', 'PRIMARY', 'OTHER']] * n,
        'DWI': [None] * n,
    })
    filt, ok = _run(df)
    assert ok, f'isolate_sequence failed; removed={list(filt.removed.keys())}'
    out = filt.dicom_table
    pre = out[out['Pre_scan'].astype(str).isin(['True', '1'])]
    assert len(pre) == 1, \
        f'single-pre invariant: exactly one pre-labeled scan, got {len(pre)}'
    assert pre.iloc[0]['Series_desc'] == 'Axial T1 FS pre', \
        'the Unknown-TriTime reference must be the labeled pre'
    posts = out[out['Post_scan'].astype(str).isin(['True', '1'])]
    assert len(posts) == 3, f'bolus + 2 numbered posts must stay post; got {len(posts)}'
    assert (posts['Series_desc'].values == 'Axial T1 pre').any(), \
        'the TriTime=0 bolus (phase-0 of the numbered dynamic) must stay post'


def test_no_tri0_zero_session_unchanged():
    """No TriTime=0 series at all: the baseline (numeric trigger time) is a
    regular post phase under the legacy trigger-time rule — 4 posts, 1 pre.
    This pins the no-op behavior of the fix when no zero-stamped series exists."""
    df = _mk('FAT: Axial T1 PRE FS')
    df.loc[df['TriTime'] == '0', 'TriTime'] = '65000'  # baseline not stamped 0
    filt, ok = _run(df)
    assert ok
    out = filt.dicom_table
    pre = out[out['Pre_scan'].astype(str).isin(['True', '1'])]
    assert len(pre) == 1
    posts = out[out['Post_scan'].astype(str).isin(['True', '1'])]
    assert len(posts) == 4
    # NB: in the no-tri0 case the pure-pre baseline carries a numeric trigger
    # time, so the legacy rule keeps it as a post phase — that pre-existing
    # behavior is intentionally NOT changed by this fix (out of scope).
