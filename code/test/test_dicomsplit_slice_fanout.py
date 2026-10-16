"""Regression tests for DICOMsplit slice fan-out (ISSUE-001).

The 9/24 MSK deployment (20260924_112405) produced two sessions with 343 and
325 "post scans" because ``sort_scans()`` derived each file's ``Slice`` index
from the second-to-last dash token of the *path*. For dash-free RIA_20-425
filenames (``img_0031_1.2.840....dcm``) the only dash is inside the archive id
in the path, so every file parsed to the same value (20). The
``len(slices) != pre_slices`` count-guard passed (114 copies of 20), and the
representative-row filter ``Slice == slices[0]`` then matched ALL files in the
group — fanning each phase out to 114 duplicate rows.

Fix under test:
  1. ``_fallback_slices_if_degenerate`` — when parsed Slice values are not
     unique, re-derive them as a deterministic per-file rank (1..n by PATH
     order) within each TriTime group, so each phase yields exactly one
     representative row (healthy dash naming is untouched — no fallback).
  2. exactly-one guard on the representative-row insert — if the min Slice
     still matches >1 file, drop the session loud (``Split_Slice_Count_Mismatch``)
     instead of fanning out.
  3. ``_fill_split_row_metadata`` — populate Series_desc/Modality/ID/NumSlices
     on the split rows (they came from the 9-column scan_all frame as NaN).

Run with: pytest code/test/test_dicomsplit_slice_fanout.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))

from DICOM import DICOMsplit  # noqa: E402

PRE_SLICES = 6
NUM_POST = 3
TRIGGERS = ['000000', '020000', '040000']


def _write_dcm(path, trigger, desc):
    """Minimal MR DICOM file carrying the tags DICOMsplit/DICOMextract read."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = 'TST001'
    ds.Modality = 'MR'
    ds.SeriesNumber = 9
    ds.StudyDate = '20240101'
    ds.BodyPartExamined = 'BREAST'
    ds.SeriesDescription = desc
    ds.TriggerTime = trigger
    ds.RepetitionTime = 500.0  # Modality() derives T1 from RT
    ds.save_as(path, enforce_file_format=True)
    return str(path)


def _make_session(tmp_path, naming):
    """Pre dir (6 files) + post dir (18 files = 3 TriTimes x 6 slices) plus the
    01-style table DICOMsplit expects. ``naming`` is 'underscore' (dash-free
    filenames — the degenerate case) or 'dash' (MSKCC-style, slice index as the
    second-to-last dash token)."""
    pre_dir = tmp_path / "SCANS" / "4" / "DICOM"
    post_dir = tmp_path / "SCANS" / "6" / "DICOM"
    pre_dir.mkdir(parents=True)
    post_dir.mkdir(parents=True)

    def fname(n):
        if naming == 'dash':
            return f"1.2.840.113745.101000.2197000.1-800-{n:03d}-abcd.dcm"
        return f"img_{n:04d}_1.2.840.113619.2.176.6945.1.1.1.dcm"

    for k in range(PRE_SLICES):
        _write_dcm(pre_dir / fname(k + 1), '000000', 'FAT: AX, 3D T1 FS, PRE')
    for ph, trig in enumerate(TRIGGERS):
        for k in range(PRE_SLICES):
            _write_dcm(post_dir / fname(ph * PRE_SLICES + k + 1), trig,
                       f'FAT: Ph{ph + 1}/AX, 3D T1 FS, POST')

    table = pd.DataFrame({
        'SessionID': ['S_1'] * 2,
        'PATH': [str(pre_dir / fname(1)), str(post_dir / fname(1))],
        'Pre_scan': [1, 0],
        'Post_scan': [0, 1],
        'NumSlices': [PRE_SLICES, PRE_SLICES * NUM_POST],
        'Series_desc': ['FAT: AX, 3D T1 FS, PRE', 'FAT: Ph1/AX, 3D T1 FS, POST'],
        'Series': [4, 6],
        'Modality': ['T1', 'T1'],
        'ID': ['TST001'] * 2,
    })
    return table


def _split(table, tmp_path):
    splitter = DICOMsplit(table, tmp_save=str(tmp_path / "tmp"))
    assert splitter.SCAN, 'init should flag a split-required session'
    splitter.scan_all()
    splitter.sort_scans()
    return splitter


def test_underscore_naming_no_fanout(tmp_path):
    """ISSUE-001 core case: dash-free filenames must NOT fan out."""
    table = _make_session(tmp_path, 'underscore')
    splitter = _split(table, tmp_path)

    assert not splitter.dicom_table.empty, 'session should survive (rank fallback)'
    post = splitter.dicom_table[splitter.dicom_table['Pre_scan'] != 1]
    # 3 phases, NOT 18 (the old fan-out emitted one row per file per phase)
    assert len(post) == NUM_POST, f'expected {NUM_POST} post rows, got {len(post)}'
    # The fallback made Slice unique WITHIN each TriTime group (1..6 per group)
    assert (splitter.scan_results
            .groupby('TriTime')['Slice']
            .apply(lambda s: s.nunique() == len(s)).all())
    # Distinct tmp PATH per phase — no 114-identical-rows
    assert post['PATH'].nunique() == NUM_POST
    # Metadata parity on the split rows
    assert post['Series_desc'].notna().all(), 'split rows must carry Series_desc'
    assert (post['Modality'] == 'T1').all()
    assert (post['NumSlices'] == PRE_SLICES).all()
    # Relocations: exactly one file per slice per phase
    assert len(splitter.temporary_relocations) == PRE_SLICES * NUM_POST


def test_dash_naming_unchanged(tmp_path):
    """Regression guard: dash-suffixed filenames (healthy convention) must NOT
    trigger the fallback and must produce the legacy one-row-per-phase result."""
    table = _make_session(tmp_path, 'dash')
    splitter = _split(table, tmp_path)
    assert not splitter.dicom_table.empty
    post = splitter.dicom_table[splitter.dicom_table['Pre_scan'] != 1]
    assert len(post) == NUM_POST
    # No fallback: parsed Slice values keep their full 1..18 spread (max==18).
    # The fallback would have collapsed them to within-group ranks (max==6).
    assert splitter.scan_results['Slice'].max() == PRE_SLICES * NUM_POST
    assert splitter.scan_results['Slice'].nunique() == PRE_SLICES * NUM_POST
    assert len(splitter.temporary_relocations) == PRE_SLICES * NUM_POST


def test_exactly_one_guard_fails_loud(tmp_path, monkeypatch):
    """Backstop: if Slice is non-unique even after the fallback (a regression in
    the fallback itself), the exactly-one guard must drop the session instead of
    appending one row per file. Isolated by no-op'ing the fallback."""
    table = _make_session(tmp_path, 'underscore')
    splitter = DICOMsplit(table, tmp_save=str(tmp_path / "tmp"))
    splitter.scan_all()
    # Force a broken fallback (constant Slice) and disable the real one.
    monkeypatch.setattr(DICOMsplit, '_fallback_slices_if_degenerate', lambda self: None)
    splitter.scan_results['Slice'] = 1  # every file claims slice 1
    splitter.sort_scans()
    # The guard must drop the session (empty table) — never fan out to 18 rows.
    assert splitter.dicom_table.empty, 'degenerate Slice must drop the session'
    assert 'Split_Slice_Count_Mismatch' in splitter.removed
