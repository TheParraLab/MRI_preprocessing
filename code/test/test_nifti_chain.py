"""
Regression tests for the .nii/.nii.gz chain in the MRI preprocessing pipeline.
Verifies that toolbox helpers are extension-agnostic and that nibabel transparently
reads .nii.gz files (the format dcm2niix emits when invoked with -z y).
"""
import sys, os, tempfile
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "code" / "preprocessing"))
import toolbox as _tb

def test_nifti_stem_plain_and_gz():
    assert _tb.nifti_stem('01.nii') == '01'
    assert _tb.nifti_stem('01.nii.gz') == '01'
    assert _tb.nifti_stem('00a.nii') == '00a'
    assert _tb.nifti_stem('00a.nii.gz') == '00a'
    assert _tb.nifti_stem('01_RAS.nii') == '01'
    assert _tb.nifti_stem('01_RAS.nii.gz') == '01'

def test_is_nifti_file():
    assert _tb.is_nifti_file('01.nii')
    assert _tb.is_nifti_file('01.nii.gz')
    assert not _tb.is_nifti_file('01.txt')
    assert not _tb.is_nifti_file('Data_table.csv')

def test_glob_nifti_mixed(tmp_path):
    (tmp_path / '01.nii').write_bytes(b'')
    (tmp_path / '02.nii.gz').write_bytes(b'')
    (tmp_path / 'notes.txt').write_bytes(b'')
    found = _tb.glob_nifti(str(tmp_path), '*')
    names = sorted(_tb.nifti_stem(p) for p in found)
    assert names == ['01', '02']

def test_nibabel_roundtrip_gz(tmp_path):
    import nibabel as nib
    import numpy as np
    data = np.zeros((4, 5, 6), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    out = str(tmp_path / 'test.nii.gz')
    nib.save(img, out)
    img2 = nib.load(out)
    assert (img2.get_fdata() == data).all()
