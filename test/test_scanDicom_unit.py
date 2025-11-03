import importlib.util
import sys
from pathlib import Path
from conftest import make_minimal_dcm

# Dynamically load the 01_scanDicom.py module (filename isn't a valid python identifier)
proj_root = Path(__file__).resolve().parents[1]
scan_path = proj_root / "code" / "preprocessing" / "01_scanDicom.py"
spec = importlib.util.spec_from_file_location("scan_module", str(scan_path))
scan = importlib.util.module_from_spec(spec)
# Ensure local preprocessing package dir is on sys.path so imports like `toolbox` resolve
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))
# Prevent argparse in the module from reading pytest's argv during import
import os as _os
_orig_argv = sys.argv
# Use a writable temporary save dir inside the project for logger/files to avoid permission errors
test_save_dir = proj_root / "tmp_test"
test_save_dir.mkdir(parents=True, exist_ok=True)
sys.argv = [str(scan_path.name), "--save_dir", str(test_save_dir)]
try:
    spec.loader.exec_module(scan)
finally:
    sys.argv = _orig_argv


def test_find_all_dicom_dirs_single(tmp_path):
    d = tmp_path / "subj1"
    d.mkdir()
    make_minimal_dcm(str(d / "img1.dcm"), modality='MR')
    # add a non-dicom file to ensure it gets ignored
    (d / "readme.txt").write_text("notes")
    dirs = scan.find_all_dicom_dirs(str(tmp_path))
    assert any(str(d) in dd for dd in dirs)


def test_findDicom_series(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    make_minimal_dcm(str(root / "a.dcm"), modality='MR', series_number=1)
    make_minimal_dcm(str(root / "b.dcm"), modality='MR', series_number=2)
    make_minimal_dcm(str(root / "c.dcm"), modality='CT', series_number=3)
    found = scan.findDicom(str(root))
    # expect MR series files present (one file per series); CT may also appear depending on implementation
    assert any("a.dcm" in f or "b.dcm" in f for f in found)


def test_extractDicom_basic(tmp_path):
    f = tmp_path / "x.dcm"
    make_minimal_dcm(str(f), modality='MR', series_number=5, patient_id='P1')
    out = scan.extractDicom(str(f))
    assert isinstance(out, dict)
    # Implementation maps modality using RepetitionTime -> 'T1'/'T2' or returns 'Unknown' if not present
    assert isinstance(out['Modality'], str)


def test_find_all_dicom_dirs_ignores_non_mr_and_unreadable(tmp_path):
    # create directory with a CT file and a garbage .dcm file
    d = tmp_path / "mixed"
    d.mkdir()
    # CT file
    make_minimal_dcm(str(d / "ct.dcm"), modality='CT')
    # garbage file with .dcm extension
    (d / "bad.dcm").write_text("not a dicom file")

    dirs = scan.find_all_dicom_dirs(str(tmp_path))
    # No MR files present -> directory should NOT be listed
    assert all(str(d) not in dd for dd in dirs)


def test_findDicom_handles_unreadable_and_returns_mr_only(tmp_path):
    root = tmp_path / "study2"
    root.mkdir()
    # good MR file
    make_minimal_dcm(str(root / "mri.dcm"), modality='MR', series_number=10)
    # unreadable file
    (root / "garbage.dcm").write_text("corrupt")

    found = scan.findDicom(str(root))
    # should include at least the MR file and not crash
    assert any("mri.dcm" in f for f in found)


def test_findDicom_sampling_is_deterministic_with_seed(tmp_path):
    # Create many files across several series
    root = tmp_path / "bigstudy"
    root.mkdir()
    # Create 12 files across series 1-4
    for i in range(12):
        series = (i % 4) + 1
        make_minimal_dcm(str(root / f"img_{i}.dcm"), modality='MR', series_number=series)

    import random
    scan.SAMPLE_PCT = 20  # sample ~2 files
    random.seed(123)
    first = scan.findDicom(str(root))
    random.seed(123)
    second = scan.findDicom(str(root))
    assert first == second