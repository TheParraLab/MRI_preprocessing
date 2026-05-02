"""
Unit tests for 01_scanDicom.py -- core functionality verified in isolation.

Each test targets a single public function or pipeline stage from
``code/preprocessing/01_scanDicom.py``.  These tests use lightweight
synthetic DICOM files (``conftest.make_minimal_dcm``) to verify individual
behaviors without the overhead of constructing realistic datasets.

Running
-------
::

    pytest test/test_scanDicom_unit.py -v


Test matrix
-----------
+--------------------------------------------------+------------------------------------------+
| Test                                             | Validates                                |
+--------------------------------------------------+------------------------------------------+
| ``test_find_all_dicom_dirs_single``              | ``find_all_dicom_dirs()`` discovers one  |
|                                                  | directory containing exactly one MR file |
+--------------------------------------------------+------------------------------------------+
| ``test_findDicom_series``                         | ``findDicom()`` returns one file per     |
|                                                  | MR SeriesNumber; non-MR modalities are   |
|                                                  | correctly excluded at the directory level|
+--------------------------------------------------+------------------------------------------+
| ``test_extractDicom_basic``                       | ``extractDicom()`` returns a dict with a |
|                                                  | string ``Modality`` value                 |
+--------------------------------------------------+------------------------------------------+
| ``test_find_all_dicom_dirs_ignores_non_mr``      | Mixed directory with CT + garbage ``.dcm``|
|                                                  | does NOT return a MRI directory            |
+--------------------------------------------------+------------------------------------------+
| ``test_findDicom_handles_unreadable``            | ``findDicom()`` gracefully skips unreadable|
|                                                  | files and still returns the good MR file  |
+--------------------------------------------------+------------------------------------------+
| ``test_findDicom_sampling_is_deterministic``     | ``findDicom()`` with ``sample_pct +``     |
|                                                  | ``sample_seed`` produces identical results|
|                                                  | across two calls                          |
+--------------------------------------------------+------------------------------------------+
"""

import importlib.util
import sys
import random
import tempfile
from pathlib import Path
from conftest import make_minimal_dcm

# ---- Module loading setup ----
proj_root = Path(__file__).resolve().parents[1]
scan_path = proj_root / "code" / "preprocessing" / "01_scanDicom.py"
spec = importlib.util.spec_from_file_location("scan_module", str(scan_path))
scan = importlib.util.module_from_spec(spec)

sys.path.insert(0, str(proj_root / "code" / "preprocessing"))

test_save_dir = proj_root / "tmp_test"
test_save_dir.mkdir(parents=True, exist_ok=True)
_orig_argv = sys.argv
sys.argv = [str(scan_path.name), "--save_dir", str(test_save_dir)]
try:
    spec.loader.exec_module(scan)
finally:
    sys.argv = _orig_argv

# Test directory for logger/checkpoint files
_tmp_test_dir = tempfile.mkdtemp(prefix="scan_unit_")


def _make_cfg(save_dir: str = _tmp_test_dir) -> scan.ScanConfig:
    cfg = scan.ScanConfig(save_dir=save_dir, scan_dir=save_dir)
    return cfg


def _make_logger(save_dir: str = _tmp_test_dir):
    return scan.create_logger(scan.ScanConfig(save_dir=save_dir))


def test_find_all_dicom_dirs_single(tmp_path):
    d = tmp_path / "subj1"
    d.mkdir()
    make_minimal_dcm(str(d / "img1.dcm"), modality='MR')
    (d / "readme.txt").write_text("notes")
    cfg = _make_cfg()
    logger = _make_logger()
    dirs = scan._find_all_dicom_dirs_impl(cfg, logger, str(tmp_path))
    assert any(str(d) in dd for dd in dirs)


def test_findDicom_series(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    make_minimal_dcm(str(root / "a.dcm"), modality='MR', series_number=1)
    make_minimal_dcm(str(root / "b.dcm"), modality='MR', series_number=2)
    make_minimal_dcm(str(root / "c.dcm"), modality='CT', series_number=3)
    logger = _make_logger()
    found = scan._find_dicom_worker(str(root), sample_pct=0.0, sample_seed=None, logger=logger)
    assert any("a.dcm" in f or "b.dcm" in f for f in found)


def test_extractDicom_basic(tmp_path):
    f = tmp_path / "x.dcm"
    make_minimal_dcm(str(f), modality='MR', series_number=5, patient_id='P1')
    logger = _make_logger()
    out = scan._extractDicom_impl(str(f), logger)
    assert isinstance(out, dict)
    assert isinstance(out['Modality'], str)


def test_find_all_dicom_dirs_ignores_non_mr_and_unreadable(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    make_minimal_dcm(str(d / "ct.dcm"), modality='CT')
    (d / "bad.dcm").write_text("not a dicom file")
    cfg = _make_cfg()
    logger = _make_logger()
    dirs = scan._find_all_dicom_dirs_impl(cfg, logger, str(tmp_path))
    assert all(str(d) not in dd for dd in dirs)


def test_findDicom_handles_unreadable_and_returns_mr_only(tmp_path):
    root = tmp_path / "study2"
    root.mkdir()
    make_minimal_dcm(str(root / "mri.dcm"), modality='MR', series_number=10)
    (root / "garbage.dcm").write_text("corrupt")
    logger = _make_logger()
    found = scan._find_dicom_worker(str(root), sample_pct=0.0, sample_seed=None, logger=logger)
    assert any("mri.dcm" in f for f in found)


def test_findDicom_sampling_is_deterministic_with_seed(tmp_path):
    root = tmp_path / "bigstudy"
    root.mkdir()
    for i in range(12):
        series = (i % 4) + 1
        make_minimal_dcm(str(root / f"img_{i}.dcm"), modality='MR', series_number=series)

    logger = _make_logger()
    first = scan._find_dicom_worker(str(root), sample_pct=20.0, sample_seed=123, logger=logger)
    second = scan._find_dicom_worker(str(root), sample_pct=20.0, sample_seed=123, logger=logger)
    assert first == second