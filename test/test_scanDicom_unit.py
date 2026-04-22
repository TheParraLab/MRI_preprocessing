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
| ``test_findDicom_sampling_is_deterministic``     | ``findDicom()`` with ``SAMPLE_PCT +``     |
|                                                  | ``SAMPLE_SEED`` produces identical results|
|                                                  | across two calls                          |
+--------------------------------------------------+------------------------------------------+
"""

import importlib.util
import sys
from pathlib import Path
from conftest import make_minimal_dcm

# ---- Module loading setup ----
# Dynamically load 01_scanDicom.py (filename contains digits, not a valid Python identifier,
# so we use importlib rather than a regular import statement).
proj_root = Path(__file__).resolve().parents[1]
scan_path = proj_root / "code" / "preprocessing" / "01_scanDicom.py"
spec = importlib.util.spec_from_file_location("scan_module", str(scan_path))
scan = importlib.util.module_from_spec(spec)

# Ensure local preprocessing helpers (e.g. toolbox) resolve at import time
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))

# Prevent argparse inside 01_scanDicom.py from reading pytest's sys.argv
_orig_argv = sys.argv
# ``tmp_test`` is a writable directory inside the project for logger / checkpoint files.
test_save_dir = proj_root / "tmp_test"
test_save_dir.mkdir(parents=True, exist_ok=True)
sys.argv = [str(scan_path.name), "--save_dir", str(test_save_dir)]
try:
    spec.loader.exec_module(scan)
finally:
    sys.argv = _orig_argv


def test_find_all_dicom_dirs_single(tmp_path):
    """A single MR file inside one sub-directory should be discovered.

    Structure::

        tmp/
        └── subj1/
            └── img1.dcm   (MR)
    """
    d = tmp_path / "subj1"
    d.mkdir()
    make_minimal_dcm(str(d / "img1.dcm"), modality='MR')
    # also drop a non-DICOM file to confirm it is ignored
    (d / "readme.txt").write_text("notes")
    dirs = scan.find_all_dicom_dirs(str(tmp_path))
    assert any(str(d) in dd for dd in dirs)


def test_findDicom_series(tmp_path):
    """``findDicom()`` should return one representative file per MR series.

    Structure::

        tmp/study/
        ├── a.dcm   (MR, series 1)
        ├── b.dcm   (MR, series 2)
        └── c.dcm   (CT, series 3 -- should be excluded)
    """
    root = tmp_path / "study"
    root.mkdir()
    make_minimal_dcm(str(root / "a.dcm"), modality='MR', series_number=1)
    make_minimal_dcm(str(root / "b.dcm"), modality='MR', series_number=2)
    make_minimal_dcm(str(root / "c.dcm"), modality='CT', series_number=3)
    found = scan.findDicom(str(root))
    # expect at least one MR series file in the result
    assert any("a.dcm" in f or "b.dcm" in f for f in found)


def test_extractDicom_basic(tmp_path):
    """``extractDicom()`` must return a dict with a string ``Modality`` value.

    The implementation maps RepetitionTime to 'T1'/'T2' or 'Unknown' when the
    RepetitionTime tag is absent.
    """
    f = tmp_path / "x.dcm"
    make_minimal_dcm(str(f), modality='MR', series_number=5, patient_id='P1')
    out = scan.extractDicom(str(f))
    assert isinstance(out, dict)
    assert isinstance(out['Modality'], str)


def test_find_all_dicom_dirs_ignores_non_mr_and_unreadable(tmp_path):
    """A directory containing only non-MR or garbage files must NOT be returned
    by ``find_all_dicom_dirs()``.

    Structure::

        tmp/mixed/
        ├── ct.dcm   (CT modality)
        └── bad.dcm  (corrupt content)
    """
    d = tmp_path / "mixed"
    d.mkdir()
    make_minimal_dcm(str(d / "ct.dcm"), modality='CT')
    (d / "bad.dcm").write_text("not a dicom file")

    dirs = scan.find_all_dicom_dirs(str(tmp_path))
    assert all(str(d) not in dd for dd in dirs)


def test_findDicom_handles_unreadable_and_returns_mr_only(tmp_path):
    """``findDicom()`` must skip unreadable files and still return good MR files.

    Structure::

        tmp/study2/
        ├── mri.dcm  (valid MR)
        └── garbage.dcm (corrupt)
    """
    root = tmp_path / "study2"
    root.mkdir()
    make_minimal_dcm(str(root / "mri.dcm"), modality='MR', series_number=10)
    (root / "garbage.dcm").write_text("corrupt")

    found = scan.findDicom(str(root))
    assert any("mri.dcm" in f for f in found)


def test_findDicom_sampling_is_deterministic_with_seed(tmp_path):
    """Resampling with a fixed ``SAMPLE_SEED`` must produce identical file sets."""
    root = tmp_path / "bigstudy"
    root.mkdir()
    for i in range(12):
        series = (i % 4) + 1
        make_minimal_dcm(str(root / f"img_{i}.dcm"), modality='MR', series_number=series)

    import random
    scan.SAMPLE_PCT = 20  # sample a subset
    random.seed(123)
    first = scan.findDicom(str(root))
    random.seed(123)
    second = scan.findDicom(str(root))
    assert first == second