"""
Integration tests for 01_scanDicom.py -- end-to-end workflow verification.

These tests invoke the **actual pipeline functions**
``find_all_dicom_dirs()`` --> ``findDicom()`` --> ``extractDicom()`` in
sequence and assert that the combined output produces a valid, non-empty
``DataFrame`` with the expected schema.

Because they construct realistic on-disk DICOM files and exercise the full
chain of file I/O, module loading, and DataFrame construction, these tests
are tagged with ``@pytest.mark.integration`` so they can be selectively
skipped in CI via ``pytest -m "not integration"`` if needed.

Running
---
::

    # run only integration tests
    pytest test/test_scanDicom_integration.py -v --integration

    # skip integration tests elsewhere
    pytest test/test_scanDicom_unit.py test_scanDocom_full.py -m "not integration"


Test matrix
------
+---------+----+---+----------+
| Test                                       | What it verifies                            |
+---------+----+---+----------+
| ``test_end_to_end_small``                     | Full pipeline: one MR DICOM --> directory  |
|                                               | discovery --> series selection --> metadata |
|                                               | extraction --> non-empty DataFrame with     |
|                                               | ``Modality`` column                         |
+---------+----+---+----------+
"""

import pytest
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


@pytest.mark.integration
def test_end_to_end_small(tmp_path, monkeypatch):
    """Verify the full pipeline chain produces a valid, non-empty DataFrame.

    Test pipeline::

        1. create single MR DICOM file on disk
        2. ``find_all_dicom_dirs()`` -- discover directory
        3. ``findDicom()`` -- select representative series file
        4. ``extractDicom()`` -- extract 22 metadata fields per file
        5. ``pd.DataFrame(info)`` -- verify schema and non-empty

    Directory structure::

        tmp/data/
        └── subj1/
            └── s1.dcm   (MR, series 1)
    """
    # 1. build a small on-disk dataset
    root = tmp_path / "data"
    a = root / "subj1"
    a.mkdir(parents=True)
    make_minimal_dcm(str(a / "s1.dcm"), modality='MR', series_number=1)

    # 2. discover MRI directories
    dicom_dirs = scan.find_all_dicom_dirs(str(root))
    assert dicom_dirs, "find_all_dicom_dirs() should find exactly one MR directory"

    # 3. select representative series file
    files = scan.findDicom(dicom_dirs[0])
    assert files, "findDicom() should return at least one .dcm file"

    # 4. extract metadata
    info = [scan.extractDicom(fp) for fp in files]
    info = [i for i in info if i is not None]  # filter out any extraction failures

    # 5. assert output DataFrame is valid
    import pandas as pd
    df = pd.DataFrame(info)
    assert not df.empty, "extractDicom output should produce a non-empty DataFrame"
    assert 'Modality' in df.columns, "DataFrame should contain a 'Modality' column"