"""
Integration tests for 01_scanDicom.py -- end-to-end workflow verification.

These tests invoke the actual pipeline functions using the new ScanConfig
API and assert that the combined output produces a valid, non-empty
DataFrame with the expected schema.

Running
---
::

    pytest test/test_scanDicom_integration.py -v
"""

import pytest
import importlib.util
import sys
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


@pytest.mark.integration
def test_end_to_end_small(tmp_path, monkeypatch):
    """Full pipeline: one MR DICOM through directory discovery, series
    selection, metadata extraction, and DataFrame construction."""
    root = tmp_path / "data"
    a = root / "subj1"
    a.mkdir(parents=True)
    make_minimal_dcm(str(a / "s1.dcm"), modality='MR', series_number=1)

    cfg = scan.ScanConfig(save_dir=str(tmp_path), scan_dir=str(root))
    logger = scan.create_logger(cfg)

    dicom_dirs = scan._find_all_dicom_dirs_impl(cfg, logger, str(root))
    assert dicom_dirs, "Should find exactly one MR directory"

    files = scan._find_dicom_worker(dicom_dirs[0], sample_pct=0.0, sample_seed=None, logger=logger)
    assert files, "Should return at least one .dcm file"

    info = [scan._extractDicom_impl(fp, logger) for fp in files]
    info = [i for i in info if i is not None]

    import pandas as pd
    df = pd.DataFrame(info)
    assert not df.empty, "Output should produce a non-empty DataFrame"
    assert 'Modality' in df.columns, "DataFrame should contain a 'Modality' column"