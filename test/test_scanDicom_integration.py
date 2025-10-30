import pytest
import importlib.util
import sys
from pathlib import Path
from conftest import make_minimal_dcm

# Dynamically load the 01_scanDicom.py module
proj_root = Path(__file__).resolve().parents[1]
scan_path = proj_root / "code" / "preprocessing" / "01_scanDicom.py"
spec = importlib.util.spec_from_file_location("scan_module", str(scan_path))
scan = importlib.util.module_from_spec(spec)
# Ensure local preprocessing package dir is on sys.path so imports like `toolbox` resolve
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))
# Prevent argparse in the module from reading pytest's argv during import
import os as _os
# Use a writable temporary save dir inside the project for logger/files to avoid permission errors
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
    # build a small dataset
    root = tmp_path / "data"
    a = root / "subj1"
    a.mkdir(parents=True)
    make_minimal_dcm(str(a / "s1.dcm"), modality='MR', series_number=1)

    # run the workflow pieces
    dicom_dirs = scan.find_all_dicom_dirs(str(root))
    assert dicom_dirs, "No dicom dirs found"

    files = scan.findDicom(dicom_dirs[0])
    assert files, "No series files found"

    info = [scan.extractDicom(fp) for fp in files]
    info = [i for i in info if i is not None]
    import pandas as pd
    df = pd.DataFrame(info)
    assert not df.empty
    # optional: assert expected columns exist
    assert 'Modality' in df.columns