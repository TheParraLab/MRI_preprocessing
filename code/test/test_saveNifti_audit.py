"""
Unit tests for code/preprocessing/03_saveNifti.py -- audit_nifti_directory().

Verifies the post-conversion NIfTI directory audit logic in isolation, using
temp fixtures (tmp_path) and monkeypatched module globals so the real
/FL_system paths are never touched.

Running
-------
::

    pytest test/test_saveNifti_audit.py -v


Test matrix
-----------
+--------------------------------+------------------------------------------+
| Test                           | Validates                                |
+--------------------------------+------------------------------------------+
| ``test_audit_clean``           | Table rows match .nii files on disk ->   |
|                                | True, json clean=true                    |
| ``test_audit_missing``         | Fewer .nii files than table rows ->      |
|                                | False, missing_files counted             |
| ``test_audit_extra``           | Unrequested .nii on disk -> False,       |
|                                | extra_files counted                      |
| ``test_audit_duplicate_major`` | Two rows same Major -> False,            |
|                                | duplicate_major_rows counted             |
| ``test_audit_ghost_session``   | Table rows but no directory on disk ->   |
|                                | False, ghost_sessions counted            |
| ``test_audit_ghost_plus_clean``| Ghost + one clean session -> False,      |
|                                | clean session not flagged                |
+--------------------------------+------------------------------------------+
"""

import importlib.util
import json
import os
import sys

from pathlib import Path

# ---- Module loading setup ----
# Load 03_saveNifti.py by file path with sys.argv neutralized so that the
# module-level ``parser.parse_args()`` does not choke on pytest's argv.
proj_root = Path(__file__).resolve().parents[2]
save_nifti_path = proj_root / "code" / "preprocessing" / "03_saveNifti.py"

sys.path.insert(0, str(proj_root / "code" / "preprocessing"))

import tempfile as _tempfile
_log_dir_for_import = _tempfile.mkdtemp(prefix="savenifti_audit_test_")
os.environ['LOG_DIR'] = _log_dir_for_import

_spec = importlib.util.spec_from_file_location("save_nifti_module", str(save_nifti_path))
mod = importlib.util.module_from_spec(_spec)
_orig_argv = sys.argv
sys.argv = ['03_saveNifti.py']
try:
    _spec.loader.exec_module(mod)
finally:
    sys.argv = _orig_argv


# ---- Helpers ------------------------------------------------------------

def _setup(tmp_path, monkeypatch, table_rows, disk_files):
    """Point the module at temp dirs and build fixtures.

    table_rows: list of (SessionID, Major) pairs -> Data_table_timing.csv
    disk_files: list of (SessionID, filename) pairs -> nifti/<sid>/<filename>
    """
    import toolbox as _tb
    load_dir = tmp_path / "data"
    save_dir = tmp_path / "nifti"
    load_dir.mkdir()
    save_dir.mkdir()

    monkeypatch.setattr(mod, 'LOAD_DIR', str(load_dir) + '/')
    monkeypatch.setattr(mod, 'SAVE_DIR', str(save_dir) + '/')
    monkeypatch.setattr(_tb, 'get_log_dir', lambda: str(tmp_path / "logs"))

    csv_path = load_dir / "Data_table_timing.csv"
    with open(csv_path, 'w') as fh:
        fh.write('SessionID,Major\n')
        for sid, major in table_rows:
            fh.write(f'{sid},{major}\n')

    for sid, fname in disk_files:
        sdir = save_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / fname).write_bytes(b'fake-nifti')
    return load_dir, save_dir


def _read_json(tmp_path):
    return json.loads((tmp_path / "logs" / "nifti_audit.json").read_text())


# ---- Tests ------------------------------------------------------------

def test_audit_clean(tmp_path, monkeypatch):
    """Table rows exactly match .nii files on disk -> True, clean=true."""
    _setup(tmp_path, monkeypatch,
           table_rows=[('S1', 0), ('S1', 1), ('S2', 0)],
           disk_files=[('S1', '00.nii'), ('S1', '01.nii'), ('S2', '00.nii')])
    assert mod.audit_nifti_directory() is True
    j = _read_json(tmp_path)
    assert j['clean'] is True
    assert j['missing_files'] == 0
    assert j['extra_files'] == 0
    assert j['duplicate_major_rows'] == 0
    assert j['ghost_sessions'] == 0


def test_audit_missing(tmp_path, monkeypatch):
    """Table expects 00 and 01, disk has only 00.nii -> False, missing counted."""
    _setup(tmp_path, monkeypatch,
           table_rows=[('S1', 0), ('S1', 1)],
           disk_files=[('S1', '00.nii')])
    assert mod.audit_nifti_directory() is False
    j = _read_json(tmp_path)
    assert j['clean'] is False
    assert j['missing_files'] == 1


def test_audit_extra(tmp_path, monkeypatch):
    """Extra 99.nii on disk with no table row -> False, extra counted."""
    _setup(tmp_path, monkeypatch,
           table_rows=[('S1', 0)],
           disk_files=[('S1', '00.nii'), ('S1', '99.nii')])
    assert mod.audit_nifti_directory() is False
    j = _read_json(tmp_path)
    assert j['clean'] is False
    assert j['extra_files'] == 1
    assert j['missing_files'] == 0


def test_audit_duplicate_major(tmp_path, monkeypatch):
    """Two table rows with the same Major -> False, dup counted."""
    _setup(tmp_path, monkeypatch,
           table_rows=[('S1', 0), ('S1', 0)],
           disk_files=[('S1', '00.nii')])
    assert mod.audit_nifti_directory() is False
    j = _read_json(tmp_path)
    assert j['clean'] is False
    assert j['duplicate_major_rows'] == 1


def test_audit_ghost_session(tmp_path, monkeypatch):
    """Table has a session with rows but no directory on disk (the new fix)."""
    _setup(tmp_path, monkeypatch,
           table_rows=[('GHOST', 0)],
           disk_files=[])
    assert mod.audit_nifti_directory() is False
    j = _read_json(tmp_path)
    assert j['clean'] is False
    assert j['ghost_sessions'] >= 1
    assert j['missing_files'] >= 1


def test_audit_ghost_plus_clean(tmp_path, monkeypatch):
    """One ghost session and one clean session -> False, clean one not flagged."""
    _setup(tmp_path, monkeypatch,
           table_rows=[('GHOST', 0), ('GHOST', 1), ('OK', 0)],
           disk_files=[('OK', '00.nii')])
    assert mod.audit_nifti_directory() is False
    j = _read_json(tmp_path)
    assert j['clean'] is False
    assert j['ghost_sessions'] == 1
    # only the ghost's 2 expected files may be counted missing
    assert j['missing_files'] == 2
