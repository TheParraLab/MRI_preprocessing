"""
End-to-end tests for the *parallel* ``run_cmd`` path of code/preprocessing/03_saveNifti.py.

Unlike test_saveNifti_audit.py (which tests the pure audit function), these tests
drive ``run_with_progress(partial(run_cmd, commands=...), ...)`` through a real
``ProcessPoolExecutor`` (fork start method), with a *fake* ``dcm2niix`` executable
placed on ``PATH``.  The fake inspects the input DICOM basename (``good_``/``slow_``/
``sleep_``/``fail_`` prefix) to decide its behaviour, so each branch of ``run_cmd``
(ok / skipped / stopped / timeout) can be exercised deterministically without a real
dcm2niix binary or real DICOM data.

Running
-------
::

    cd /mnt/projects/MRI_preprocessing
    python -m pytest code/test/test_saveNifti_multiprocessing.py -v

Test matrix
-----------
+------------------------------------------+--------------------------------------+
| Test                                     | Validates                            |
+------------------------------------------+--------------------------------------+
| test_parallel_ok_logging_and_completion  | 3 good items -> 3 ok, valid nii on   |
|                                          | disk, worker [START]/[DONE] lines    |
|                                          | reach the PARENT log file (mp queue  |
|                                          | wiring), _remove_completed /         |
|                                          | _finalize_progress clean up          |
| test_resume_skips_existing               | pre-existing valid nii -> 'skipped', |
|                                          | dcm2niix not re-invoked for that     |
|                                          | item, invoked for the new one        |
| test_disk_threshold_stops_all            | impossible disk threshold -> all     |
|                                          | 'stopped', zero dcm2niix calls,      |
|                                          | _remove_completed=0, checkpoint=3    |
| test_in_flight_completes_queued_stop     | in-flight worker finishes + valid    |
|                                          | nii while queued worker stops;       |
|                                          | checkpoint retains only the stopped  |
|                                          | command                             |
| test_timeout_kills_dcm2niix              | sleep item + small timeout ->        |
|                                          | 'failed'(timeout), [TIMEOUT] logged, |
|                                          | no orphan process, partial cleaned   |
+------------------------------------------+--------------------------------------+
"""

import importlib.util
import os
import pickle
import stat
import sys
import time
import threading
from functools import partial as _partial
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

# ---- Fake-dcm2niix behaviour log (shared with the fake script) -------------
MRI_FAKE_DCM_LOG = 'MRI_FAKE_DCM_LOG'

# ---- Fake dcm2niix: written to tmp_path/'bin'/'dcm2niix' -------------------
# Shebang #!/usr/bin/env python3 -> same nibabel/numpy as the parent (verified).
FAKE_DCM2NIIX = r'''#!/usr/bin/env python3
import sys, os, time

def _log_call(path):
    log = os.environ.get('MRI_FAKE_DCM_LOG')
    if log:
        try:
            with open(log, 'a') as fh:
                fh.write(path + '\n')
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass

def _write_nii(out_dir, name):
    import numpy as np
    import nibabel as nib
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    dst = os.path.join(out_dir, name + '.nii.gz')
    img = nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4))
    nib.save(img, dst)

def main():
    argv = list(sys.argv[1:])
    # last element is the input DICOM path (dcm2niix convention)
    inp = argv[-1] if argv else ''
    base = os.path.basename(inp)
    out_dir, name = '.', '00'
    i = 0
    while i < len(argv) - 1:
        tok = argv[i]
        if tok == '-o':
            out_dir = argv[i + 1]; i += 2
        elif tok == '-f':
            name = argv[i + 1]; i += 2
        else:
            i += 1
    # record the real dcm2niix invocation BEFORE any behaviour
    _log_call(inp)
    if base.startswith('slow_'):
        time.sleep(2)
        _write_nii(out_dir, name); sys.exit(0)
    if base.startswith('sleep_'):
        time.sleep(120); sys.exit(0)
    if base.startswith('fail_'):
        sys.stderr.write('boom\n'); sys.stderr.flush(); sys.exit(3)
    if base.startswith('good_'):
        _write_nii(out_dir, name); sys.exit(0)
    # default: pretend success
    _write_nii(out_dir, name); sys.exit(0)

if __name__ == '__main__':
    main()
'''

# ---- Module loading (mirrors test_saveNifti_audit.py) ----------------------
proj_root = Path(__file__).resolve().parents[2]
save_nifti_path = proj_root / "code" / "preprocessing" / "03_saveNifti.py"
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))

import tempfile as _tempfile
_log_dir_for_import = _tempfile.mkdtemp(prefix="savenifti_mp_test_")
os.environ['LOG_DIR'] = _log_dir_for_import

_spec = importlib.util.spec_from_file_location("save_nifti_module", str(save_nifti_path))
mod = importlib.util.module_from_spec(_spec)
_orig_argv = sys.argv
sys.argv = ['03_saveNifti.py']
try:
    _spec.loader.exec_module(mod)
finally:
    sys.argv = _orig_argv

# Register so ProcessPoolExecutor children (fork) can pickle mod.run_cmd back to
# the parent for result marshalling.
sys.modules['save_nifti_module'] = mod

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _write_fake_dcm2niix(tmp_path):
    """Create the fake dcm2niix executable at tmp_path/'bin'/'dcm2niix'."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / 'dcm2niix'
    exe.write_text(FAKE_DCM2NIIX)
    st = exe.stat()
    os.chmod(str(exe), (st.st_mode & ~0o777) | 0o755)
    return str(exe)


def _redirect_logger_file(tmp_path, monkeypatch):
    """Point the live ``03_saveNifti`` logger's file handler at this test's tmp
    dir and align ``_file_path`` accordingly.

    The ``03_saveNifti`` logger is a *singleton*: whichever test module imported
    it first owns the ``QueueListener``/``FileHandler`` (an idempotency guard in
    ``toolbox.get_logger`` means a later import only re-points ``_file_path`` and
    leaves the existing FileHandler untouched).  That makes the file that actually
    receives the worker / drain-thread records depend on import order, which is
    why reading ``LOGGER._file_path`` can miss the real file in a combined run.

    Swapping in a fresh per-test FileHandler (and closing it on teardown) makes
    log assertions import-order independent and isolated per test.  Returns the
    new FileHandler so the caller can close it.
    """
    import logging
    logger = getattr(mod.LOGGER, '_wrapped', mod.LOGGER)
    name = getattr(logger, 'name', '03_saveNifti')
    log_dir = tmp_path / 'logs'
    log_dir.mkdir(exist_ok=True)
    target = str(log_dir / (name + '.log'))

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    new_fh = logging.FileHandler(target, mode='a', encoding='utf-8')
    new_fh.setLevel(logging.DEBUG)
    new_fh.setFormatter(fmt)

    import toolbox as _tb
    registry = getattr(_tb, '_listener_registry', {})
    listener = registry.get(name)
    if listener is not None and hasattr(listener, 'handlers'):
        keep = [h for h in list(getattr(listener, 'handlers', []))
                if not hasattr(h, 'baseFilename')]  # drop the old FileHandler
        monkeypatch.setattr(listener, 'handlers', [new_fh] + keep)

    monkeypatch.setattr(logger, '_file_path', os.path.abspath(target))
    return new_fh


def _prepare_env(tmp_path, monkeypatch):
    """Point the module at temp dirs, install the fake dcm2niix, force the fork
    start method, and retarget the logger's file handler.  Returns the new
    FileHandler (for teardown) plus (load_dir, save_dir)."""
    import toolbox as _tb
    load_dir = tmp_path / 'data'
    save_dir = tmp_path / 'nifti'
    load_dir.mkdir(exist_ok=True)
    save_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(mod, 'LOAD_DIR', str(load_dir) + '/')
    monkeypatch.setattr(mod, 'SAVE_DIR', str(save_dir) + '/')
    monkeypatch.setattr(_tb, 'get_log_dir', lambda: str(tmp_path / "logs"))

    fake = _write_fake_dcm2niix(tmp_path)
    monkeypatch.setenv('PATH', str(tmp_path / 'bin') + os.pathsep + os.environ['PATH'])
    monkeypatch.delenv('MRI_FAKE_DCM_LOG', raising=False)
    monkeypatch.setenv(MRI_FAKE_DCM_LOG, str(tmp_path / 'dcm2niix_calls.log'))

    import multiprocessing as _mp
    try:
        _mp.set_start_method('fork')
    except RuntimeError:
        pass

    new_fh = _redirect_logger_file(tmp_path, monkeypatch)
    return new_fh


def _write_nii(path):
    """Write a valid 8x8x8 NIfTI to `path` (parent dirs created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)), str(p))


def _make_item(idx, mode, save_dir, src_name='src'):
    """Create LOAD_DIR/<src_name><idx>/<mode>_<idx>.dcm + the matching dcm2niix
    command.  Returns (command, input_path)."""
    src = f'{mod.LOAD_DIR}{src_name}{idx}/'
    os.makedirs(src, exist_ok=True)
    inp = f'{src}{mode}_{idx}.dcm'
    with open(inp, 'w') as fh:
        fh.write('fake dicom\n')
    cmd = ['dcm2niix', '-z', 'y', '-o', f'{mod.SAVE_DIR}S{idx:03d}', '-f', '00', inp]
    return cmd, inp


def _load_log_file():
    """Read the parent logger's log file path at call time and return its text
    (empty string on any failure)."""
    path = getattr(mod.LOGGER, '_file_path', '')
    if not path or not os.path.exists(path):
        return ''
    try:
        with open(path, 'r', errors='replace') as fh:
            return fh.read()
    except Exception:
        return ''


def _poll_log(pred, timeout=15.0, interval=0.25):
    """Poll the (async, single-thread-flushed) log file until `pred(text)` is
    True or the deadline elapses.  Returns the matching text or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = _load_log_file()
        if pred(text):
            return text
        time.sleep(interval)
    text = _load_log_file()
    return text if pred(text) else None


def _calls(path):
    """Number of dcm2niix invocations logged (one line per call)."""
    if not os.path.exists(path):
        return 0
    with open(path, 'r', errors='replace') as fh:
        return sum(1 for ln in fh if ln.strip())


def _log_lines_containing(path, needle):
    """True if the dcm2niix-call log has a line containing `needle`."""
    if not os.path.exists(path):
        return False
    with open(path, 'r', errors='replace') as fh:
        return any(needle in ln for ln in fh)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def stop_flag_reset():
    mod.stop_flag.clear()
    yield
    mod.stop_flag.clear()


@pytest.fixture
def env(tmp_path, monkeypatch):
    new_fh = _prepare_env(tmp_path, monkeypatch)
    yield
    # Close our per-test FileHandler (monkeypatch restores listener.handlers /
    # _file_path to the audit-owning module's original values).
    try:
        new_fh.close()
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_parallel_ok_logging_and_completion(tmp_path, monkeypatch, env, stop_flag_reset):
    """3 good items -> all ok, valid nii on disk, worker lines reach parent log."""
    dcm_log = os.environ[MRI_FAKE_DCM_LOG]
    cmds = []
    for i in range(3):
        cmd, _ = _make_item(i, 'good', tmp_path / 'nifti')
        cmds.append(cmd)

    results = mod.run_with_progress(_partial(mod.run_cmd, commands=cmds), cmds, True)

    # Every result is a dict with status ok (or a lost worker -> drop to dict check)
    assert len(results) == 3, f"expected 3 results, got {len(results)}: {results!r}"
    assert all(isinstance(r, dict) and r.get('status') == 'ok' for r in results), \
        f"expected all ok: {results!r}"

    # 3 valid .nii.gz on disk, readable.
    for i in range(3):
        p = f'{mod.SAVE_DIR}S{i:03d}/00.nii.gz'
        assert os.path.exists(p), f'missing output {p}'
        img = nib.load(p)
        assert max(img.shape) >= 4
        _ = img.get_fdata()

    # Parent log file received worker lines (proves the mp log-queue wiring).
    text = _poll_log(lambda t: t.count('[DONE] 00') >= 3 and '[START] 00' in t)
    assert text is not None, 'parent log never received [DONE] 00 x3 + [START] 00'
    assert text.count('[DONE] 00') >= 3, f"[DONE] 00 count = {text.count('[DONE] 00')}"
    assert text.count('[START] 00') >= 3, f"[START] 00 count = {text.count('[START] 00')}"

    # Parent-side bookkeeping: remove all 3, then finalize -> no checkpoint file.
    removed = mod._remove_completed(cmds, results, 't1')
    assert removed == 3, f'expected 3 removed, got {removed} (remaining {len(cmds)})'
    assert len(cmds) == 0
    mod._finalize_progress(cmds)
    assert not os.path.exists(f'{mod.LOAD_DIR}saveNifti_progress.pkl'), \
        'checkpoint file should not exist after a clean run'


def test_resume_skips_existing(tmp_path, monkeypatch, env, stop_flag_reset):
    """Pre-existing valid S001/00.nii.gz -> S001 skipped, dcm2niix not re-invoked."""
    dcm_log = os.environ[MRI_FAKE_DCM_LOG]
    # Pre-create a valid S001 output (resume path).
    _write_nii(f'{mod.SAVE_DIR}S001/00.nii.gz')

    cmd0, inp0 = _make_item(0, 'good', tmp_path / 'nifti')
    cmd1, inp1 = _make_item(1, 'good', tmp_path / 'nifti')
    cmds = [cmd0, cmd1]

    results = mod.run_with_progress(_partial(mod.run_cmd, commands=cmds), cmds, True)
    assert len(results) == 2, f"expected 2 results, got {results!r}"

    r0 = next((r for r in results if r.get('session') == 'S000'), None)
    r1 = next((r for r in results if r.get('session') == 'S001'), None)
    assert r0 is not None and r1 is not None, f"missing S000/S001 results: {results!r}"
    assert r1.get('status') == 'skipped', f"S001 status = {r1.get('status')}"
    assert r0.get('status') == 'ok', f"S000 status = {r0.get('status')}"

    # dcm2niix must NOT have been re-invoked for S001, but WAS for S000.
    assert not _log_lines_containing(dcm_log, inp1), \
        f'dcm2niix unexpectedly re-invoked for S001: {open(dcm_log).read()!r}'
    assert _log_lines_containing(dcm_log, inp0), 'dcm2niix was not invoked for S000'


def test_disk_threshold_stops_all(tmp_path, monkeypatch, env, stop_flag_reset):
    """Impossibly high disk threshold: check_disk_space fails -> all stopped."""
    dcm_log = os.environ[MRI_FAKE_DCM_LOG]
    monkeypatch.setattr(mod, 'DISK_SPACE_THRESHOLD', 2 ** 62)

    cmds = []
    for i in range(3):
        cmd, _ = _make_item(i, 'good', tmp_path / 'nifti')
        cmds.append(cmd)

    results = mod.run_with_progress(_partial(mod.run_cmd, commands=cmds), cmds, True)
    assert len(results) == 3, f"expected 3 results, got {results!r}"
    assert all(isinstance(r, dict) and r.get('status') == 'stopped' for r in results), \
        f"expected all stopped: {results!r}"

    # stop flag set by the abort path.
    assert mod.stop_flag.is_set(), 'stop flag should be set after disk abort'

    # Zero dcm2niix invocations (checks failed before the subprocess).
    n = _calls(dcm_log)
    assert n == 0, f'expected 0 dcm2niix calls, got {n}'

    # No output written.
    for i in range(3):
        assert not os.path.exists(f'{mod.SAVE_DIR}S{i:03d}/00.nii.gz')

    # Bookkeeping: nothing removed; checkpoint retains all 3 commands.
    removed = mod._remove_completed(cmds, results, 't1')
    assert removed == 0, f'expected 0 removed, got {removed}'
    assert len(cmds) == 3
    mod._finalize_progress(cmds)
    cp = f'{mod.LOAD_DIR}saveNifti_progress.pkl'
    assert os.path.exists(cp), 'checkpoint file should exist after a stopped run'
    with open(cp, 'rb') as fh:
        stored = pickle.load(fh)
    assert stored == cmds, 'checkpoint should contain exactly the 3 remaining commands'


def test_in_flight_completes_queued_stop(tmp_path, monkeypatch, env, stop_flag_reset):
    """A worker already inside dcm2niix finishes with a valid nii, while a later
    worker (stuck in its pre-flight source check) is stopped.  Deterministic via
    timing: slow check (5s) on the poison item vs fast check (1s)+2s dcm2niix on
    the good item."""
    dcm_log = os.environ[MRI_FAKE_DCM_LOG]

    # S000: slow_good -> fast check (1s) then dcm2niix sleeps 2s, writes valid nii.
    cmd0, inp0 = _make_item(0, 'slow', tmp_path / 'nifti')
    # S001: poison dir -> source check blocks 5s and returns False -> stopped.
    cmd1, inp1 = _make_item(1, 'slow', tmp_path / 'nifti', src_name='poison_src')

    monkeypatch.setattr(mod.args, 'cpus', 2)

    def slow_check(path):
        time.sleep(5 if 'poison' in path else 1)
        return 'poison' not in path

    monkeypatch.setattr(mod, 'check_source_files', slow_check)

    cmds = [cmd0, cmd1]
    results = mod.run_with_progress(_partial(mod.run_cmd, commands=cmds), cmds, True)
    assert len(results) == 2, f"expected 2 results, got {results!r}"

    r0 = next((r for r in results if r.get('session') == 'S000'), None)
    r1 = next((r for r in results if r.get('session') == 'S001'), None)
    assert r0 is not None and r1 is not None, f"missing S000/S001 results: {results!r}"
    assert r0.get('status') == 'ok', f"S000 status = {r0.get('status')}"
    assert r1.get('status') == 'stopped', f"S001 status = {r1.get('status')}"

    # Valid nii written by the in-flight worker.
    p = f'{mod.SAVE_DIR}S000/00.nii.gz'
    assert os.path.exists(p), f'missing in-flight output {p}'
    _ = nib.load(p).get_fdata()

    # Exactly one dcm2niix invocation (the in-flight S000).
    n = _calls(dcm_log)
    assert n == 1, f'expected exactly 1 dcm2niix call, got {n}'
    assert _log_lines_containing(dcm_log, inp0), 'S000 dcm2niix call not logged'
    assert not _log_lines_containing(dcm_log, inp1), 'S001 should not have invoked dcm2niix'

    # The abort reason was logged.
    text = _poll_log(lambda t: ('[ABORT]' in t) or ('Disk/source unavailable' in t))
    assert text is not None, 'expected an [ABORT]/Disk-source-unavailable log line'

    assert mod.stop_flag.is_set(), 'stop flag should be set after source abort'

    # Bookkeeping: only S001 remains; checkpoint has exactly 1 command.
    removed = mod._remove_completed(cmds, results, 't1')
    assert removed == 1, f'expected 1 removed (S000), got {removed}'
    assert len(cmds) == 1, f'expected 1 command remaining, got {len(cmds)}'
    assert cmds[0] == cmd1, 'remaining command should be the stopped S001 command'

    mod._finalize_progress(cmds)
    cp = f'{mod.LOAD_DIR}saveNifti_progress.pkl'
    assert os.path.exists(cp), 'checkpoint file should exist after a stopped run'
    with open(cp, 'rb') as fh:
        stored = pickle.load(fh)
    assert stored == [cmd1], f'checkpoint should contain exactly 1 command, got {stored!r}'


def test_timeout_kills_dcm2niix(tmp_path, monkeypatch, env, stop_flag_reset):
    """A long-running dcm2niix hits the timeout, is killed, and leaves no orphan."""
    dcm_log = os.environ[MRI_FAKE_DCM_LOG]
    fake_exe = _write_fake_dcm2niix(tmp_path)
    monkeypatch.setattr(mod, 'DCM2NIIX_TIMEOUT', 2)

    cmd, inp = _make_item(0, 'sleep', tmp_path / 'nifti')
    cmds = [cmd]

    t0 = time.monotonic()
    results = mod.run_with_progress(_partial(mod.run_cmd, commands=cmds), cmds, True)
    elapsed = time.monotonic() - t0
    assert len(results) == 1, f"expected 1 result, got {results!r}"
    r = results[0]
    assert isinstance(r, dict) and r.get('status') == 'failed', f"expected failed, got {r!r}"
    assert 'timeout' in (r.get('error') or ''), f"error should mention timeout: {r.get('error')!r}"
    # The whole thing should be bounded near the 2s timeout (not the 120s sleep).
    assert elapsed < 30, f'test took too long ({elapsed:.1f}s) — fake not killed in time'

    # Exactly one dcm2niix invocation.
    n = _calls(dcm_log)
    assert n == 1, f'expected exactly 1 dcm2niix call, got {n}'
    assert _log_lines_containing(dcm_log, inp), 'sleep dcm2niix call not logged'

    # [TIMEOUT] line logged.
    text = _poll_log(lambda t: '[TIMEOUT]' in t)
    assert text is not None, 'expected a [TIMEOUT] log line'

    # No orphaned fake dcm2niix process remains (poll /proc).
    needle = os.path.abspath(fake_exe)

    def _find_orphan():
        for entry in os.listdir('/proc'):
            if not entry.isdigit():
                continue
            try:
                with open(f'/proc/{entry}/cmdline', 'rb') as fh:
                    data = fh.read()
            except Exception:
                continue
            cmd = data.decode('latin-1', errors='replace')
            if needle in cmd or 'dcm2niix' in cmd and 'python' in cmd:
                return entry
        return None

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if _find_orphan() is None:
            break
        time.sleep(0.2)
    assert _find_orphan() is None, f'orphaned fake dcm2niix process still alive: {_find_orphan()}'

    # Partial output cleaned up.
    assert not os.path.exists(f'{mod.SAVE_DIR}S000/00.nii.gz'), \
        'partial .nii.gz should have been cleaned up after timeout'
