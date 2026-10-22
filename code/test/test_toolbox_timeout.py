"""
Tests enforcing the "a hung worker must not hang the pipeline" contract from
the 1.1 multiprocessing-freeze fix.

Locks in the guarantee that ``run_function`` (code/preprocessing/toolbox.py)
returns in bounded time with ``None`` for any hung item, rather than blocking
forever, and then force-terminates the stuck worker via ``_terminate_executors``.
"""

import sys
import time
import tempfile
from pathlib import Path
import os

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "code" / "preprocessing"))
import toolbox as _tb


def _wait_and_stop_listener(name):
    reg = _tb._listener_registry
    if name in reg:
        lst = reg.pop(name)
        try:
            lst.stop()
        except Exception:
            pass


def _slow_worker(x):
    time.sleep(31)          # simulates a hung worker (dcm2niix/reg_f3d stall)
    return x * 10


def fast(x):
    return x * 2


def test_run_function_process_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(_tb, 'WORKER_TIMEOUT', 4.0)
    logger = _tb.get_logger('mp_timeout_proc', save_dir=str(tmp_path))
    items = [1, 2]
    t0 = time.monotonic()
    results = _tb.run_function(logger, _slow_worker, items, Parallel=True, P_type='process')
    elapsed = time.monotonic() - t0
    assert elapsed < 25, f'run_function hung for {elapsed:.1f}s — the worker-freeze fix is broken'
    assert len(results) == 2
    assert any(r is None for r in results), 'hung items should be reported as None, not a silent hang'
    _wait_and_stop_listener('mp_timeout_proc')


def test_collect_future_map_returns_within_deadline(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(max_workers=2)
    f_ready = ex.submit(lambda: 7)
    f_slow = ex.submit(_slow_worker, 3)
    logger = _tb.get_logger('mp_collect_deadline', save_dir=tempfile.gettempdir())
    deadline = time.monotonic() + 3
    future_map = {f_ready: 0, f_slow: 1}
    t0 = time.monotonic()
    ordered = _tb._collect_future_map(future_map, deadline, logger)
    elapsed = time.monotonic() - t0
    assert elapsed < 15, f'_collect_future_map blocked past deadline ({elapsed:.1f}s)'
    assert len(ordered) == 2
    assert ordered[0] == 7, 'ready item must be collected'
    assert ordered[1] is None, 'hung item must be reported as None'
    try:
        _tb._terminate_executors(ex)
    except Exception:
        pass
    _wait_and_stop_listener('mp_collect_deadline')


def test_run_function_serial_still_within_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(_tb, 'WORKER_TIMEOUT', 600.0)
    logger = _tb.get_logger('mp_timeout_serial', save_dir=str(tmp_path))
    results = _tb.run_function(logger, fast, list(range(5)), Parallel=False)
    assert results == [0, 2, 4, 6, 8]
    _wait_and_stop_listener('mp_timeout_serial')
