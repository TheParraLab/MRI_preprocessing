"""
Tests enforcing the "a hung worker must not hang the pipeline" contract from
the 1.1 multiprocessing-freeze fix.

Locks in the guarantee that ``run_function`` / ``_run_windowed``
(code/preprocessing/toolbox.py) return in bounded time with ``None`` for any
hung item, rather than blocking forever. The per-item budget is now
``toolbox.ITEM_TIMEOUT`` (env ``MRI_ITEM_TIMEOUT`` / legacy ``MRI_WORKER_TIMEOUT``)
applied per in-flight future by ``_run_windowed`` — there is no global deadline.
"""

import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
    monkeypatch.setattr(_tb, 'ITEM_TIMEOUT', 4.0)
    # The pool's shutdown backstop (executor.shutdown(wait=True)) would otherwise
    # block up to 60 s on the still-hung worker; neutralise it so the test only
    # measures the per-item timeout contract, not the teardown backstop.
    monkeypatch.setattr(_tb, '_shutdown_bounded', lambda ex, timeout=60.0: None)
    logger = _tb.get_logger('mp_timeout_proc', save_dir=str(tmp_path))
    items = [1, 2]
    t0 = time.monotonic()
    results = _tb.run_function(logger, _slow_worker, items, Parallel=True, P_type='process')
    elapsed = time.monotonic() - t0
    assert elapsed < 25, f'run_function hung for {elapsed:.1f}s — the worker-freeze fix is broken'
    assert len(results) == 2
    assert any(r is None for r in results), 'hung items should be reported as None, not a silent hang'
    _wait_and_stop_listener('mp_timeout_proc')


def test_run_windowed_per_item_timeout(monkeypatch):
    # Direct contract test of the per-item timeout collector: one ready future,
    # one hung future. The ready item is collected; the hung item is cancelled
    # and reported as None — within its per-item budget, not a global deadline.
    ex = ThreadPoolExecutor(max_workers=2)
    f_ready = ex.submit(lambda: 7)
    f_slow = ex.submit(_slow_worker, 3)
    logger = _tb.get_logger('mp_collect_deadline', save_dir=tempfile.gettempdir())
    submit_order = iter([f_ready, f_slow])
    t0 = time.monotonic()
    ordered = _tb._run_windowed(lambda idx: next(submit_order), 2, 2, logger,
                                per_deadline=3.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 15, f'_run_windowed blocked past per-item budget ({elapsed:.1f}s)'
    assert len(ordered) == 2
    assert ordered[0] == 7, 'ready item must be collected'
    assert ordered[1] is None, 'hung item must be reported as None'
    ex.shutdown(wait=False, cancel_futures=True)
    _wait_and_stop_listener('mp_collect_deadline')


def test_run_function_serial_still_works(tmp_path, monkeypatch):
    # Serial path: no timeout guard applies (each item runs in the parent), so
    # we only assert the results are returned in order.
    logger = _tb.get_logger('mp_timeout_serial', save_dir=str(tmp_path))
    results = _tb.run_function(logger, fast, list(range(5)), Parallel=False)
    assert results == [0, 2, 4, 6, 8]
    _wait_and_stop_listener('mp_timeout_serial')
