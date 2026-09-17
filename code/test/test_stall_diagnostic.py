"""
Tests for the self-diagnosing worker stall detector in toolbox.
Covers the /proc-based per-worker snapshot and the stall logging path in
_collect_future_map, plus the _terminate_executors fix to iterate Process objects.
"""
import os
import sys
import time
import types
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "code" / "preprocessing"))

import toolbox
from concurrent.futures import Future


class _Rec:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *a, **k):
        self.warnings.append(msg)

    def error(self, msg, *a, **k):
        pass

    def debug(self, msg, *a, **k):
        pass

    def exception(self, msg, *a, **k):
        pass


@pytest.mark.skipif(not os.path.exists('/proc/self'), reason='Linux /proc only')
def test_worker_stall_diagnostic_self():
    info = toolbox._worker_stall_diagnostic(os.getpid())
    assert info['state'] in {'R', 'S', 'D'}
    assert 'read_bytes' in info['io'] and info['io']['read_bytes'] >= 0
    assert isinstance(info['open_files'], list)


def test_collect_future_map_logs_stall_diagnostic(monkeypatch):
    monkeypatch.setattr(toolbox, 'STALL_TIMEOUT', 1.0)
    rec = _Rec()
    fut = Future()
    ex = types.SimpleNamespace(_processes={os.getpid(): None})
    deadline = time.monotonic() + 5
    res = toolbox._collect_future_map({fut: 0}, deadline, rec, executor=ex)
    assert res == [None]
    joined = '\n'.join(rec.warnings)
    assert 'stalled' in joined
    assert 'state=' in joined
    assert 'wchan=' in joined
    assert 'open_files=' in joined


def test_terminate_executors_uses_process_objects():
    class FakeProc:
        def __init__(self):
            self.terminated = False
            self.joined = False
            self.killed = False

        def terminate(self):
            self.terminated = True

        def is_alive(self):
            return False

        def join(self, timeout=None):
            self.joined = True

        def kill(self):
            self.killed = True

    proc = FakeProc()
    ex = types.SimpleNamespace(_processes={99999: proc})
    toolbox._terminate_executors(ex)
    assert proc.terminated is True
    assert proc.joined is True
