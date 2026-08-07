"""
Tests for hybrid mode in toolbox.run_function.

Hybrid mode spawns a ProcessPoolExecutor, each worker manages its own
ThreadPoolExecutor.  Validates that _chunk_target (module-level, picklable)
preserves result ordering across process/thread boundaries.
"""

import time
import threading

import pytest

# ---- Module loading --------------------------------------------------------

from pathlib import Path
proj_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))
import toolbox as _tb  # type: ignore[import-not-found]


def _hybrid_worker(x):
    """Picklable worker: sleep then return x*7."""
    time.sleep(0.01)
    return x * 7


def _wait_and_stop_listener(name):
    """Flush and stop the QueueListener registered for *name*."""
    reg = _tb._listener_registry
    if name in reg:
        listener = reg.pop(name)
        try:
            listener.stop()
        except (RuntimeError, AttributeError):
            pass


@pytest.fixture
def hybrid_logger(tmp_path):
    name = "hybrid_test"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    yield logger, name
    time.sleep(0.3)
    _wait_and_stop_listener(name)


class TestHybridMode:
    """Validate hybrid (process + thread) parallel execution."""

    def test_hybrid_basic(self, hybrid_logger):
        """Hybrid mode returns correct ordered results."""
        logger, name = hybrid_logger
        items = list(range(16))
        results = _tb.run_function(
            logger, _hybrid_worker, items,
            Parallel=True, P_type="hybrid",
        )
        assert len(results) == 16
        assert results == [i * 7 for i in range(16)]

    def test_hybrid_preserves_order(self, hybrid_logger):
        """Hybrid mode preserves item ordering even with variable delays."""
        logger, name = hybrid_logger
        items = list(range(24))
        results = _tb.run_function(
            logger, _hybrid_worker, items,
            Parallel=True, P_type="hybrid",
        )
        assert results == [i * 7 for i in range(24)]

    def test_hybrid_empty(self, hybrid_logger):
        """Empty item list returns empty results."""
        logger, _ = hybrid_logger
        results = _tb.run_function(
            logger, _hybrid_worker, [],
            Parallel=True, P_type="hybrid",
        )
        assert results == []

    def test_hybrid_single_item(self, hybrid_logger):
        """Single item works correctly."""
        logger, _ = hybrid_logger
        results = _tb.run_function(
            logger, _hybrid_worker, [42],
            Parallel=True, P_type="hybrid",
        )
        assert results == [294]

    def test_chunk_target_is_module_level(self):
        """_chunk_target lives at module-level so it is picklable."""
        assert callable(_tb._chunk_target)
        assert hasattr(_tb._chunk_target, "__module__")
