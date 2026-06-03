"""
Tests for code/preprocessing/toolbox.py -- logger, parallel runner, progress bar.

Verifies correctness and performance characteristics after any change to the
queue-based logging infrastructure and parallel execution helpers.

Running
-------
::

    pytest test/test_toolbox.py -v

Test matrix
-----------
+-------------------------------------+------------------------------------------+
| Test                                | Validates                                |
+-------------------------------------+------------------------------------------+
| ``test_get_logger_returns_proxy``   | Return type is _LoggerProxy              |
| ``test_logger_all_levels``          | debug/info/warning/error/critical emit   |
| ``test_logger_writes_to_file``      | File output contains logged messages     |
| ``test_logger_debug_in_file``       | File handler accepts DEBUG-level records |
| ``test_proxy_attribute_access``     | Custom attrs on proxy forward to logger  |
| ``test_proxy_setattr_getattr``      | Stashed attrs are readable               |
| ``test_file_handler_lock_emit``      | FileHandlerWithLock writes to file       |
| ``test_run_function_serial``        | Serial execution returns ordered results |
| ``test_run_function_thread``        | Thread pool preserves result order       |
| ``test_run_function_process``       | Process pool preserves result order      |
| ``test_run_function_empty_items``   | Empty input -> empty output              |
| ``test_run_function_partial_target`` | partial-wrapped target works            |
| ``test_run_function_tuple_results`` | Tuple results unzipped backwards compat  |
| ``test_progress_bar_init``          | ProgressBar state on construction        |
| *Logging integrity tests*           | No dupes in serial/thread/process mode   |
| *Performance tests*                | Throughput under concurrent logging      |
"""

import logging as _logging_module
import os
import sys
import time
import tempfile
import threading

from pathlib import Path
from functools import partial


# ---- Module loading --------------------------------------------------------
# Direct import (not via importlib.util) so that internal functions such as
# ``_process_worker`` live in a proper module namespace and are picklable for
# ProcessPoolExecutor workers.

proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))
from toolbox import (                            # noqa: E402
    FileHandlerWithLock,
)                                                # noqa: E402


def _wait_and_stop_listener(name):
    """Flush and stop the QueueListener registered for *name*."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    reg = _tb._listener_registry
    if name in reg:
        listener = reg.pop(name)
        try:
            listener.stop()
        except (RuntimeError, AttributeError):
            pass                                 # thread may never have started


# ---- Logger creation tests -------------------------------------------------

def test_get_logger_returns_proxy(tmp_path):
    """Return type is the custom logger proxy."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    lgr = _tb.get_logger("test_proxy", save_dir=str(tmp_path))
    assert isinstance(lgr, _tb._LoggerProxy)


def test_logger_all_levels(tmp_path):
    """debug/info/warning/error/critical all emit without raising."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "test_levels"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    logger.debug("d")
    logger.info("i")
    logger.warning("w")
    logger.error("e")
    logger.critical("c")
    time.sleep(0.3)
    _wait_and_stop_listener(name)


def test_logger_writes_to_file(tmp_path):
    """File output contains logged messages."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "test_fileoutput"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    logger.info("HELLO_FILE_OUTPUT")
    time.sleep(0.5)
    _wait_and_stop_listener(name)

    log_file = tmp_path / f"{name}.log"
    assert log_file.exists(), "Log file was never created by the listener thread."
    contents = log_file.read_text()
    assert "HELLO_FILE_OUTPUT" in contents


def test_logger_debug_in_file(tmp_path):
    """File handler is DEBUG level so it captures debug records too."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "test_debug_capture"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    logger.debug("DEBUG_RECORD_HERE")
    time.sleep(0.5)
    _wait_and_stop_listener(name)

    log_file = tmp_path / f"{name}.log"
    assert log_file.exists()
    contents = log_file.read_text()
    assert "DEBUG_RECORD_HERE" in contents


def test_proxy_attribute_access(tmp_path):
    """Proxy forwards attribute access to underlying logger."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "test_attr_fwd"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    time.sleep(0.3)
    _wait_and_stop_listener(name)

    assert logger.name == name


def test_proxy_setattr_getattr(tmp_path):
    """Stashed attrs like _log_level, _file_path are readable."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "test_attrs"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    assert hasattr(logger, "_log_level")
    assert logger._log_level == _logging_module.DEBUG
    assert hasattr(logger, "_file_path")
    assert isinstance(logger._file_path, str)
    assert len(logger._file_path) > 0
    time.sleep(0.3)
    _wait_and_stop_listener(name)


# ---- FileHandlerWithLock tests ---------------------------------------------

def test_file_handler_lock_emit(tmp_path):
    """FileHandlerWithLock writes formatted record to file."""
    fh = FileHandlerWithLock(str(tmp_path / "locktest.log"))
    fmt = _logging_module.Formatter("%(message)s")
    fh.setFormatter(fmt)

    record = _logging_module.LogRecord(
        name="test", level=_logging_module.INFO, pathname="", lineno=0,
        msg="LOCK_WORKS", args=None, exc_info=None
    )
    fh.emit(record)
    fh.flush()

    text = Path(tmp_path / "locktest.log").read_text()
    assert "LOCK_WORKS" in text


def test_file_handler_concurrent_emits(tmp_path):
    """Multiple threads writing via FileHandlerWithLock don't corrupt file."""
    log_file_str = str(tmp_path / "concurrent.locked.log")
    fh = FileHandlerWithLock(log_file_str)
    fmt = _logging_module.Formatter("%(message)s")
    fh.setFormatter(fmt)

    n_records = 200

    def _worker(tid):
        for i in range(n_records):
            record = _logging_module.LogRecord(
                name="test", level=_logging_module.INFO, pathname="", lineno=0,
                msg=f"MSG_{tid}_{i}", args=None, exc_info=None
            )
            fh.emit(record)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(4)]
    for thr in threads:
        thr.start()
    for thr in threads:
        thr.join()

    total_expected = n_records * 4
    text = Path(log_file_str).read_text()
    actual_lines = len([l for l in text.strip().split("\n") if l])
    assert actual_lines == total_expected, (
        f"Expected {total_expected} lines but got {actual_lines}"
    )

    # Every unique message appears exactly once.
    lines = [l for l in text.strip().split("\n") if l]
    seen: set[str] = set()
    for line in lines:
        msg = line.strip()
        assert msg not in seen, f"Duplicate message: {msg}"
        seen.add(msg)

    expected_msgs = {f"MSG_{t}_{i}" for t in range(4) for i in range(n_records)}
    assert len(seen) == len(expected_msgs), (
        f"Expected {len(expected_msgs)} unique messages but got {len(seen)}"
    )


# ---- Helper worker definitions ---------------------------------------------
# These live at module-level so that they are picklable.

def _worker_double(x):
    """Picklable: return x*2."""
    time.sleep(0.02)
    return x * 2


def _worker_triple(x):
    """Picklable: return x*3 with small delay to exercise ordering."""
    time.sleep(0.01)
    return x * 3


def _worker_square(x):
    """Picklable: return x**2 (used for process-pool test)."""
    return x ** 2


# Worker that logs inside the thread (mimics real pipeline usage in 01_scanDicom).
# The logger is passed via partial kwarg, matching how _find_dicom_worker receives it.
def _logging_thread_worker(item, logger):
    """Worker that produces a distinct log line per item."""
    logger.info(f"THREAD_LOG_{item}")
    return item * 2


# ---- run_function tests -----------------------------------------------------

def test_run_function_serial():
    """Serial execution returns results in order."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    lgr = _logging_module.getLogger("serial_test")
    items = list(range(10))
    results = _tb.run_function(lgr, lambda x: x * 2, items, Parallel=False)
    assert results == [i * 2 for i in range(10)]


def test_run_function_thread(tmp_path):
    """Thread pool preserves result order."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "thread_order_test"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    items = list(range(20))

    results = _tb.run_function(logger, _worker_triple, items, Parallel=True, P_type="thread")
    assert len(results) == 20
    assert results == [i * 3 for i in range(20)]
    time.sleep(0.3)
    _wait_and_stop_listener(name)


def test_run_function_process(tmp_path):
    """Process pool preserves result order."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "process_order_test"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    items = list(range(15))

    results = _tb.run_function(logger, _worker_square, items, Parallel=True, P_type="process")
    assert len(results) == 15
    expected = [i ** 2 for i in range(15)]
    assert results == expected
    time.sleep(0.3)
    _wait_and_stop_listener(name)


def test_run_function_empty_items(tmp_path):
    """Empty input yields empty output."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "empty_test"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    results = _tb.run_function(logger, lambda x: x, [], Parallel=False)
    assert results == []
    time.sleep(0.3)
    _wait_and_stop_listener(name)


def test_run_function_partial_target(tmp_path):
    """Partial-wrapped target works and logs correct function name."""
    import toolbox as _tb                         # type: ignore[import-not-found]

    def base(a, b):
        return a + b

    wrapped = partial(base, b=10)
    name = "partial_test"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    results = _tb.run_function(logger, wrapped, [1, 2, 3], Parallel=False)
    assert results == [11, 12, 13]
    time.sleep(0.3)
    _wait_and_stop_listener(name)


def test_run_function_tuple_unzip(tmp_path):
    """Workers returning tuples get unzipped for backwards compatibility."""
    import toolbox as _tb                         # type: ignore[import-not-found]

    def worker(x):
        return (x * 2, x * 3)

    name = "unzip_test"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))
    items = [1, 2, 3]
    results = _tb.run_function(logger, worker, items, Parallel=False)
    assert len(results) == 2
    assert list(results[0]) == [2, 4, 6]
    assert list(results[1]) == [3, 6, 9]
    time.sleep(0.3)
    _wait_and_stop_listener(name)


# ---- Real-usage logging integrity tests -------------------------------------

def test_sequential_logging_no_duplication(tmp_path):
    """Sequential execution: each log message appears exactly once in file."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "seq_log_nodup"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))

    n_items = 50
    for i in range(n_items):
        logger.info(f"SEQ_{i}")
    time.sleep(1.0)
    _wait_and_stop_listener(name)

    log_file = tmp_path / f"{name}.log"
    assert log_file.exists()
    lines = [l.strip() for l in log_file.read_text().strip().split("\n") if l.strip()]
    seen: set[str] = set()
    for line in lines:
        assert line not in seen, f"Sequential duplicate found: {line}"
        seen.add(line)


def test_thread_logging_count_integrity(tmp_path):
    """Thread pool logging: total log lines match items + run_function bookkeeping."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "thread_log_integrity"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))

    n_items = 30
    worker = partial(_logging_thread_worker, logger=logger)
    results = _tb.run_function(logger, worker, list(range(n_items)), Parallel=True, P_type="thread")
    assert len(results) == n_items
    time.sleep(1.5)
    _wait_and_stop_listener(name)

    log_file = tmp_path / f"{name}.log"
    assert log_file.exists()
    lines = [l.strip() for l in log_file.read_text().strip().split("\n") if l.strip()]

    # Every worker-produced marker appears exactly once.
    markers_found: dict[str, int] = {}
    for line in lines:
        msg_part = line.split(" - ")[-1]
        for i in range(n_items):
            expected = f"THREAD_LOG_{i}"
            if msg_part == expected:
                markers_found[expected] = markers_found.get(expected, 0) + 1

    for i in range(n_items):
        marker = f"THREAD_LOG_{i}"
        count = markers_found.get(marker, 0)
        assert count == 1, f"{marker} appeared {count} times (expected exactly 1)"


def test_process_logging_count_integrity(tmp_path):
    """Process pool logging: child-process logs don't duplicate across processes."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    name = "proc_log_integrity"
    logger = _tb.get_logger(name, save_dir=str(tmp_path))

    n_items = 25
    results = _tb.run_function(logger, _worker_square, list(range(n_items)), Parallel=True, P_type="process")
    assert len(results) == n_items
    time.sleep(1.0)
    _wait_and_stop_listener(name)

    log_file = tmp_path / f"{name}.log"
    assert log_file.exists()
    lines = [l.strip() for l in log_file.read_text().strip().split("\n") if l.strip()]
    seen: set[str] = set()
    for line in lines:
        assert line not in seen, f"Process duplicate found: {line}"
        seen.add(line)


# ---- ProgressBar tests ----------------------------------------------------

def test_progress_bar_init():
    """ProgressBar initializes with correct default state."""
    import toolbox as _tb                         # type: ignore[import-not-found]
    pb = _tb.ProgressBar(total=100)
    assert pb.total == 100
    assert pb.current == 0
    assert pb.splits == 20
    assert pb.update_interval == 1


# ---- Performance / throughput tests -----------------------------------------

class TestLoggerPerformance:
    """Verify the queue-based logger maintains good throughput under load."""

    def test_sequential_log_throughput(self, tmp_path):
        """Baseline: ~5k sequential log calls should complete in < 2 s."""
        import toolbox as _tb                     # type: ignore[import-not-found]
        name = "perf_seq"
        logger = _tb.get_logger(name, save_dir=str(tmp_path))

        t0 = time.perf_counter()
        n = 5000
        for i in range(n):
            logger.debug(f"seq_{i}")
        elapsed = time.perf_counter() - t0

        # Allow listener to drain queue.
        time.sleep(1.0)
        _wait_and_stop_listener(name)

        log_file = tmp_path / f"{name}.log"
        assert log_file.exists()
        lines = log_file.read_text().count("\n")
        assert lines >= n, f"Expected >={n} lines but got {lines}"

        # 5k msgs in < 2 s is reasonable for a queue-based logger.
        assert elapsed < 2.0, f"{elapsed:.2f}s to log {n} messages -- too slow"

    def test_concurrent_thread_log_throughput(self, tmp_path):
        """Multiple threads logging concurrently should finish fast."""
        import toolbox as _tb                     # type: ignore[import-not-found]
        name = "perf_thread"
        logger = _tb.get_logger(name, save_dir=str(tmp_path))

        msgs_per_thread = 1000
        n_threads = 8
        total_msgs = msgs_per_thread * n_threads
        barrier = threading.Barrier(n_threads)

        def _log_worker(tid):
            barrier.wait()
            for i in range(msgs_per_thread):
                logger.debug(f"T{tid}_{i}")

        threads = [threading.Thread(target=_log_worker, args=(t,))
                   for t in range(n_threads)]
        t0 = time.perf_counter()
        for thr in threads:
            thr.start()
        for thr in threads:
            thr.join(timeout=15)
        elapsed = time.perf_counter() - t0

        # Allow listener to finish.
        time.sleep(1.5)
        _wait_and_stop_listener(name)

        log_file = tmp_path / f"{name}.log"
        assert log_file.exists()
        lines = log_file.read_text().count("\n")
        assert lines >= total_msgs, (
            f"Expected >= {total_msgs} lines but got {lines}"
        )

        # 8 threads x 1k msgs < 5 s.
        assert elapsed < 5.0, (
            f"{elapsed:.2f}s for {n_threads}x{msgs_per_thread} msgs -- too slow"
        )