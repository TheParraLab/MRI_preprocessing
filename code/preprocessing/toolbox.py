import time
import logging
import os
import fcntl
import queue
import atexit as _atexit
import sys

from typing import Callable, List, Any, Optional
from functools import partial
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from logging.handlers import QueueHandler, QueueListener


# ---- Module state ----------------------------------------------------------
_listener_registry: dict[str, QueueListener] = {}


def _stop_all_listeners() -> None:
    """Flush + stop every listener started by get_logger."""
    for lst in list(_listener_registry.values()):
        try:
            # Drain pending records then unblock the consumer thread.
            lst.flush()
            lst.enqueue_sentinel()
        except (RuntimeError, OSError):
            pass                     # interpreter tearing down already


_atexit.register(_stop_all_listeners)


# ---- Handlers --------------------------------------------------------------

class FileHandlerWithLock(logging.FileHandler):
    """File handler with per-emit advisory lock for child processes.

    Used when multiple **processes** (ProcessPoolExecutor workers) write to the
    same log file concurrently.  Each emit() opens its own handle, acquires an
    exclusive flock(), writes, then closes — so no shared mutable stream state."""

    def __init__(self, filename: str, mode: str = 'a', encoding: Optional[str] = None):
        super().__init__(filename, mode, encoding, delay=True)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        with open(self.baseFilename, self.mode, encoding=self.encoding) as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(msg + self.terminator)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


# ---- Child-process initialiser ---------------------------------------------

def _init_child_logger(
    logger_name: str,
    logger_level: int,
    file_path: str,
    formatter_str: str,
) -> None:
    """Called once per spawned child process.

    Installs a direct FileHandlerWithLock (no queue needed in an isolated process)."""
    lgr = logging.getLogger(logger_name)
    lgr.handlers.clear()
    lgr.setLevel(logger_level)
    lgr._log_level = logger_level          # so run_function can read it back.

    fmt = logging.Formatter(formatter_str)
    fh = FileHandlerWithLock(file_path, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    lgr.addHandler(fh)


# ---- Process worker wrapper ------------------------------------------------

def _process_worker(target: Callable[..., Any], item: Any, *args: Any, **kwargs: Any):
    """Top-level callable submitted to ProcessPoolExecutor."""
    return target(item, *args, **kwargs)


# ---- Logger proxy (drop-in replacement for a raw logging.Logger) -----------

class _LoggerProxy(logging.Logger):
    """Wraps a logging.Logger so that attribute access is forwarded.

    Allows us to stash extra attributes (_log_level, _file_path, etc.) without
    polluting the global Logger class — but callers never notice: they still
    have ``LOGGER.debug(...)`` working exactly as before."""

    def __init__(self, logger: logging.Logger):
        # Stash a reference we can reach via __getattr / __setattr__.
        object.__setattr__(self, '_wrapped', logger)
        # Copy over instance-level attrs so that the underlying loggers are
        # independent if get_logger() is called twice with a previously-unseen name.

    def _fwd(self: logging.Logger, *a: Any, **kw: Any) -> None: ...  # type: ignore[override]

    def debug(self, msg: str, *args: Any, **kwargs: Any):
        self._wrapped.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any):
        self._wrapped.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any):
        self._wrapped.warning(msg, *args, **kwargs)

    def warn(self, msg: str, *args: Any, **kwargs: Any):
        self._wrapped.warn(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any):
        self._wrapped.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any):
        self._wrapped.exception(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any):
        self._wrapped.critical(msg, *args, **kwargs)

    def fatal(self, msg: str, *args: Any, **kwargs: Any):
        self._wrapped.fatal(msg, *args, **kwargs)

    # ---- attribute delegation -----------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return object.__getattribute__(self, '_wrapped').__getattribute__(name)

    def __setattr__(self, name: str, value: Any):
        if name == "_wrapped":
            super().__setattr__(name, value)
        else:
            object.__getattribute__(self, '_wrapped').__setattr__(name, value)


# ---- Public API ------------------------------------------------------------

def get_logger(name: str, save_dir: str = '') -> _LoggerProxy:
    """Create a logger that is fast under high concurrency.

    The hot-path from *every* producer thread / process is an expensive-free
    ``queue.put(record)`` call to our :class:`~logging.handlers.QueueHandler`.  A
    single daemon consumer drains the queue and does all file + stream I/O
    sequentially — meaning zero per-emit lock contention."""

    if save_dir:
        if save_dir[-1] != '/':
            save_dir += '/'
        os.makedirs(save_dir, exist_ok=True)

    log_level = logging.DEBUG
    formatter_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_path = save_dir + name + '.log'

    # --- underlying Logger (managed by Python's logging system) ---------------
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(log_level)

    fmt = logging.Formatter(formatter_str)

    # Consumer-side handlers written to sequentially ----------
    fh_file = FileHandlerWithLock(file_path, mode='a')
    fh_file.setLevel(logging.DEBUG)
    fh_file.setFormatter(fmt)

    ch_stream = logging.StreamHandler()
    ch_stream.setLevel(logging.INFO)
    ch_stream.setFormatter(fmt)

    # Producer-side QueueHandler (cheap put only) -----------------
    log_queue: 'queue.Queue[logging.LogRecord]' = queue.Queue(-1)
    qh = QueueHandler(log_queue)
    logger.addHandler(qh)

    listener = QueueListener(
        log_queue, fh_file, ch_stream,
        respect_handler_level=True,
    )
    # Non-daemon so interpreter waits for it -> flushes pending records.
    listener.daemon_threads = False          # type: ignore[attr-defined]
    try:
        listener._thread.daemon = False      # explicit flag for older stdlib versions
    except AttributeError:
        pass

    if hasattr(listener, '_stopper'):              # Python ≥3.12 renamed the internal attr
        pass                                    # already handled
    elif not getattr(listener, 'daemon_threads', True):  # type: ignore[attr-defined]
        _listener_registry[name] = listener    # register for atexit flush + stop

    logger._log_level = logging.DEBUG
    logger._file_path = os.path.abspath(file_path) if file_path else ''
    logger._formatter_str = formatter_str

    ctx = _LoggerProxy(logger)
    return ctx


# ---- Parallel runner -------------------------------------------------------

def run_function(
    LOGGER: Any,                          # can be a Logger or _LoggerProxy
    target: Callable[..., Any], items: List[Any],
    Parallel: bool = True, P_type: str = 'thread', N_CPUS: int = 0,
    stop_flag: Optional[object] = None, *args: Any, **kwargs: Any,
) -> List[Any]:
    """Run a function over *items* in parallel or sequentially.

    Args:
        LOGGER (:class:`logging.Logger`): Logger for diagnostic output.
        target (Callable[..., Any]): Worker function. First argument receives the item.
            In thread / sequential mode logger is passed via closure or global state;
            under process mode child processes receive their own freshly initialised logger
            (we must NOT send LOGGER across a pickle boundary).
        items (List[Any]): Items to feed into *target* one by one.
        Parallel (bool): Whether to dispatch in parallel at all (False → serial loop).
        P_type (str): ``'thread'`` or ``'process'``.  Anything else falls back to serial.
        N_CPUS (int): Suggested worker count; 0 means "best auto-guess".

    Returns:
        List[Any]: Results in the same order as *items*.  If every result is a tuple,
        returns ``list(zip(*results))`` for backwards compatibility."""

    target_name = target.func.__name__ if isinstance(target, partial) else target.__name__

    def _effective_cpus(n: int) -> int:
        total = cpu_count() - 1
        return n if n > 0 else max(total, 1)

    N_CPUS = _effective_cpus(N_CPUS)

    LOGGER.debug(f'Running {target_name} {" in parallel" if Parallel else "serially"}')
    LOGGER.debug(f'Number of items: {len(items)}')

    results: List[Any] = []
    try:
        # ───────── process mode ─────────
        if Parallel and P_type == 'process':
            max_workers = min(32, 2 * N_CPUS)
            LOGGER.debug(f'Using {P_type} workers={max_workers}')
            init_args = (LOGGER.name, LOGGER._log_level,
                         LOGGER._file_path, LOGGER._formatter_str)

            with ProcessPoolExecutor(max_workers=max_workers,
                                     initializer=_init_child_logger,
                                     initargs=init_args) as executor:
                future_map = {executor.submit(_process_worker, target, item, *args, **kwargs): i
                              for i, item in enumerate(items)}
                ordered: List[Optional[Any]] = [None] * len(future_map)

                for fut in as_completed(future_map):
                    idx = future_map.pop(fut)
                    if stop_flag and getattr(stop_flag, 'is_set', lambda: False)():
                        LOGGER.info('Stopping parallel processing (stop flag).')
                        break
                    try:
                        result = fut.result()         # fast path for already-completed work
                        ordered[idx] = result
                        LOGGER.debug(f'Future {idx} completed successfully')
                    except KeyboardInterrupt:
                        LOGGER.error('KeyboardInterrupt received. Stopping processing.')
                        if stop_flag and getattr(stop_flag, 'set', None):
                            stop_flag.set()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception as e:
                        LOGGER.error(
                            f'Error parallel processing item {idx}: {e}', exc_info=True)
                        ordered[idx] = None

                results = list(ordered)

        # ───────── thread mode ────────────────
        elif Parallel and P_type == 'thread':
            max_workers = min(32, 2 * N_CPUS)
            LOGGER.debug(f'Using {P_type} workers={max_workers}')
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(target, item, *args, **kwargs): i
                              for i, item in enumerate(items)}
                ordered = [None] * len(future_map)

                for fut in as_completed(future_map):
                    idx = future_map.pop(fut)
                    if stop_flag and getattr(stop_flag, 'is_set', lambda: False)():
                        LOGGER.info('Stopping parallel processing (stop flag).')
                        break
                    try:
                        result = fut.result()
                        ordered[idx] = result
                        LOGGER.debug(f'Future {idx} completed successfully')
                    except KeyboardInterrupt:
                        LOGGER.error('KeyboardInterrupt received. Stopping processing.')
                        if stop_flag and getattr(stop_flag, 'set', None):
                            stop_flag.set()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception as e:
                        LOGGER.error(f'Error parallel processing item {idx}: {e}', exc_info=True)
                        ordered[idx] = None

                results = list(ordered)

        # ───────── fallback serial ─────────────
        else:
            if Parallel and P_type not in ('thread', 'process'):
                LOGGER.error(f'Unknown P_type={P_type}, falling back to serial.')
            for i, item in enumerate(items):
                if stop_flag and getattr(stop_flag, 'is_set', lambda: False)():
                    break
                try:
                    results.append(target(item, *args, **kwargs))
                except Exception as exc:
                    LOGGER.exception(f'Error at index {i}')

    except KeyboardInterrupt:
        LOGGER.error('KeyboardInterrupt received. Stopping processing.')
        if stop_flag and getattr(stop_flag, 'set', None):
            stop_flag.set()
    finally:
        LOGGER.debug(f'Completed {target_name} {" in parallel" if Parallel else "serially"}')
        LOGGER.debug(f'Number of results: {len(results)}')

    # Backwards compat with workers returning (list, dict) tuples.
    if results and isinstance(results[0], tuple):
        return list(zip(*results))
    return results


# ---- Progress bar ----------------------------------------------------------

class ProgressBar:
    def __init__(self, total, splits=20, update_interval=1):
        self.total = total
        self.splits = splits
        self.current = 0
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update(index=0)

    def update(self, index = None, status=''):
       # with self.lock:
        if index is None:
            index = self.current + 1

        if index % self.update_interval != 0 and index != self.total:
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if index > 0:
            avg_time_per_step = elapsed_time / index
            remaining_steps = self.total - index
            eta = avg_time_per_step * remaining_steps
        else:
            eta = 0

        current = int((index / self.total) * self.splits)
        current_progress = ''
        for i in range(self.splits):
            if i < current:
                current_progress += '■'
            else:
                current_progress += '□'

        eta_formatted = self.format_time(eta)
        print(f'\r {current_progress} | {index}/{self.total} | {status} | ETA: {eta_formatted} |', end='', flush=True)
        self.current = index

    @staticmethod
    def format_time(seconds):
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        return f'{hours:02}:{mins:02}:{secs:02}'

# Example usage
if __name__ == '__main__':
    import random
    total_steps = 100
    progress_bar = ProgressBar(total_steps)

    for i in range(total_steps):
        time.sleep(random.random()/2)  # Simulate work
        progress_bar.update(i + 1, status='Processing')