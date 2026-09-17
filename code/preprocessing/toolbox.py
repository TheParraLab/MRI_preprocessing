import time
import logging
import os
import fcntl
import queue
import atexit as _atexit
import sys
import multiprocessing
import threading

from typing import Callable, List, Any, Optional, Literal
from functools import partial
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait
from logging.handlers import QueueHandler, QueueListener


# ---- Module state ----------------------------------------------------------
_listener_registry: dict[str, QueueListener] = {}


def _stop_all_listeners() -> None:
    """Flush + stop every listener started by get_logger."""
    for lst in list(_listener_registry.values()):
        try:
            lst.stop()               # drains queue then exits consumer thread
        except (RuntimeError, OSError):
            pass                     # interpreter tearing down already


_atexit.register(_stop_all_listeners)


# ---- Writable-target guard (read-only SIF defense) -------------------------

_DATA_BASE = '/FL_system/data'


def _derive_bind_hint(dir_path: str) -> str:
    """Return the ``--bind`` hint line for a failing in-container path, or
    ``''`` if the path is outside the canonical writable base.

    Paths under ``/FL_system/data/<sub>/`` map to
    ``$PWD/mri_data_base/<sub>:/FL_system/data/<sub>``.
    """
    norm = os.path.normpath(dir_path)
    base = _DATA_BASE
    if norm == base or norm.startswith(base + os.sep):
        sub = os.path.relpath(norm, base)
        if sub == '.':
            host_sub = 'mri_data_base'
            container = base
        else:
            host_sub = f'mri_data_base/{sub}'
            container = norm
        return (
            "  If running under Apptainer/Singularity manually, bind a writable scratch dir:\n"
            f'    --bind "$PWD/{host_sub}:{container}"\n'
        )
    return ''


def ensure_dir_writable(dir_path: str, context: str = 'scratch') -> None:
    """Create `dir_path` if missing; raise RuntimeError with an actionable
    message if the containing filesystem is read-only (Apptainer/Singularity
    SIF default, or a manual `apptainer run` without a writable base bind).

    The bind hint is auto-derived from the in-container path: any failing
    path under /FL_system/data/<sub>/ becomes
        --bind "$PWD/mri_data_base/<sub>:/FL_system/data/<sub>"
    Paths outside that base get a generic message.

    Callers: anywhere in the pipeline that must create a subdir under a
    user-controlled path that may land on the squashfs build layer.
    """
    if os.path.exists(dir_path):
        return
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        bind_block = _derive_bind_hint(dir_path)
        raise RuntimeError(
            f"Cannot create directory '{dir_path}' ({context}): {e}\n"
            f"  The containing filesystem is read-only.\n"
            f"{bind_block}"
            f"  or launch via start_control.sh (it binds the writable base automatically)."
        ) from e


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
    mp_log_queue: Optional[Any] = None,
) -> None:
    """Called once per spawned child process.

    If ``mp_log_queue`` is given, attach a :class:`~logging.handlers.QueueHandler`
    for it and route every log record through that queue.  The parent
    process has a :func:`_drain_mp_log_to_parent` thread consuming from
    the queue and forwarding into its own file + stream pipeline (see
    ``get_logger``).  Worker logs therefore land in the SAME file as
    parent logs, all written by one single thread in the parent —
    no N-way shared-file write contention on NFS.

    No ``StreamHandler`` is attached in this branch (would double-print
    to stdout; the parent's stream handler already surfaces the record
    once it drains from the queue).

    If ``mp_log_queue`` is ``None`` (legacy / caller that didn't pass one),
    fall back to console-only stdout for this child.
    """
    lgr = logging.getLogger(logger_name)
    lgr.handlers.clear()
    lgr.setLevel(logger_level)
    lgr._log_level = logger_level          # so run_function can read it back.
    lgr._formatter_str = formatter_str
    lgr._file_path = ''                    # child never opens its own file

    fmt = logging.Formatter(formatter_str)

    if mp_log_queue is not None:
        qh = QueueHandler(mp_log_queue)
        lgr.addHandler(qh)
    else:
        ch_stream = logging.StreamHandler()
        ch_stream.setLevel(logging.INFO)
        ch_stream.setFormatter(fmt)
        lgr.addHandler(ch_stream)

    lgr.propagate = False

    root = logging.getLogger()
    if not root.handlers:
        if mp_log_queue is not None:
            root.addHandler(QueueHandler(mp_log_queue))
        else:
            root_fh = logging.StreamHandler()
            root_fh.setLevel(logger_level)
            root_fh.setFormatter(fmt)
            root.addHandler(root_fh)


def _drain_mp_log_to_parent(mp_q: Any, parent_logger: Any, stop_event: Any) -> None:
    """Thread target. Runs in the PARENT process.

    Drains a ``multiprocessing.Queue`` of ``logging.LogRecord`` objects
    (put there by workers via ``QueueHandler``) and forwards each record
    into the parent logger's normal pipeline.  The parent already owns
    the file handler and ``QueueListener`` (see ``get_logger``), so
    forwarding via ``parent_logger.handle(record)`` ultimately writes to
    the SAME single log file the parent's own records use.

    Runs until ``stop_event`` is set AND the queue is drained (any
    records that were already queued still get processed — no silent
    log loss on shutdown).
    """
    while True:
        try:
            record = mp_q.get(timeout=0.25)
        except queue.Empty:
            if stop_event is not None and stop_event.is_set():
                break
            continue
        except (EOFError, OSError) as _e:
            # Queue closed / pipe dead — normal on shutdown
            break
        try:
            parent_logger.handle(record)
        except Exception:
            # Never let a broken log record drop the whole drain
            pass


# ---- Hybrid chunk worker (module-level so it is picklable) -----------

def _chunk_target(
    global_start: int,
    chunk_items: List[Any],
    target_fn: Callable,
    target_args: tuple,
    target_kwargs: dict,
    threads: int,
) -> tuple:
    """Work inside one ProcessPoolExecutor child.

    Must live at module-level so ProcessPoolExecutor can pickle it and ship
    it to worker processes via the ``spawn`` start method."""
    ordered: List[Optional[Any]] = [None] * len(chunk_items)

    with ThreadPoolExecutor(max_workers=threads) as inner_pool:
        fut_map = {}
        for j, item in enumerate(chunk_items):
            fut = inner_pool.submit(_process_worker, target_fn,
                                    item, *target_args, **target_kwargs)
            fut_map[fut] = j

        for fut in as_completed(fut_map):
            idx_in_chunk = fut_map.pop(fut)
            try:
                result = fut.result()
                ordered[idx_in_chunk] = result
            except Exception as e:
                root = logging.getLogger()
                root.error(
                    f'Hybrid thread error (offset {global_start+idx_in_chunk} '
                    f'in {getattr(target_fn, "__name__", "unknown")}): {e}',
                    exc_info=True,
                )
                ordered[idx_in_chunk] = None

    return global_start, ordered


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


# ---- NIfTI name helpers (shared across 03/04/05/06) -------------------
# NIfTI files are `.nii` or `.nii.gz`. dcm2niix emits plain `.nii` but
# we now pass `-z y` so it emits `.nii.gz`.  nibabel is transparent to
# either.  These helpers give every pipeline step one source of truth
# for name comparisons so `.nii` / `.nii.gz` / `00a.nii` / `00a.nii.gz`
# all collapse to the same stem ('00', '01', '00a', ...).
def nifti_stem(path: str) -> str:
    """Reduce a NIfTI basename to its stem: '01.nii'->'01',
    '05.nii.gz'->'05', '00a.nii'->'00a', '00a.nii.gz'->'00a'.
    Also strips a `_RAS` suffix so '01_RAS.nii.gz'->'01'.
    """
    name = os.path.basename(path)
    for suffix in ('.nii.gz', '.nii'):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.endswith('_RAS'):
        name = name[:-4]
    return name


def is_nifti_file(name: str) -> bool:
    """True if *name* is a NIfTI file (`.nii` or `.nii.gz`)."""
    b = os.path.basename(name)
    return b.endswith('.nii') or b.endswith('.nii.gz')


def glob_nifti(directory: str, pattern: str = '*') -> list:
    """Glob NIfTI files in *directory* matching *pattern* (prefix of stem).
    Returns an absolute sorted list of paths.  Matches both `.nii` and
    `.nii.gz`.  Example: glob_nifti(d, '*.nii') is the legacy call;
    glob_nifti(d, '*') returns every NIfTI in the directory.
    """
    import glob as _glob
    cands = (
        _glob.glob(os.path.join(directory, pattern + '.nii')) +
        _glob.glob(os.path.join(directory, pattern + '.nii.gz'))
    )
    # de-dup (in case pattern is already '*.nii' etc.) and sort by stem
    uniq = sorted({os.path.realpath(p) for p in cands}, key=lambda p: (nifti_stem(p), p))
    return uniq


# ---- Public API ------------------------------------------------------------

def resolve_dir(flag_value: Optional[str], env_name: str, default: str) -> str:
    """Resolve a deployment directory path.

    Order (first non-empty value wins):
      1. ``flag_value``  — an explicit CLI argument for that directory.
      2. ``env_name``    — an environment variable set by the launcher
         (e.g. ``RAW_DIR``, ``NIFTI_DIR``); on a native Conda HPC run the
         launcher exports the host-local path, while inside a container the
         variable is left unset so the container default applies.
      3. ``default``     — the container-stable path (``/FL_system/data/...``).

    Returns the path verbatim (trailing-slash handling is left to the caller).
    Empty strings are treated as "not provided".
    """
    if flag_value:
        return flag_value
    env_val = os.environ.get(env_name, '').strip()
    if env_val:
        return env_val
    return default


def get_log_dir() -> str:
    """Centralised log directory for the current deployment.

    Resolves in order:
      1. ``LOG_DIR`` environment variable — set by start_control.sh for every
         runtime (container-mounted ``/deployment/logs`` inside Docker and
         Singularity; a host-local deployment log dir on bare Conda HPC).
      2. Fallback for manual/local runs: ``<repo_root>/logs`` so that log
         creation never requires write access to the filesystem root.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return resolve_dir(None, 'LOG_DIR', os.path.join(repo_root, 'logs'))


def get_logger(name: str, save_dir: str = '') -> _LoggerProxy:
    """Create a logger that is fast under high concurrency.

    The hot-path from *every* producer thread / process is an expensive-free
    ``queue.put(record)`` call to our :class:`~logging.handlers.QueueHandler`.  A
    single daemon consumer drains the queue and does all file + stream I/O
    sequentially — meaning zero per-emit lock contention.

    Idempotent: if this logger already has live handlers (from _init_child_logger
    or a previous call), return the existing proxy immediately so workers calling
    get_logger() on every invocation do NOT spawn new queues or threads."""

    if not save_dir:
        save_dir = get_log_dir()
    if save_dir[-1] != '/':
        save_dir += '/'

    # A Singularity/Apptainer SIF is mounted read-only, so the log directory may
    # not be writable.  Degrade to console-only logging instead of crashing the
    # pipeline the moment the first handler tries to open the file.
    file_ok = True
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        file_ok = False
        sys.stderr.write(
            f'[WARNING] log directory {save_dir!r} is not writable ({e}). '
            f'Logging to console only for this run. '
            f'Bind a writable directory (e.g. --bind "$PWD/mri_data_base/logs:/deployment/logs") '
            f'or set LOG_DIR to a writable path.\n'
        )

    # --- underlying Logger (managed by Python's logging system) ---------------
    logger = logging.getLogger(name)

    # file_path must always be defined before either path hits it (original code
    # had this early; the idempotent guard moved inside and the ref on line 244+
    # would hit an UnboundLocalError otherwise).
    file_path = save_dir + name + '.log' if save_dir else ''

    # Idempotent guard — if handlers are already installed (either by a previous
    # get_logger() call or by _init_child_logger in a spawned worker), return
    # immediately.  Avoids creating duplicate QueueListener threads per-worker.
    if logger.handlers:
        log_level = logging.DEBUG
        formatter_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # Keep a previously-established console-only decision (empty _file_path)
        # instead of resurrecting a file path that was never writable.
        if not getattr(logger, '_file_path', '') and not file_ok:
            file_path = ''
        else:
            file_path = save_dir + name + '.log' if save_dir else ''
        logger._log_level = log_level
        logger._file_path = os.path.abspath(file_path) if file_path else ''
        logger._formatter_str = formatter_str
        return _LoggerProxy(logger)

    # Stop any existing listener for this name to prevent thread + handler leak.
    old_listener = _listener_registry.pop(name, None)
    if old_listener is not None:
        try:
            old_listener.stop()
        except (RuntimeError, OSError):
            pass

    logger.handlers.clear()
    log_level = logging.DEBUG
    formatter_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger.setLevel(log_level)

    fmt = logging.Formatter(formatter_str)
    ch_stream = logging.StreamHandler()
    ch_stream.setLevel(logging.INFO)
    ch_stream.setFormatter(fmt)

    if file_ok:
        # Use plain FileHandler for the parent QueueListener consumer path.
        # The listener drains records from a single thread, so there's only one
        # concurrent writer and we don't need per-emit flock overhead.
        fh_file = logging.FileHandler(file_path, mode='a')
        fh_file.setLevel(logging.DEBUG)
        fh_file.setFormatter(fmt)

        # Ensure the root logger also has a handler so bare `logging.error()` calls work.
        if not logging.getLogger().handlers:
            root_fh = FileHandlerWithLock(file_path, mode='a')
            root_fh.setLevel(log_level)
            root_fh.setFormatter(fmt)
            logging.getLogger().addHandler(root_fh)
    else:
        # Console-only degradation: no file handler, nothing opened.
        fh_file = None

        root = logging.getLogger()
        if not root.handlers:
            root_fh = logging.StreamHandler()
            root_fh.setLevel(log_level)
            root_fh.setFormatter(fmt)
            root.addHandler(root_fh)

    # Producer-side QueueHandler (cheap put only) -----------------
    log_queue: 'queue.Queue[logging.LogRecord]' = queue.Queue(-1)
    qh = QueueHandler(log_queue)
    logger.addHandler(qh)

    # Prevent every log line from double-writing via propagation to root handler
    logger.propagate = False

    listener = QueueListener(
        log_queue, *[h for h in (fh_file, ch_stream) if h is not None],
        respect_handler_level=True,
    )
    # Non-daemon so interpreter waits for it -> flushes pending records.
    listener.daemon_threads = False          # type: ignore[attr-defined]
    try:
        listener._thread.daemon = False      # explicit flag for older stdlib versions
    except AttributeError:
        pass

    _listener_registry[name] = listener   # unconditionally register for atexit flush

    # Unconditional registration ensures pending queue records are never lost

    listener.start()                           # begin draining the queue immediately

    logger._log_level = logging.DEBUG
    # Empty _file_path signals "console-only": run_function propagates it to
    # spawned children so they also skip constructing a file handler.
    logger._file_path = os.path.abspath(file_path) if (file_ok and file_path) else ''
    logger._formatter_str = formatter_str

    ctx = _LoggerProxy(logger)
    return ctx


# ---- Parallel runner -------------------------------------------------------

# Max seconds to wait for workers before force-terminating. Override per run via env MRI_WORKER_TIMEOUT.
import os as _os
WORKER_TIMEOUT = float(_os.environ.get('MRI_WORKER_TIMEOUT', '1800'))


def _terminate_executors(*executors) -> None:
    import signal, logging
    _log = logging.getLogger(__name__)
    for ex in executors:
        procs = getattr(ex, '_processes', None)
        if procs:
            for p in list(procs):
                try:
                    p.terminate()
                except Exception:
                    pass
            for p in list(procs):
                try:
                    p.join(timeout=3)
                    if p.is_alive():
                        p.kill()
                except Exception:
                    pass
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            _log.warning(f'Failed to shut down executor gracefully: {e!r}')


def _collect_future_map(future_map, deadline, LOGGER):
    pending = dict(future_map)
    ordered = [None] * len(future_map)
    while pending:
        remaining = max(0.0, deadline - time.monotonic())
        remap = dict()
        done, not_done = wait(list(pending.keys()), timeout=min(5.0, remaining if remaining > 0 else 0.0))
        for fut in done:
            idx = future_map[fut]
            try:
                ordered[idx] = fut.result(timeout=0.1)
            except Exception as e:
                LOGGER.error(f'Error processing item {idx}: {e}', exc_info=True)
                ordered[idx] = None
            pending.pop(fut, None)
        if time.monotonic() >= deadline:
            break
    for fut in list(pending.keys()):
        idx = future_map[fut]
        pending.pop(fut, None)
        if ordered[idx] is None:
            LOGGER.error(f'Item {idx} timed out (worker exceeded deadline) and was cancelled.')
        try:
            fut.cancel()
        except Exception:
            pass
    return ordered


def run_function(
    LOGGER: Any,                          # can be a Logger or _LoggerProxy
    target: Callable[..., Any], items: List[Any],
    Parallel: bool = True, P_type: str = 'thread', N_CPUS: int = 0, N_THREADS: int = 0,
    P_role: Literal['io', 'compute'] | None = None,
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
        P_type (str): ``'thread'``, ``'process'`` or ``'hybrid'``.  Anything else falls back to serial.
            Hyper threading mode spawns ProcessPoolExecutor workers -- each managing its own 
            ThreadPoolExecutor of size *N_THREADS* for concurrent I/O within process-scoped network address space isolation.
        N_CPUS (int): Suggested worker count; 0 means "best auto-guess".
        N_THREADS (int): Thread pool size per-hyper-worker or max workers when P_type == 'thread';
            0 uses default (2 * N_CPUS).
        P_role (str | None): ``'io'`` for I/O-bound workloads, ``'compute'`` for CPU-bound.
            I/O-bound: caps workers at min(8, cpu_count()-1) or half the available cores on larger machines.
            Compute-bound: uses full core capacity.  None falls back to legacy behavior.

    Returns:
        List[Any]: Results in the same order as *items*.  If every result is a tuple,
        returns ``list(zip(*results))`` for backwards compatibility."""

    target_name = target.func.__name__ if isinstance(target, partial) else target.__name__

    def _effective_cpus(n: int) -> int:
        total = cpu_count() - 1
        return n if n > 0 else max(total, 1)

    N_CPUS = _effective_cpus(N_CPUS)

    def _effective_workers(count: int, role: str | None = None) -> int:
        if role == 'io':
            return min(8, max(2, count // 2))
        return count

    LOGGER.debug(f'Running {target_name} {" in parallel" if Parallel else "serially"}')
    LOGGER.debug(f'Number of items: {len(items)}')

    results: List[Any] = []

    # Cross-process log routing: workers push records to _mp_q;
    # parent drain thread forwards them into the logger's file+stream pipeline.
    _use_mp_q = Parallel and P_type in ('process', 'hybrid')
    _mp_q: Optional[multiprocessing.SimpleQueue | multiprocessing.Queue] = None
    _mp_q_thread: Optional[threading.Thread] = None
    _mp_q_stop: Optional[threading.Event] = None
    if _use_mp_q:
        _mp_q = multiprocessing.Queue(maxsize=10000)
        _mp_q_stop = threading.Event()
        _mp_q_thread = threading.Thread(
            target=_drain_mp_log_to_parent,
            args=(_mp_q, LOGGER, _mp_q_stop),
            daemon=False,
        )
        _mp_q_thread.start()

    try:
        deadline = time.monotonic() + WORKER_TIMEOUT

        # ───────── process mode ─────────
        if Parallel and P_type == 'process':
            effective = _effective_workers(N_CPUS, role=P_role)
            max_workers = min(32, 2 * effective)
            LOGGER.debug(f'Using {P_type} workers={max_workers} (role={P_role})')
            init_args = (LOGGER.name, LOGGER._log_level,
                         LOGGER._file_path, LOGGER._formatter_str,
                         _mp_q)
            executor = ProcessPoolExecutor(max_workers=max_workers,
                                           initializer=_init_child_logger,
                                           initargs=init_args)
            try:
                future_map = {executor.submit(_process_worker, target, item, *args, **kwargs): i
                              for i, item in enumerate(items)}
                ordered = _collect_future_map(future_map, deadline, LOGGER)
                results = list(ordered)
            except KeyboardInterrupt:
                LOGGER.info('KeyboardInterrupt received. Cancelling queued and terminating workers...')
                _terminate_executors(executor)
                raise
            finally:
                if time.monotonic() < deadline:
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except Exception as e:
                        LOGGER.warning(f'Graceful shutdown failed: {e!r}; force-terminating')
                        _terminate_executors(executor)
                else:
                    LOGGER.error('Worker deadline exceeded — force-terminating workers.')
                    _terminate_executors(executor)
            # Tear down mp log queue: signal stop, wait for drain, close pipe
            if _use_mp_q and _mp_q is not None:
                _mp_q_stop.set()
                _mp_q_thread.join(timeout=5)
                try:
                    _mp_q.close()
                    _mp_q.join_thread()
                except Exception:
                    pass

        # ───────── thread mode ────────────────
        elif Parallel and P_type == 'thread':
            effective = _effective_workers(N_CPUS, role=P_role)
            max_workers = min(32, 2 * effective)
            LOGGER.debug(f'Using {P_type} workers={max_workers} (role={P_role})')
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                future_map = {executor.submit(target, item, *args, **kwargs): i
                              for i, item in enumerate(items)}
                ordered = _collect_future_map(future_map, deadline, LOGGER)
                results = list(ordered)
            except KeyboardInterrupt:
                LOGGER.info('KeyboardInterrupt received. Cancelling queued and terminating workers...')
                _terminate_executors(executor)
                raise
            finally:
                if time.monotonic() < deadline:
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except Exception as e:
                        LOGGER.warning(f'Graceful shutdown failed: {e!r}; force-terminating')
                        _terminate_executors(executor)
                else:
                    LOGGER.error('Worker deadline exceeded — force-terminating workers.')
                    _terminate_executors(executor)

        # ───────── hybrid: processes chunk + threads reuse I/O per-chunk ────
        elif Parallel and P_type == 'hybrid':
            # Cap total concurrency to avoid filesystem thrashing on I/O-bound DICOM scans.
            max_workers = min(16, N_CPUS)
            effective_threads = N_THREADS if N_THREADS > 0 else max(2 * max_workers, cpu_count())
            threads_per_worker = max(2, effective_threads // max(max_workers, 1))

            LOGGER.debug(f'Using {P_type}: ~{max_workers} process workers, ~{threads_per_worker} threads each')

            init_args = (LOGGER.name, LOGGER._log_level,
                         LOGGER._file_path, LOGGER._formatter_str,
                         _mp_q)

            # Create evenly-sized chunks and track global indices in parent.
            n_workers = min(max_workers, len(items)) if items else 0
            workers: List[Any] = []
            for i in range(n_workers):
                start = (i * len(items)) // n_workers
                end = ((i + 1) * len(items)) // n_workers if i < n_workers - 1 else len(items)
                chunk = items[start:end]
                if chunk:
                    workers.append((start, chunk))

            results: List[Optional[Any]] = [None] * len(items)

            if workers:
                pexecutor = ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_child_logger,
                    initargs=init_args,
                )
                try:
                    futures = [
                        pexecutor.submit(_chunk_target, start, chunk, target, args, kwargs,
                                         threads_per_worker)
                        for start, chunk in workers
                    ]
                    chunk_meta = [(start, start + len(chunk)) for start, chunk in workers]
                    future_map = dict(zip(futures, range(len(futures))))
                    ordered = _collect_future_map(future_map, deadline, LOGGER)
                    for (global_start, end_pos), chunk_result in zip(chunk_meta, ordered):
                        if not chunk_result:
                            continue
                        global_start, ordered_list = chunk_result
                        if not isinstance(ordered_list, list):
                            ordered_list = list(ordered_list)
                        for k, val in zip(range(global_start, min(global_start + len(ordered_list), len(results))),
                                         ordered_list):
                            if k < len(results):
                                results[k] = val
                except KeyboardInterrupt:
                    LOGGER.info('KeyboardInterrupt received. Cancelling queued and terminating workers...')
                    _terminate_executors(pexecutor)
                    raise
                finally:
                    if time.monotonic() < deadline:
                        try:
                            pexecutor.shutdown(wait=True, cancel_futures=True)
                        except Exception as e:
                            LOGGER.warning(f'Graceful shutdown failed: {e!r}; force-terminating')
                            _terminate_executors(pexecutor)
                    else:
                        LOGGER.error('Worker deadline exceeded — force-terminating workers.')
                        _terminate_executors(pexecutor)
            if _use_mp_q and _mp_q is not None:
                _mp_q_stop.set()
                _mp_q_thread.join(timeout=5)
                try:
                    _mp_q.close()
                    _mp_q.join_thread()
                except Exception:
                    pass

            results = list(results)

        # ───────── fallback serial ─────────────
        else:
            if Parallel and P_type not in ('thread', 'process'):
                LOGGER.error(f'Unknown P_type={P_type}, falling back to serial.')
            for i, item in enumerate(items):
                if (stop_flag and getattr(stop_flag, 'is_set', lambda: False)()) or (time.monotonic() >= deadline):
                    break
                try:
                    results.append(target(item, *args, **kwargs))
                except Exception as exc:
                    LOGGER.exception(f'Error at index {i}')

    except KeyboardInterrupt:
        LOGGER.info('KeyboardInterrupt received. In-flight workers completed, queued cancelled. Returning collected results.')
        raise
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