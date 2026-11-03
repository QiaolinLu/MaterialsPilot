"""Bound concurrent MLIP work independently of API request concurrency."""
from contextlib import contextmanager
import ctypes
import fcntl
import gc
import math
import os
from pathlib import Path
import time


_last_resource_failure = None


class MLIPResourceError(RuntimeError):
    """An infrastructure failure, never a measured material-quality failure."""

    def __init__(self, message):
        super().__init__(message)
        # The runner may catch model/critic exceptions for ordinary user-facing
        # diagnostics. Preserve resource identity for the durable worker too.
        global _last_resource_failure
        _last_resource_failure = self


def reset_resource_failure():
    global _last_resource_failure
    _last_resource_failure = None


def get_resource_failure():
    return _last_resource_failure


def _release_unused_cpu_memory():
    gc.collect()
    # NumPy/PyTorch CPU allocations may otherwise accumulate in every API
    # worker. This releases only free allocator pages, never live model data.
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        pass  # Non-glibc platforms do not provide this optional reclamation.


@contextmanager
def mlip_resource_slot():
    """Acquire one process-owned lock slot; crashes automatically release it.

    The optional lock directory belongs to an experiment, outside its frozen
    runtime. With no directory configured, ordinary library use is unchanged.
    """
    directory = os.getenv("MATERIALSPILOT_MLIP_SLOT_DIR")
    if not directory:
        yield
        return
    try:
        count = int(os.getenv("MATERIALSPILOT_MLIP_MAX_CONCURRENT", "4"))
        timeout = float(os.getenv("MATERIALSPILOT_MLIP_SLOT_TIMEOUT_SECONDS", "600"))
        if count < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("MLIP concurrency and slot timeout must be positive")
    except ValueError as exc:
        raise MLIPResourceError(f"Invalid MLIP resource configuration: {exc}") from exc
    root = Path(directory)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MLIPResourceError(f"Cannot initialize MLIP resource slots: {exc}") from exc
    deadline = time.monotonic() + timeout
    handle = None
    while handle is None:
        for offset in range(count):
            try:
                candidate = (root / f"slot_{(os.getpid() + offset) % count}.lock").open("a")
            except OSError as exc:
                raise MLIPResourceError(f"Cannot open MLIP resource slot: {exc}") from exc
            try:
                fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle = candidate
                break
            except BlockingIOError:
                candidate.close()
            except OSError as exc:
                candidate.close()
                raise MLIPResourceError(f"Cannot lock MLIP resource slot: {exc}") from exc
        if handle is None:
            if time.monotonic() >= deadline:
                raise MLIPResourceError(f"Timed out waiting {timeout:g}s for an MLIP resource slot")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            _release_unused_cpu_memory()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
