"""Shared helpers for data providers."""
from __future__ import annotations

import contextlib
import functools
import logging
import signal
import socket
import threading
import time
from typing import Callable, TypeVar

log = logging.getLogger("quantkit.data")
T = TypeVar("T")

# Many upstream libraries (akshare, baostock) open sockets without a timeout; a stalled
# connection would otherwise hang an ingestion job forever.
DEFAULT_SOCKET_TIMEOUT = 30.0


def ensure_socket_timeout(seconds: float = DEFAULT_SOCKET_TIMEOUT) -> None:
    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(seconds)


def retry(times: int = 4, wait: float = 2.0, backoff: float = 2.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a flaky network call with exponential backoff; re-raise the last error."""

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            delay = wait
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - upstream raises many types
                    if attempt == times:
                        raise
                    log.warning("%s failed (%s: %s), retry %d/%d in %.0fs",
                                fn.__name__, type(exc).__name__, str(exc)[:120], attempt, times - 1, delay)
                    time.sleep(delay)
                    delay *= backoff
            raise RuntimeError("unreachable")

        return wrapper

    return deco


@contextlib.contextmanager
def hard_timeout(seconds: int, what: str = "call"):
    """Abort a blocking call after ``seconds`` (SIGALRM; main thread on Unix only).

    Some clients (BaoStock) swallow socket timeouts and can block forever; this guarantees
    an unattended ingestion job makes progress. No-op where SIGALRM is unavailable.
    """
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(signum, frame):
        raise TimeoutError(f"{what} exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
