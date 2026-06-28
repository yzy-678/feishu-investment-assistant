"""Shared provider reliability utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar

T = TypeVar("T")


class ProviderTimeoutError(TimeoutError):
    """Provider call exceeded its configured timeout."""


def run_with_timeout(
    func: Callable[[], T],
    timeout_seconds: float,
    label: str,
) -> T:
    """Run a blocking provider call with a best-effort timeout."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="provider")
    future = executor.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise ProviderTimeoutError(
            f"{label} timed out after {timeout_seconds:.1f}s"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
