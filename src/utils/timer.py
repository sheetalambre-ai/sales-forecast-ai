"""
Timing utilities.

Provides:
- Context manager for timing code blocks
- Decorator for timing functions
- Elapsed time retrieval
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from utils.logger import get_logger

logger = get_logger(__name__)


class Timer:
    """
    Context manager for measuring execution time.

    Example
    -------
    with Timer("Training"):
        train_model()
    """

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.info("%s started.", self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

        logger.info(
            "%s completed in %.4f seconds.",
            self.name,
            self.elapsed,
        )

        return False


def time_function(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """
    Decorator for measuring execution time.

    Example
    -------
    @time_function
    def train():
        ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        logger.info(
            "%s executed in %.4f seconds.",
            func.__name__,
            elapsed,
        )

        return result

    return wrapper


def measure_execution(
    func: Callable[..., Any],
    *args,
    **kwargs,
):
    """
    Execute a callable and return both the result
    and execution time.

    Returns
    -------
    tuple
        (result, elapsed_seconds)
    """

    start = time.perf_counter()

    result = func(*args, **kwargs)

    elapsed = time.perf_counter() - start

    return result, elapsed